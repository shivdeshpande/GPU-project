#include "dynamicBANG.h"
#include "utils/utils.h"
#include <cstring>
#include <cstdio>

// ============================================================================
// GPU KERNELS FOR INSERT OPERATIONS
// ============================================================================

/**
 * Copy vectors to fresh index and increment counter atomically
 */
__global__ void insertVectorsKernel(uint8_t* d_pIndex_fresh,
                                   datatype_t* d_vectors,
                                   uint32_t* d_ids,
                                   uint32_t* d_fresh_count,
                                   uint32_t batch_size,
                                   uint32_t fresh_capacity) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // Atomic increment to get insertion position
        uint32_t pos = atomicAdd(d_fresh_count, 1);

        if (pos < fresh_capacity) {
            // Copy vector to fresh index
            uint8_t* dest = d_pIndex_fresh + (pos * INDEX_ENTRY_LEN);
            datatype_t* src = d_vectors + (idx * D);

            for (uint32_t i = 0; i < D; i++) {
                ((datatype_t*)dest)[i] = src[i];
            }

            // Initialize degree to 0
            uint32_t* degree_ptr = (uint32_t*)(dest + D * sizeof(datatype_t));
            *degree_ptr = 0;

            // Store mapping (optional, for tracking)
            d_ids[idx] = pos;
        }
    }
}

/**
 * Build graph edges for inserted vectors using greedy search
 * This is simplified - full implementation would run greedy search
 */
__global__ void buildGraphEdgesKernel(uint8_t* d_pIndex_static,
                                     uint8_t* d_pIndex_fresh,
                                     uint32_t* d_fresh_count,
                                     uint32_t static_size,
                                     uint32_t start_idx,
                                     uint32_t end_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t fresh_id = start_idx + idx;

    if (fresh_id >= end_idx) return;

    // Get vector from fresh index
    uint8_t* node = d_pIndex_fresh + (fresh_id * INDEX_ENTRY_LEN);
    datatype_t* vec = (datatype_t*)node;

    // Simple strategy: Connect to MEDOID and a few random nodes from static index
    // In full implementation, this would run greedy search to find neighbors

    uint32_t* degree_ptr = (uint32_t*)(node + D * sizeof(datatype_t));
    uint32_t* neighbors = degree_ptr + 1;

    // Add MEDOID as first neighbor
    neighbors[0] = MEDOID;
    *degree_ptr = 1;

    // TODO: Full implementation would:
    // 1. Run greedy search on combined static+fresh index
    // 2. Find R nearest neighbors using robust prune
    // 3. Update bidirectional edges
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

void initFreshIndex(FreshIndex* index) {
    printf("[FreshIndex] Initializing...\n");

    index->capacity = FRESH_INDEX_CAPACITY;
    index->total_size_bytes = index->capacity * INDEX_ENTRY_LEN;

    // Allocate device memory
    cudaError_t err = cudaMalloc(&index->d_pIndex, index->total_size_bytes);
    gpuErrchk(err);

    err = cudaMemset(index->d_pIndex, 0, index->total_size_bytes);
    gpuErrchk(err);

    // Allocate atomic counter on device
    err = cudaMalloc(&index->d_count, sizeof(uint32_t));
    gpuErrchk(err);

    err = cudaMemset(index->d_count, 0, sizeof(uint32_t));
    gpuErrchk(err);

    // Allocate host mirror
    index->h_count = (uint32_t*)malloc(sizeof(uint32_t));
    if (!index->h_count) {
        fprintf(stderr, "Failed to allocate host counter\n");
        exit(1);
    }
    *index->h_count = 0;

    // Allocate host mirror (optional, for consolidation)
    index->h_pIndex = (uint8_t*)malloc(index->total_size_bytes);
    if (!index->h_pIndex) {
        fprintf(stderr, "Failed to allocate host fresh index\n");
        exit(1);
    }

    printf("[FreshIndex] Capacity: %u nodes, %.2f MB\n",
           index->capacity,
           index->total_size_bytes / (1024.0 * 1024.0));
}

void insertBatch(FreshIndex* fresh, StaticIndex* static_idx, DeleteBuffer* del_buf,
                 datatype_t* h_vectors, uint32_t* h_ids, uint32_t batch_size) {
    if (batch_size == 0) return;

    // Check capacity
    uint32_t current_count;
    cudaMemcpy(&current_count, fresh->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (current_count + batch_size > fresh->capacity) {
        fprintf(stderr, "[Insert] Warning: Fresh index capacity exceeded. Need consolidation.\n");
        batch_size = fresh->capacity - current_count;
        if (batch_size == 0) return;
    }

    // Allocate device memory for batch
    datatype_t* d_vectors;
    uint32_t* d_ids;

    cudaError_t err = cudaMalloc(&d_vectors, batch_size * D * sizeof(datatype_t));
    gpuErrchk(err);

    err = cudaMalloc(&d_ids, batch_size * sizeof(uint32_t));
    gpuErrchk(err);

    // Copy vectors to device
    err = cudaMemcpy(d_vectors, h_vectors, batch_size * D * sizeof(datatype_t), cudaMemcpyHostToDevice);
    gpuErrchk(err);

    // Insert vectors
    uint32_t num_blocks = (batch_size + 256 - 1) / 256;
    insertVectorsKernel<<<num_blocks, 256>>>(
        fresh->d_pIndex,
        d_vectors,
        d_ids,
        fresh->d_count,
        batch_size,
        fresh->capacity
    );

    err = cudaDeviceSynchronize();
    gpuErrchk(err);

    // Build graph edges (simplified)
    buildGraphEdgesKernel<<<num_blocks, 256>>>(
        static_idx->d_pIndex,
        fresh->d_pIndex,
        fresh->d_count,
        static_idx->num_nodes,
        current_count,
        current_count + batch_size
    );

    err = cudaDeviceSynchronize();
    gpuErrchk(err);

    // Copy result IDs back
    err = cudaMemcpy(h_ids, d_ids, batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    gpuErrchk(err);

    // Update host counter
    cudaMemcpy(fresh->h_count, fresh->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("[Insert] Inserted %u vectors, fresh index now has %u nodes\n",
           batch_size, *fresh->h_count);

    // Cleanup
    cudaFree(d_vectors);
    cudaFree(d_ids);
}

void clearFreshIndex(FreshIndex* index) {
    // Reset memory
    cudaError_t err = cudaMemset(index->d_pIndex, 0, index->total_size_bytes);
    gpuErrchk(err);

    err = cudaMemset(index->d_count, 0, sizeof(uint32_t));
    gpuErrchk(err);

    *index->h_count = 0;

    printf("[FreshIndex] Cleared\n");
}

uint32_t getFreshIndexSize(FreshIndex* index) {
    cudaError_t err = cudaMemcpy(index->h_count, index->d_count,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost);
    gpuErrchk(err);

    return *index->h_count;
}

void freeFreshIndex(FreshIndex* index) {
    if (index->d_pIndex) {
        cudaFree(index->d_pIndex);
        index->d_pIndex = nullptr;
    }
    if (index->d_count) {
        cudaFree(index->d_count);
        index->d_count = nullptr;
    }
    if (index->h_count) {
        free(index->h_count);
        index->h_count = nullptr;
    }
    if (index->h_pIndex) {
        free(index->h_pIndex);
        index->h_pIndex = nullptr;
    }
    index->capacity = 0;
}
