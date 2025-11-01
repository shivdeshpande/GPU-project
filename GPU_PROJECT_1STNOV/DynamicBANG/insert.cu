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
 * Compute distances from a fresh vector to static index for neighbor selection
 */
__global__ void computeDistancesToStaticKernel(uint8_t* d_pIndex_static,
                                               uint8_t* d_pIndex_fresh,
                                               uint32_t fresh_id,
                                               uint32_t static_size,
                                               float* d_distances,
                                               uint32_t* d_indices) {
    uint32_t static_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (static_id >= static_size) return;

    // Get fresh vector
    datatype_t* fresh_vec = (datatype_t*)(d_pIndex_fresh + (fresh_id * INDEX_ENTRY_LEN));

    // Get static vector
    datatype_t* static_vec = (datatype_t*)(d_pIndex_static + (static_id * INDEX_ENTRY_LEN));

    // Compute L2 distance
    float dist = 0.0f;
    for (uint32_t i = 0; i < D; i++) {
        float diff = (float)fresh_vec[i] - (float)static_vec[i];
        dist += diff * diff;
    }

    d_distances[static_id] = dist;
    d_indices[static_id] = static_id;
}

/**
 * Simple selection sort to find top-K neighbors (runs on single thread per vector)
 * For small K and infrequent inserts, this is acceptable
 */
__global__ void selectTopKNeighborsKernel(float* d_distances,
                                         uint32_t* d_indices,
                                         uint32_t* d_neighbors,
                                         uint32_t static_size,
                                         uint32_t num_neighbors,
                                         uint32_t fresh_id) {
    // Each block handles one fresh vector
    if (blockIdx.x != fresh_id) return;

    // Use shared memory for top-K selection
    extern __shared__ float s_data[];
    float* s_dists = s_data;
    uint32_t* s_indices = (uint32_t*)&s_dists[num_neighbors];

    // Initialize with first num_neighbors elements
    if (threadIdx.x < num_neighbors && threadIdx.x < static_size) {
        s_dists[threadIdx.x] = d_distances[threadIdx.x];
        s_indices[threadIdx.x] = d_indices[threadIdx.x];
    }
    __syncthreads();

    // Simple insertion into top-K for remaining elements
    for (uint32_t i = threadIdx.x + num_neighbors; i < static_size; i += blockDim.x) {
        float dist = d_distances[i];
        uint32_t idx = d_indices[i];

        // Find position in top-K
        for (uint32_t j = 0; j < num_neighbors; j++) {
            if (dist < s_dists[j]) {
                // Shift and insert
                for (uint32_t k = num_neighbors - 1; k > j; k--) {
                    s_dists[k] = s_dists[k - 1];
                    s_indices[k] = s_indices[k - 1];
                }
                s_dists[j] = dist;
                s_indices[j] = idx;
                break;
            }
        }
    }
    __syncthreads();

    // Write results
    if (threadIdx.x < num_neighbors) {
        d_neighbors[threadIdx.x] = s_indices[threadIdx.x];
    }
}

/**
 * Build graph edges for inserted vectors using nearest neighbor search
 * Connects each fresh node to its R nearest neighbors from the static index
 */
__global__ void buildGraphEdgesKernel(uint8_t* d_pIndex_fresh,
                                     uint32_t* d_neighbors,
                                     uint32_t fresh_id,
                                     uint32_t max_degree) {
    if (blockIdx.x != 0) return;

    // Get fresh node
    uint8_t* node = d_pIndex_fresh + (fresh_id * INDEX_ENTRY_LEN);
    uint32_t* degree_ptr = (uint32_t*)(node + D * sizeof(datatype_t));
    uint32_t* neighbors = degree_ptr + 1;

    // Copy neighbors
    if (threadIdx.x < max_degree) {
        neighbors[threadIdx.x] = d_neighbors[threadIdx.x];
    }

    // Set degree
    if (threadIdx.x == 0) {
        *degree_ptr = max_degree;
    }
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

    // Build graph edges for each inserted vector
    printf("[Insert] Building graph edges for %u vectors...\n", batch_size);

    // Allocate temporary buffers for distance computation and neighbor selection
    float* d_distances;
    uint32_t* d_indices;
    uint32_t* d_selected_neighbors;

    err = cudaMalloc(&d_distances, static_idx->num_nodes * sizeof(float));
    gpuErrchk(err);
    err = cudaMalloc(&d_indices, static_idx->num_nodes * sizeof(uint32_t));
    gpuErrchk(err);
    err = cudaMalloc(&d_selected_neighbors, R * sizeof(uint32_t));
    gpuErrchk(err);

    // For each freshly inserted vector, find its R nearest neighbors
    for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t fresh_id = current_count + i;

        // Compute distances to all static vectors
        uint32_t dist_blocks = (static_idx->num_nodes + 255) / 256;
        computeDistancesToStaticKernel<<<dist_blocks, 256>>>(
            static_idx->d_pIndex,
            fresh->d_pIndex,
            fresh_id,
            static_idx->num_nodes,
            d_distances,
            d_indices
        );
        gpuErrchk(cudaDeviceSynchronize());

        // Select top R neighbors
        uint32_t num_neighbors_to_select = (R < static_idx->num_nodes) ? R : static_idx->num_nodes;
        size_t shared_mem_size = num_neighbors_to_select * (sizeof(float) + sizeof(uint32_t));
        selectTopKNeighborsKernel<<<1, 256, shared_mem_size>>>(
            d_distances,
            d_indices,
            d_selected_neighbors,
            static_idx->num_nodes,
            num_neighbors_to_select,
            fresh_id
        );
        gpuErrchk(cudaDeviceSynchronize());

        // Build edges in fresh index
        buildGraphEdgesKernel<<<1, 256>>>(
            fresh->d_pIndex,
            d_selected_neighbors,
            fresh_id,
            num_neighbors_to_select
        );
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Cleanup temporary buffers
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_selected_neighbors);

    printf("[Insert] Graph construction complete. Each node connected to %u neighbors.\n", R);

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
