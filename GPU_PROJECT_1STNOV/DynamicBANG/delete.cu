#include "dynamicBANG.h"
#include "utils/utils.h"
#include <cstring>
#include <cstdio>
#include <cassert>
// ============================================================================
// GPU KERNELS FOR DELETE BUFFER OPERATIONS
// ============================================================================

/**
 * Mark nodes as deleted in the bitmap (batch operation)
 */
__global__ void markDeletedKernel(uint32_t* d_bitmap, uint32_t* d_ids, uint32_t batch_size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        uint32_t node_id = d_ids[idx];
        uint32_t word_idx = node_id / 32;
        uint32_t bit_idx = node_id % 32;

        // Atomic OR to set the bit (thread-safe)
        atomicOr(&d_bitmap[word_idx], (1U << bit_idx));
    }
}

/**
 * Count total deleted nodes (parallel reduction)
 */
__global__ void countDeletedKernel(const uint32_t* d_bitmap, uint32_t* d_count, uint32_t bitmap_words) {
    __shared__ uint32_t shared_count[256];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Count set bits in this thread's words
    uint32_t local_count = 0;
    if (idx < bitmap_words) {
        uint32_t word = d_bitmap[idx];
        // Population count (count set bits)
        local_count = __popc(word);
    }

    shared_count[tid] = local_count;
    __syncthreads();

    // Parallel reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(d_count, shared_count[0]);
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

void initDeleteBuffer(DeleteBuffer* buffer, uint32_t total_capacity) {
    // Calculate bitmap size (1 bit per node)
    uint32_t bitmap_words = (total_capacity + 31) / 32;
    buffer->bitmap_size_bytes = bitmap_words * sizeof(uint32_t);
    buffer->total_nodes = total_capacity;
    buffer->num_deleted = 0;

    // Allocate device bitmap (initialized to 0)
    cudaError_t err = cudaMalloc(&buffer->d_bitmap, buffer->bitmap_size_bytes);
    gpuErrchk(err);

    err = cudaMemset(buffer->d_bitmap, 0, buffer->bitmap_size_bytes);
    gpuErrchk(err);

    // Allocate host bitmap
    buffer->h_bitmap = (uint32_t*)malloc(buffer->bitmap_size_bytes);
    if (!buffer->h_bitmap) {
        fprintf(stderr, "Failed to allocate host delete bitmap\n");
        exit(1);
    }
    memset(buffer->h_bitmap, 0, buffer->bitmap_size_bytes);

    printf("[DeleteBuffer] Initialized: %u nodes, %.2f MB\n",
           total_capacity, buffer->bitmap_size_bytes / (1024.0 * 1024.0));
}

void deleteBatch(DeleteBuffer* del_buf, uint32_t* h_ids, uint32_t batch_size) {
    if (batch_size == 0) return;

    // Allocate device memory for IDs
    uint32_t* d_ids;
    cudaError_t err = cudaMalloc(&d_ids, batch_size * sizeof(uint32_t));
    gpuErrchk(err);

    // Copy IDs to device
    err = cudaMemcpy(d_ids, h_ids, batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    gpuErrchk(err);

    // Launch kernel to mark deleted
    uint32_t num_blocks = (batch_size + 256 - 1) / 256;
    markDeletedKernel<<<num_blocks, 256>>>(del_buf->d_bitmap, d_ids, batch_size);

    err = cudaDeviceSynchronize();
    gpuErrchk(err);

    // Update deletion count
    del_buf->num_deleted += batch_size;

    // Cleanup
    cudaFree(d_ids);
}

void clearDeleteBuffer(DeleteBuffer* buffer) {
    // Reset bitmap to all zeros
    cudaError_t err = cudaMemset(buffer->d_bitmap, 0, buffer->bitmap_size_bytes);
    gpuErrchk(err);

    if (buffer->h_bitmap) {
        memset(buffer->h_bitmap, 0, buffer->bitmap_size_bytes);
    }
    buffer->num_deleted = 0;

    printf("[DeleteBuffer] Cleared\n");
}

uint32_t countDeleted(DeleteBuffer* buffer) {
    // Allocate device counter
    uint32_t* d_count;
    cudaError_t err = cudaMalloc(&d_count, sizeof(uint32_t));
    gpuErrchk(err);

    err = cudaMemset(d_count, 0, sizeof(uint32_t));
    gpuErrchk(err);

    // Launch counting kernel
    uint32_t bitmap_words = (buffer->total_nodes + 31) / 32;
    uint32_t num_blocks = (bitmap_words + 256 - 1) / 256;

    countDeletedKernel<<<num_blocks, 256>>>(buffer->d_bitmap, d_count, bitmap_words);

    err = cudaDeviceSynchronize();
    gpuErrchk(err);

    // Copy result back
    uint32_t count;
    err = cudaMemcpy(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    gpuErrchk(err);

    cudaFree(d_count);

    buffer->num_deleted = count;
    return count;
}

bool isNodeDeleted(const DeleteBuffer* buffer, uint32_t node_id) {
    if (node_id >= buffer->total_nodes) return false;

    uint32_t word_idx = node_id / 32;
    uint32_t bit_idx = node_id % 32;
    return (buffer->h_bitmap[word_idx] & (1U << bit_idx)) != 0;
}

void freeDeleteBuffer(DeleteBuffer* buffer) {
    if (buffer->d_bitmap) {
        cudaFree(buffer->d_bitmap);
        buffer->d_bitmap = nullptr;
    }
    if (buffer->h_bitmap) {
        free(buffer->h_bitmap);
        buffer->h_bitmap = nullptr;
    }
    buffer->num_deleted = 0;
    buffer->total_nodes = 0;
}
