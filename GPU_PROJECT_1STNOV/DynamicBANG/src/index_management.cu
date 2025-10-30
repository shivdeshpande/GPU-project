#include "../include/common.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// ============================================================================
// STATIC INDEX INITIALIZATION
// ============================================================================

void initStaticIndex(StaticIndex* index, const char* vector_file, const char* graph_file) {
    printf("[StaticIndex] Initializing...\n");

    // Load vectors from file
    float* h_vectors = nullptr;
    uint32_t num_vectors = 0, dim = 0;
    loadFvecsFile(vector_file, &h_vectors, &num_vectors, &dim);

    if (dim != D) {
        fprintf(stderr, "Error: Vector dimension mismatch. Expected %d, got %u\n", D, dim);
        exit(1);
    }

    index->num_nodes = num_vectors;
    index->capacity = INITIAL_CAPACITY;
    index->vector_size_bytes = D * sizeof(float);
    index->node_size_bytes = (1 + R) * sizeof(uint32_t);  // degree + neighbors

    // Allocate device memory for vectors
    size_t vectors_bytes = index->capacity * index->vector_size_bytes;
    cudaError_t err = cudaMalloc(&index->d_vectors, vectors_bytes);
    checkCudaError(err, "Failed to allocate device memory for static vectors");

    // Copy vectors to device
    err = cudaMemcpy(index->d_vectors, h_vectors,
                     num_vectors * index->vector_size_bytes,
                     cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vectors to device");

    // Load or initialize graph
    uint32_t* h_graph = nullptr;
    uint32_t graph_nodes = 0;

    if (graph_file != nullptr) {
        loadGraphBinary(graph_file, &h_graph, &graph_nodes);
        if (graph_nodes != num_vectors) {
            fprintf(stderr, "Warning: Graph size (%u) != vector count (%u). Using graph size.\n",
                    graph_nodes, num_vectors);
            index->num_nodes = graph_nodes;
        }
    } else {
        // Initialize empty graph (will build later)
        printf("[StaticIndex] No graph file provided, initializing empty graph\n");
        h_graph = (uint32_t*)calloc(num_vectors * (1 + R), sizeof(uint32_t));
        if (!h_graph) {
            fprintf(stderr, "Failed to allocate host graph\n");
            exit(1);
        }
    }

    // Allocate device memory for graph
    size_t graph_bytes = index->capacity * index->node_size_bytes;
    err = cudaMalloc(&index->d_graph, graph_bytes);
    checkCudaError(err, "Failed to allocate device memory for static graph");

    // Copy graph to device
    err = cudaMemcpy(index->d_graph, h_graph,
                     index->num_nodes * index->node_size_bytes,
                     cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy graph to device");

    printf("[StaticIndex] Loaded: %u nodes, %d dimensions\n", index->num_nodes, D);
    printf("[StaticIndex] Memory: Vectors %.2f MB, Graph %.2f MB\n",
           vectors_bytes / (1024.0 * 1024.0),
           graph_bytes / (1024.0 * 1024.0));

    // Cleanup host memory
    free(h_vectors);
    if (h_graph) free(h_graph);
}

void freeStaticIndex(StaticIndex* index) {
    if (index->d_vectors) {
        cudaFree(index->d_vectors);
        index->d_vectors = nullptr;
    }
    if (index->d_graph) {
        cudaFree(index->d_graph);
        index->d_graph = nullptr;
    }
    index->num_nodes = 0;
    index->capacity = 0;
}

// ============================================================================
// FRESH INDEX INITIALIZATION
// ============================================================================

void initFreshIndex(FreshIndex* index) {
    printf("[FreshIndex] Initializing...\n");

    index->capacity = FRESH_INDEX_CAPACITY;
    index->vector_size_bytes = D * sizeof(float);
    index->node_size_bytes = (1 + R) * sizeof(uint32_t);

    // Allocate device memory for vectors
    size_t vectors_bytes = index->capacity * index->vector_size_bytes;
    cudaError_t err = cudaMalloc(&index->d_vectors, vectors_bytes);
    checkCudaError(err, "Failed to allocate device memory for fresh vectors");

    err = cudaMemset(index->d_vectors, 0, vectors_bytes);
    checkCudaError(err, "Failed to initialize fresh vectors");

    // Allocate device memory for graph
    size_t graph_bytes = index->capacity * index->node_size_bytes;
    err = cudaMalloc(&index->d_graph, graph_bytes);
    checkCudaError(err, "Failed to allocate device memory for fresh graph");

    err = cudaMemset(index->d_graph, 0, graph_bytes);
    checkCudaError(err, "Failed to initialize fresh graph");

    // Allocate atomic counter on device
    err = cudaMalloc(&index->d_count, sizeof(uint32_t));
    checkCudaError(err, "Failed to allocate device counter for fresh index");

    err = cudaMemset(index->d_count, 0, sizeof(uint32_t));
    checkCudaError(err, "Failed to initialize device counter");

    // Allocate host mirror of counter
    index->h_count = (uint32_t*)malloc(sizeof(uint32_t));
    if (!index->h_count) {
        fprintf(stderr, "Failed to allocate host counter\n");
        exit(1);
    }
    *index->h_count = 0;

    printf("[FreshIndex] Capacity: %u nodes\n", index->capacity);
    printf("[FreshIndex] Memory: Vectors %.2f MB, Graph %.2f MB\n",
           vectors_bytes / (1024.0 * 1024.0),
           graph_bytes / (1024.0 * 1024.0));
}

void clearFreshIndex(FreshIndex* index) {
    // Reset vectors and graph to zero
    size_t vectors_bytes = index->capacity * index->vector_size_bytes;
    size_t graph_bytes = index->capacity * index->node_size_bytes;

    cudaError_t err = cudaMemset(index->d_vectors, 0, vectors_bytes);
    checkCudaError(err, "Failed to clear fresh vectors");

    err = cudaMemset(index->d_graph, 0, graph_bytes);
    checkCudaError(err, "Failed to clear fresh graph");

    err = cudaMemset(index->d_count, 0, sizeof(uint32_t));
    checkCudaError(err, "Failed to reset fresh index counter");

    *index->h_count = 0;

    printf("[FreshIndex] Cleared\n");
}

uint32_t getFreshIndexSize(FreshIndex* index) {
    // Copy count from device to host
    cudaError_t err = cudaMemcpy(index->h_count, index->d_count,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to get fresh index size");

    return *index->h_count;
}

void freeFreshIndex(FreshIndex* index) {
    if (index->d_vectors) {
        cudaFree(index->d_vectors);
        index->d_vectors = nullptr;
    }
    if (index->d_graph) {
        cudaFree(index->d_graph);
        index->d_graph = nullptr;
    }
    if (index->d_count) {
        cudaFree(index->d_count);
        index->d_count = nullptr;
    }
    if (index->h_count) {
        free(index->h_count);
        index->h_count = nullptr;
    }
    index->capacity = 0;
}

// ============================================================================
// INDEX STATISTICS
// ============================================================================

void printIndexStats(const StaticIndex* static_idx, const FreshIndex* fresh_idx,
                     const DeleteBuffer* del_buf) {
    uint32_t fresh_size = *fresh_idx->h_count;
    uint32_t total_nodes = static_idx->num_nodes + fresh_size;
    uint32_t active_nodes = total_nodes - del_buf->num_deleted;

    printf("\n========== Index Statistics ==========\n");
    printf("Static Index:  %u nodes\n", static_idx->num_nodes);
    printf("Fresh Index:   %u / %u nodes (%.1f%% full)\n",
           fresh_size, fresh_idx->capacity,
           100.0 * fresh_size / fresh_idx->capacity);
    printf("Deleted:       %u nodes (%.1f%% of total)\n",
           del_buf->num_deleted,
           100.0 * del_buf->num_deleted / total_nodes);
    printf("Active Nodes:  %u\n", active_nodes);
    printf("======================================\n\n");
}

// ============================================================================
// GPU UTILITY FUNCTIONS
// ============================================================================

// Get vector from static index
__device__ inline void getStaticVector(const StaticIndex* index, uint32_t node_id,
                                       float* out_vector) {
    const float* vec = &index->d_vectors[node_id * D];
    for (int i = 0; i < D; i++) {
        out_vector[i] = vec[i];
    }
}

// Get vector from fresh index
__device__ inline void getFreshVector(const FreshIndex* index, uint32_t fresh_id,
                                      float* out_vector) {
    const float* vec = &index->d_vectors[fresh_id * D];
    for (int i = 0; i < D; i++) {
        out_vector[i] = vec[i];
    }
}

// Get neighbors from static graph
__device__ inline uint32_t getStaticNeighbors(const StaticIndex* index, uint32_t node_id,
                                              uint32_t* out_neighbors) {
    const uint32_t* node = &index->d_graph[node_id * (1 + R)];
    uint32_t degree = node[0];

    if (degree > R) degree = R;  // Safety check

    for (uint32_t i = 0; i < degree; i++) {
        out_neighbors[i] = node[1 + i];
    }
    return degree;
}

// Get neighbors from fresh graph
__device__ inline uint32_t getFreshNeighbors(const FreshIndex* index, uint32_t fresh_id,
                                             uint32_t* out_neighbors) {
    const uint32_t* node = &index->d_graph[fresh_id * (1 + R)];
    uint32_t degree = node[0];

    if (degree > R) degree = R;

    for (uint32_t i = 0; i < degree; i++) {
        out_neighbors[i] = node[1 + i];
    }
    return degree;
}
