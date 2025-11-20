#include "insert.h"
#include "lockfree_graph.cuh"

// GPU kernel to copy vector data into graph
__global__ void copyVectorKernel(uint8_t* d_graph,
                                  float* d_vector,
                                  unsigned pointId,
                                  unsigned graphEntrySize) {
    unsigned tid = threadIdx.x;

    if (tid < D) {
        uint8_t* graphEntry = d_graph + pointId * graphEntrySize;
        float* vectorDest = (float*)graphEntry;
        vectorDest[tid] = d_vector[tid];
    }
}

// GPU kernel to extract vector data from graph
__global__ void extractVectorKernel(uint8_t* d_graph,
                                     float* d_vector,
                                     unsigned pointId,
                                     unsigned graphEntrySize) {
    unsigned tid = threadIdx.x;

    if (tid < D) {
        uint8_t* graphEntry = d_graph + pointId * graphEntrySize;
        float* vectorSrc = (float*)graphEntry;
        d_vector[tid] = vectorSrc[tid];
    }
}

// Helper: Copy vector into graph at specified position
void copyVectorToGraph(uint8_t* d_graph,
                       float* d_vector,
                       unsigned pointId) {
    unsigned threadsPerBlock = D;
    copyVectorKernel<<<1, threadsPerBlock>>>(d_graph, d_vector, pointId, graphEntrySize);
    gpuErrchk(cudaDeviceSynchronize());
}

// Helper: Extract vector from graph at specified position
void extractVectorFromGraph(uint8_t* d_graph,
                             float* d_vector,
                             unsigned pointId) {
    unsigned threadsPerBlock = D;
    extractVectorKernel<<<1, threadsPerBlock>>>(d_graph, d_vector, pointId, graphEntrySize);
    gpuErrchk(cudaDeviceSynchronize());
}

/**
 * Insert a single point into the graph
 *
 * This function implements the FreshDiskANN insert algorithm:
 * 1. Copy new vector into graph at newPointId position
 * 2. Run GreedySearch from medoid to build candidate set
 * 3. Call computeOutNeighbors which internally uses pruneOutNeighbors
 *    (RobustPrune with α-RNG property) to select R neighbors
 * 4. Reverse edges are updated by computeOutNeighbors
 */
void insertPoint(uint8_t* d_graph,
                 float* d_newVector,
                 unsigned newPointId,
                 float alpha,
                 unsigned medoid) {

    // Allocate GPU memory for visited set (candidate neighbors)
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    gpuErrchk(cudaMalloc(&d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, sizeof(unsigned)));

    // Step 1: Copy new vector into graph at newPointId position
    copyVectorToGraph(d_graph, d_newVector, newPointId);

    // Step 2: Run GreedySearch to build candidate set (visited set)
    // This finds nearest neighbors that will be considered for edges
    greedySearch(d_graph,
                 d_newVector,
                 d_visitedSet,
                 d_visitedSetCount,
                 newPointId,
                 1,  // batchSize = 1
                 L); // searchL = L (default)

    // Step 3: Use existing computeOutNeighbors
    // This internally calls pruneOutNeighbors (RobustPrune with α-RNG)
    // to select R neighbors while maintaining graph density
    computeOutNeighbors(d_graph,
                        d_newVector,
                        d_visitedSet,
                        d_visitedSetCount,
                        alpha,
                        d_reverseEdgeIndex,
                        newPointId,
                        1);  // batchSize = 1

    // Step 4: Update reverse edges to maintain bidirectional connectivity
    computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);

    // Cleanup
    gpuErrchk(cudaFree(d_visitedSet));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
}

/**
 * Insert multiple points in batch
 *
 * More efficient than calling insertPoint multiple times because:
 * - Amortizes GPU memory allocation overhead
 * - Better GPU utilization with larger batch sizes
 * - Reduces kernel launch overhead
 */
void batchInsertPoints(uint8_t* d_graph,
                       float* d_newVectors,
                       unsigned* newPointIds,
                       unsigned batchSize,
                       float alpha,
                       unsigned medoid) {

    if (batchSize == 0) return;

    // Allocate GPU memory for visited sets (one per point in batch)
    unsigned* d_visitedSets;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    gpuErrchk(cudaMalloc(&d_visitedSets, batchSize * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, batchSize * sizeof(unsigned)));

    // Copy all new vectors into graph
    for (unsigned i = 0; i < batchSize; i++) {
        float* d_vector = d_newVectors + i * D;
        copyVectorToGraph(d_graph, d_vector, newPointIds[i]);
    }

    // Process all points in batch
    // Note: We still need to process one at a time because greedySearch
    // and computeOutNeighbors expect a batch starting at consecutive IDs
    // In a production system, you'd modify these functions to handle
    // arbitrary point IDs
    for (unsigned i = 0; i < batchSize; i++) {
        unsigned pointId = newPointIds[i];
        float* d_vector = d_newVectors + i * D;
        unsigned* d_visitedSet = d_visitedSets + i * MAX_PARENTS_PERQUERY;
        unsigned* d_count = d_visitedSetCount + i;

        // Run GreedySearch for this point
        greedySearch(d_graph,
                     d_vector,
                     d_visitedSet,
                     d_count,
                     pointId,
                     1,
                     L); // searchL = L (default)

        // Compute out-neighbors with RobustPrune
        computeOutNeighbors(d_graph,
                            d_vector,
                            d_visitedSet,
                            d_count,
                            alpha,
                            d_reverseEdgeIndex,
                            pointId,
                            1);
    }

    // Update reverse edges for all points
    computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);

    // Cleanup
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
}

/**
 * Reinsert a deleted point (for consolidation)
 *
 * Similar to insertPoint but handles previously deleted points.
 * The vector is already in the graph, we just need to:
 * 1. Rebuild its neighbor list with RobustPrune
 * 2. Update reverse edges
 * 3. Unmark from DeleteList
 */
void reinsertPoint(uint8_t* d_graph,
                   unsigned pointId,
                   float alpha,
                   DeleteList* deleteList) {

    // Check if point is actually deleted
    if (!deleteList->isDeleted(pointId)) {
        fprintf(stderr, "Warning: Attempting to reinsert non-deleted point %u\n", pointId);
        return;
    }

    // Allocate GPU memory
    float* d_vector;
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    gpuErrchk(cudaMalloc(&d_vector, D * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, sizeof(unsigned)));

    // Extract vector from graph (it's already there, just needs edges rebuilt)
    extractVectorFromGraph(d_graph, d_vector, pointId);

    // Run GreedySearch to build candidate set
    greedySearch(d_graph,
                 d_vector,
                 d_visitedSet,
                 d_visitedSetCount,
                 pointId,
                 1,
                 L); // searchL = L (default)

    // Compute out-neighbors with RobustPrune
    computeOutNeighbors(d_graph,
                        d_vector,
                        d_visitedSet,
                        d_visitedSetCount,
                        alpha,
                        d_reverseEdgeIndex,
                        pointId,
                        1);

    // Update reverse edges
    computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);

    // Note: Caller should handle unmarking from deleteList after consolidation
    // We don't unmark here because consolidation may process many points

    // Cleanup
    gpuErrchk(cudaFree(d_vector));
    gpuErrchk(cudaFree(d_visitedSet));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
}

/**
 * GPU kernel to add reverse edges with version protection
 *
 * For each entry in reverse edge index, add an edge from parent to child
 * using version numbers for thread-safety.
 */
__global__ void addReverseEdgesVersionedKernel(uint8_t* d_graph,
                                                unsigned* d_versions,
                                                uint8_t* d_reverseEdgeIndex,
                                                unsigned graphEntrySize,
                                                unsigned reverseIndexEntrySize) {
    unsigned childId = blockIdx.x * blockDim.x + threadIdx.x;
    if (childId >= N) return;

    // Get reverse edge entry for this child
    uint8_t* reverseEntry = d_reverseEdgeIndex + childId * reverseIndexEntrySize;
    unsigned* reverseCount = (unsigned*)reverseEntry;
    unsigned* reverseParents = reverseCount + 1;

    unsigned count = *reverseCount;
    if (count == 0) return;

    // Add edges from each parent to this child
    for (unsigned i = 0; i < count && i < MAX_REVERSE_INDEX_ENTRIES; i++) {
        unsigned parentId = reverseParents[i];

        // Use version-protected add
        addReverseEdgeVersioned(d_graph, d_versions, parentId, childId,
                                 graphEntrySize, D, R);
    }
}

/**
 * Insert a single point with version-based concurrency control
 */
void insertPointVersioned(uint8_t* d_graph,
                          unsigned* d_versions,
                          float* d_newVector,
                          unsigned newPointId,
                          float alpha,
                          unsigned medoid) {

    // Allocate GPU memory for visited set (candidate neighbors)
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    gpuErrchk(cudaMalloc(&d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_reverseEdgeIndex, 0, N * reverseIndexEntrySize * sizeof(uint8_t)));

    // Step 1: Copy new vector into graph at newPointId position
    // No version needed - this is the new point's own memory
    copyVectorToGraph(d_graph, d_newVector, newPointId);

    // Step 2: Run GreedySearch to build candidate set
    // This reads other vertices - they may be modified concurrently
    // The version-based reads in greedySearch will handle this
    greedySearch(d_graph,
                 d_newVector,
                 d_visitedSet,
                 d_visitedSetCount,
                 newPointId,
                 1,  // batchSize = 1
                 L); // searchL = L (default)

    // Step 3: Compute out-neighbors with RobustPrune
    // This writes to newPointId's adjacency list
    // Use version protection for this vertex
    computeOutNeighbors(d_graph,
                        d_newVector,
                        d_visitedSet,
                        d_visitedSetCount,
                        alpha,
                        d_reverseEdgeIndex,
                        newPointId,
                        1);  // batchSize = 1

    // Mark the new point's write as complete
    // (computeOutNeighbors already wrote to it)
    // Increment version to signal readers
    unsigned* h_version = &d_versions[newPointId];
    cudaMemset(h_version, 0, sizeof(unsigned));  // Reset to 0 (even = stable)

    // Step 4: Update reverse edges using version-protected kernel
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addReverseEdgesVersionedKernel<<<numBlocks, threadsPerBlock>>>(
        d_graph, d_versions, d_reverseEdgeIndex,
        graphEntrySize, reverseIndexEntrySize);

    gpuErrchk(cudaDeviceSynchronize());

    // Cleanup
    gpuErrchk(cudaFree(d_visitedSet));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
}

// ============== Pre-allocated Insert Functions ==============

void allocateInsertBuffers(InsertBuffers* buffers) {
    gpuErrchk(cudaMalloc(&buffers->d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_visitedSetCount, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));

    // Pre-allocate greedy search buffers (avoids 14 cudaMalloc per insert!)
    allocateGreedySearchBuffers(&buffers->gsBuffers, 1);

    // Pre-allocate out-neighbors buffers (avoids 9 cudaMalloc per insert!)
    allocateOutNeighborsBuffers(&buffers->outNbrsBuffers, 1);

    buffers->allocated = true;
}

void freeInsertBuffers(InsertBuffers* buffers) {
    if (buffers->allocated) {
        gpuErrchk(cudaFree(buffers->d_visitedSet));
        gpuErrchk(cudaFree(buffers->d_visitedSetCount));
        gpuErrchk(cudaFree(buffers->d_reverseEdgeIndex));

        // Free greedy search buffers
        freeGreedySearchBuffers(&buffers->gsBuffers);

        // Free out-neighbors buffers
        freeOutNeighborsBuffers(&buffers->outNbrsBuffers);

        buffers->allocated = false;
    }
}

void insertPointVersionedPrealloc(uint8_t* d_graph,
                                   unsigned* d_versions,
                                   float* d_newVector,
                                   unsigned newPointId,
                                   float alpha,
                                   InsertBuffers* buffers,
                                   cudaStream_t stream,
                                   unsigned medoid) {

    // Use pre-allocated buffers (no cudaMalloc overhead!)
    unsigned* d_visitedSet = buffers->d_visitedSet;
    unsigned* d_visitedSetCount = buffers->d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex = buffers->d_reverseEdgeIndex;

    // Reset buffers async
    gpuErrchk(cudaMemsetAsync(d_visitedSetCount, 0, sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(d_reverseEdgeIndex, 0, N * reverseIndexEntrySize * sizeof(uint8_t), stream));

    // Step 1: Copy new vector into graph at newPointId position
    copyVectorToGraph(d_graph, d_newVector, newPointId);

    // Step 2: Run GreedySearch with pre-allocated buffers (avoids 14 cudaMalloc!)
    greedySearchVersionedPrealloc(d_graph,
                                   d_versions,
                                   d_newVector,
                                   d_visitedSet,
                                   d_visitedSetCount,
                                   newPointId,
                                   1,  // batchSize = 1
                                   L,  // searchL = L (default)
                                   nullptr,  // d_deleted
                                   &buffers->gsBuffers,
                                   stream);

    // Step 3: Compute out-neighbors with pre-allocated buffers (avoids 9 cudaMalloc!)
    computeOutNeighborsPrealloc(d_graph,
                                 d_newVector,
                                 d_visitedSet,
                                 d_visitedSetCount,
                                 alpha,
                                 d_reverseEdgeIndex,
                                 newPointId,
                                 1,  // batchSize = 1
                                 &buffers->outNbrsBuffers,
                                 stream);

    // Mark the new point's write as complete
    unsigned* h_version = &d_versions[newPointId];
    cudaMemsetAsync(h_version, 0, sizeof(unsigned), stream);

    // Step 4: Update reverse edges using version-protected kernel
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addReverseEdgesVersionedKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_graph, d_versions, d_reverseEdgeIndex,
        graphEntrySize, reverseIndexEntrySize);

    gpuErrchk(cudaStreamSynchronize(stream));
}
