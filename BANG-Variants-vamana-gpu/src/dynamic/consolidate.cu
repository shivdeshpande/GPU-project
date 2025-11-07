#include "consolidate.h"
#include "insert.h"

/**
 * GPU kernel to find affected nodes
 *
 * Each thread checks one node to see if it has deleted neighbors.
 * If yes, atomically adds it to the affected list.
 */
__global__ void findAffectedNodesKernel(uint8_t* d_graph,
                                         unsigned int* d_deleted,
                                         unsigned* d_affectedNodes,
                                         unsigned* d_affectedCount,
                                         unsigned numPoints,
                                         unsigned graphEntrySize) {
    unsigned nodeId = blockIdx.x * blockDim.x + threadIdx.x;

    if (nodeId >= numPoints) return;

    // Skip if this node itself is deleted
    if (d_deleted[nodeId] != 0) return;

    // Get neighbor list for this node
    uint8_t* graphEntry = d_graph + nodeId * graphEntrySize;

    // Skip vector part (D floats)
    unsigned* neighborCount = (unsigned*)(graphEntry + D * sizeof(float));
    unsigned* neighbors = (unsigned*)(graphEntry + D * sizeof(float) + sizeof(unsigned));

    // Check if any neighbor is deleted
    bool hasDeletedNeighbor = false;
    for (unsigned i = 0; i < *neighborCount && i < R; i++) {
        unsigned neighborId = neighbors[i];
        if (d_deleted[neighborId] != 0) {
            hasDeletedNeighbor = true;
            break;
        }
    }

    // If affected, add to list
    if (hasDeletedNeighbor) {
        unsigned idx = atomicAdd(d_affectedCount, 1);
        d_affectedNodes[idx] = nodeId;
    }
}

/**
 * Find all affected nodes
 */
void findAffectedNodes(uint8_t* d_graph,
                       DeleteList* deleteList,
                       unsigned* d_affectedNodes,
                       unsigned* d_affectedCount) {

    // Reset counter
    gpuErrchk(cudaMemset(d_affectedCount, 0, sizeof(unsigned)));

    // Launch kernel
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    findAffectedNodesKernel<<<numBlocks, threadsPerBlock>>>(
        d_graph,
        deleteList->getDevicePointer(),
        d_affectedNodes,
        d_affectedCount,
        N,
        graphEntrySize
    );

    gpuErrchk(cudaDeviceSynchronize());
}

/**
 * Rebuild neighbor lists for affected nodes
 *
 * For each affected node, we need to:
 * 1. Extract its vector
 * 2. Run GreedySearch to find candidates (filtering deleted points)
 * 3. Use RobustPrune to select R neighbors
 * 4. Update reverse edges
 */
void rebuildAffectedNodes(uint8_t* d_graph,
                          unsigned* d_affectedNodes,
                          unsigned affectedCount,
                          DeleteList* deleteList,
                          float alpha) {

    if (affectedCount == 0) return;

    // Allocate GPU memory for batch processing
    unsigned* d_visitedSets;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    float* d_vectors;

    gpuErrchk(cudaMalloc(&d_visitedSets, affectedCount * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, affectedCount * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMalloc(&d_vectors, affectedCount * D * sizeof(float)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, affectedCount * sizeof(unsigned)));

    // Copy affected node IDs to host for iteration
    unsigned* h_affectedNodes = (unsigned*)malloc(affectedCount * sizeof(unsigned));
    gpuErrchk(cudaMemcpy(h_affectedNodes, d_affectedNodes, affectedCount * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    // Process each affected node
    for (unsigned i = 0; i < affectedCount; i++) {
        unsigned nodeId = h_affectedNodes[i];
        float* d_vector = d_vectors + i * D;
        unsigned* d_visitedSet = d_visitedSets + i * MAX_PARENTS_PERQUERY;
        unsigned* d_count = d_visitedSetCount + i;

        // Extract vector from graph
        extractVectorFromGraph(d_graph, d_vector, nodeId);

        // Run GreedySearch to find candidates (filtering deleted points)
        greedySearch(d_graph,
                     d_vector,
                     d_visitedSet,
                     d_count,
                     nodeId,
                     1,
                     L, // searchL = L (default)
                     deleteList->getDevicePointer());

        // Recompute neighbors with RobustPrune
        computeOutNeighbors(d_graph,
                            d_vector,
                            d_visitedSet,
                            d_count,
                            alpha,
                            d_reverseEdgeIndex,
                            nodeId,
                            1);
    }

    // Update reverse edges for all affected nodes
    computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);

    // Cleanup
    free(h_affectedNodes);
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
    gpuErrchk(cudaFree(d_vectors));
}

/**
 * Check if consolidation should be triggered
 */
bool shouldConsolidate(DeleteList* deleteList,
                       unsigned totalPoints,
                       float thresholdPercent) {
    unsigned deleteCount = deleteList->getDeleteCount();
    unsigned threshold = (unsigned)((float)totalPoints * thresholdPercent / 100.0f);
    return deleteCount >= threshold;
}

/**
 * Main consolidate function
 *
 * Implements Algorithm 4 from FreshDiskANN paper.
 */
unsigned consolidateDeletes(uint8_t* d_graph,
                            DeleteList* deleteList,
                            float alpha,
                            bool verbose) {

    unsigned deleteCount = deleteList->getDeleteCount();

    if (deleteCount == 0) {
        if (verbose) {
            printf("No deletions to consolidate.\n");
        }
        return 0;
    }

    if (verbose) {
        printf("\n╔════════════════════════════════════════════╗\n");
        printf("║   Consolidating %6u Deletions           ║\n", deleteCount);
        printf("╚════════════════════════════════════════════╝\n");
    }

    // Step 1: Find affected nodes (those with deleted neighbors)
    unsigned* d_affectedNodes;
    unsigned* d_affectedCount;

    gpuErrchk(cudaMalloc(&d_affectedNodes, N * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_affectedCount, sizeof(unsigned)));

    if (verbose) printf("Finding affected nodes...\n");

    findAffectedNodes(d_graph, deleteList, d_affectedNodes, d_affectedCount);

    unsigned affectedCount;
    gpuErrchk(cudaMemcpy(&affectedCount, d_affectedCount, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    if (verbose) {
        printf("✓ Found %u affected nodes (%.2f%% of total)\n",
               affectedCount, (float)affectedCount / N * 100.0f);
    }

    // Step 2: Rebuild neighbor lists for affected nodes
    if (affectedCount > 0) {
        if (verbose) printf("Rebuilding neighbor lists...\n");

        rebuildAffectedNodes(d_graph, d_affectedNodes, affectedCount, deleteList, alpha);

        if (verbose) printf("✓ Rebuilt %u neighbor lists\n", affectedCount);
    }

    // Step 3: Clear DeleteList
    deleteList->clear();

    if (verbose) {
        printf("✓ DeleteList cleared\n");
        printf("Consolidation complete!\n\n");
    }

    // Cleanup
    gpuErrchk(cudaFree(d_affectedNodes));
    gpuErrchk(cudaFree(d_affectedCount));

    return affectedCount;
}
