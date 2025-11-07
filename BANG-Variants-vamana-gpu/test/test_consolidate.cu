#include "../src/dynamic/consolidate.h"
#include "../src/dynamic/insert.h"
#include "../src/dynamic/deleteList.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/**
 * Unit test for Consolidate Deletes
 * Tests:
 * 1. shouldConsolidate threshold logic
 * 2. Finding affected nodes
 * 3. Full consolidation workflow
 */

// Helper: Initialize a test graph with some edges
void initializeTestGraphWithEdges(uint8_t** h_graph, uint8_t** d_graph) {
    *h_graph = (uint8_t*)malloc(N * graphEntrySize * sizeof(uint8_t));
    memset(*h_graph, 0, N * graphEntrySize * sizeof(uint8_t));

    // Initialize with test vectors and some edges
    for (unsigned i = 0; i < N; i++) {
        uint8_t* entry = *h_graph + i * graphEntrySize;
        float* vec = (float*)entry;

        // Simple test vectors
        for (unsigned j = 0; j < D; j++) {
            vec[j] = (float)i + (float)j / 100.0f;
        }

        // Add some neighbors (up to R/2 neighbors per node)
        unsigned* neighborCount = (unsigned*)(entry + D * sizeof(float));
        unsigned* neighbors = (unsigned*)(entry + D * sizeof(float) + sizeof(unsigned));

        // Add neighbors: for node i, connect to nodes (i+1) % N, (i+2) % N, etc.
        unsigned numNeighbors = (i < N - 5) ? 5 : 0;
        *neighborCount = numNeighbors;

        for (unsigned j = 0; j < numNeighbors; j++) {
            neighbors[j] = (i + j + 1) % N;
        }
    }

    // Copy to GPU
    gpuErrchk(cudaMalloc(d_graph, N * graphEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(*d_graph, *h_graph, N * graphEntrySize * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
}

void testShouldConsolidate() {
    printf("\n=== Test 1: shouldConsolidate Threshold Logic ===\n");

    DeleteList dl(10000);

    // Test with no deletions
    assert(shouldConsolidate(&dl, 10000, 5.0f) == false);
    printf("✓ No consolidation needed with 0 deletions\n");

    // Add 499 deletions (4.99% of 10000)
    unsigned* pointsToDelete = (unsigned*)malloc(499 * sizeof(unsigned));
    for (unsigned i = 0; i < 499; i++) {
        pointsToDelete[i] = i;
    }
    dl.batchMarkDeleted(pointsToDelete, 499);

    assert(shouldConsolidate(&dl, 10000, 5.0f) == false);
    printf("✓ No consolidation at 4.99%% (below 5%% threshold)\n");

    // Add one more deletion to reach 500 (5.0%)
    dl.markDeleted(499);

    assert(shouldConsolidate(&dl, 10000, 5.0f) == true);
    printf("✓ Consolidation triggered at 5.0%% threshold\n");

    free(pointsToDelete);
    printf("Test 1 PASSED\n");
}

void testFindAffectedNodes() {
    printf("\n=== Test 2: Finding Affected Nodes ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraphWithEdges(&h_graph, &d_graph);

    DeleteList dl(N);

    // Mark some nodes as deleted
    // Node 0 has neighbors: 1, 2, 3, 4, 5
    // So if we delete node 1, then node 0 should be affected
    unsigned deletedNodes[] = {1, 2, 100};
    dl.batchMarkDeleted(deletedNodes, 3);
    printf("Marked nodes 1, 2, 100 as deleted\n");

    // Find affected nodes
    unsigned* d_affectedNodes;
    unsigned* d_affectedCount;

    gpuErrchk(cudaMalloc(&d_affectedNodes, N * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_affectedCount, sizeof(unsigned)));

    findAffectedNodes(d_graph, &dl, d_affectedNodes, d_affectedCount);

    unsigned affectedCount;
    gpuErrchk(cudaMemcpy(&affectedCount, d_affectedCount, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    printf("Found %u affected nodes\n", affectedCount);

    // Verify that node 0 is affected (has deleted neighbor 1)
    // And other nodes with deleted neighbors are also affected
    assert(affectedCount > 0);
    printf("✓ At least one node is affected\n");

    // Cleanup
    free(h_graph);
    cudaFree(d_graph);
    cudaFree(d_affectedNodes);
    cudaFree(d_affectedCount);

    printf("Test 2 PASSED\n");
}

void testFullConsolidation() {
    printf("\n=== Test 3: Full Consolidation Workflow ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraphWithEdges(&h_graph, &d_graph);

    DeleteList dl(N);

    // Mark 600 nodes as deleted (6% of 10000)
    unsigned* deletedNodes = (unsigned*)malloc(600 * sizeof(unsigned));
    for (unsigned i = 0; i < 600; i++) {
        deletedNodes[i] = i * 10;  // Delete every 10th node
    }
    dl.batchMarkDeleted(deletedNodes, 600);

    printf("Marked 600 nodes as deleted (6.0%%)\n");

    // Verify consolidation is needed
    assert(shouldConsolidate(&dl, N, 5.0f) == true);
    printf("✓ Consolidation threshold reached\n");

    unsigned deleteCountBefore = dl.getDeleteCount();
    assert(deleteCountBefore == 600);

    // Run consolidation
    unsigned affectedCount = consolidateDeletes(d_graph, &dl, 1.2f, true);

    printf("Consolidation updated %u nodes\n", affectedCount);

    // Verify DeleteList was cleared
    unsigned deleteCountAfter = dl.getDeleteCount();
    assert(deleteCountAfter == 0);
    printf("✓ DeleteList cleared after consolidation\n");

    // Cleanup
    free(h_graph);
    free(deletedNodes);
    cudaFree(d_graph);

    printf("Test 3 PASSED\n");
}

void testEmptyConsolidation() {
    printf("\n=== Test 4: Consolidation with No Deletions ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraphWithEdges(&h_graph, &d_graph);

    DeleteList dl(N);

    // Run consolidation with no deletions
    unsigned affectedCount = consolidateDeletes(d_graph, &dl, 1.2f, true);

    assert(affectedCount == 0);
    printf("✓ No nodes updated when no deletions\n");

    // Cleanup
    free(h_graph);
    cudaFree(d_graph);

    printf("Test 4 PASSED\n");
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║   Consolidate Deletes Unit Tests          ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    testShouldConsolidate();
    testFindAffectedNodes();
    testFullConsolidation();
    testEmptyConsolidation();

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   ALL TESTS PASSED ✓                      ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    return 0;
}
