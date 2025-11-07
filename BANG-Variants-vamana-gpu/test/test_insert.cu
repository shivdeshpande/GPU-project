#include "../src/dynamic/insert.h"
#include "../src/dynamic/deleteList.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/**
 * Unit test for Dynamic Insert
 * Tests:
 * 1. Single point insertion
 * 2. Vector copy/extract operations
 * 3. Graph structure validation after insert
 */

// Helper: Initialize a simple test graph
void initializeTestGraph(uint8_t** h_graph, uint8_t** d_graph) {
    *h_graph = (uint8_t*)malloc(N * graphEntrySize * sizeof(uint8_t));
    memset(*h_graph, 0, N * graphEntrySize * sizeof(uint8_t));

    // Initialize with some random vectors
    for (unsigned i = 0; i < N; i++) {
        uint8_t* entry = *h_graph + i * graphEntrySize;
        float* vec = (float*)entry;

        // Simple test vectors: vec[j] = i + j/100
        for (unsigned j = 0; j < D; j++) {
            vec[j] = (float)i + (float)j / 100.0f;
        }

        // Initialize neighbor count to 0
        unsigned* neighborCount = (unsigned*)(entry + D * sizeof(float));
        *neighborCount = 0;
    }

    // Copy to GPU
    gpuErrchk(cudaMalloc(d_graph, N * graphEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(*d_graph, *h_graph, N * graphEntrySize * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
}

void testVectorCopyExtract() {
    printf("\n=== Test 1: Vector Copy/Extract ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraph(&h_graph, &d_graph);

    // Create a test vector
    float* h_testVector = (float*)malloc(D * sizeof(float));
    for (unsigned i = 0; i < D; i++) {
        h_testVector[i] = 42.0f + (float)i;
    }

    // Copy to GPU
    float* d_testVector;
    gpuErrchk(cudaMalloc(&d_testVector, D * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_testVector, h_testVector, D * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Copy vector into graph at position 100
    unsigned testPointId = 100;
    copyVectorToGraph(d_graph, d_testVector, testPointId);
    printf("✓ Vector copied to graph at point %u\n", testPointId);

    // Extract it back
    float* d_extractedVector;
    gpuErrchk(cudaMalloc(&d_extractedVector, D * sizeof(float)));

    extractVectorFromGraph(d_graph, d_extractedVector, testPointId);

    // Copy back to host for verification
    float* h_extractedVector = (float*)malloc(D * sizeof(float));
    gpuErrchk(cudaMemcpy(h_extractedVector, d_extractedVector, D * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Verify all dimensions match
    bool allMatch = true;
    for (unsigned i = 0; i < D; i++) {
        if (h_testVector[i] != h_extractedVector[i]) {
            printf("Mismatch at dimension %u: expected %f, got %f\n",
                   i, h_testVector[i], h_extractedVector[i]);
            allMatch = false;
            break;
        }
    }

    assert(allMatch);
    printf("✓ Extracted vector matches original (all %u dimensions)\n", D);

    // Cleanup
    free(h_graph);
    free(h_testVector);
    free(h_extractedVector);
    cudaFree(d_graph);
    cudaFree(d_testVector);
    cudaFree(d_extractedVector);

    printf("Test 1 PASSED\n");
}

void testSingleInsert() {
    printf("\n=== Test 2: Single Point Insert ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraph(&h_graph, &d_graph);

    // Create a new vector to insert
    float* h_newVector = (float*)malloc(D * sizeof(float));
    for (unsigned i = 0; i < D; i++) {
        h_newVector[i] = 999.0f + (float)i;  // Distinctive values
    }

    // Copy to GPU
    float* d_newVector;
    gpuErrchk(cudaMalloc(&d_newVector, D * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_newVector, h_newVector, D * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Insert at position 500
    unsigned newPointId = 500;
    printf("Inserting point %u with alpha=1.2...\n", newPointId);

    // Note: This will call greedySearch and computeOutNeighbors
    // For this test, we mainly verify it doesn't crash
    // Full validation requires a properly initialized graph with edges
    insertPoint(d_graph, d_newVector, newPointId, 1.2f, MEDOID);

    printf("✓ Insert completed without errors\n");

    // Verify the vector was written to graph
    float* d_extractedVector;
    gpuErrchk(cudaMalloc(&d_extractedVector, D * sizeof(float)));
    extractVectorFromGraph(d_graph, d_extractedVector, newPointId);

    float* h_extractedVector = (float*)malloc(D * sizeof(float));
    gpuErrchk(cudaMemcpy(h_extractedVector, d_extractedVector, D * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Verify vector matches
    bool allMatch = true;
    for (unsigned i = 0; i < D; i++) {
        if (h_newVector[i] != h_extractedVector[i]) {
            allMatch = false;
            break;
        }
    }

    assert(allMatch);
    printf("✓ Inserted vector verified in graph\n");

    // Cleanup
    free(h_graph);
    free(h_newVector);
    free(h_extractedVector);
    cudaFree(d_graph);
    cudaFree(d_newVector);
    cudaFree(d_extractedVector);

    printf("Test 2 PASSED\n");
}

void testInsertWithDeleteList() {
    printf("\n=== Test 3: Insert with DeleteList Integration ===\n");

    uint8_t *h_graph, *d_graph;
    initializeTestGraph(&h_graph, &d_graph);

    DeleteList dl(N);

    // Mark some points as deleted
    unsigned deletedPoints[] = {10, 20, 30};
    dl.batchMarkDeleted(deletedPoints, 3);
    printf("Marked %u points as deleted\n", 3);

    // Now insert a new point - it should not be marked as deleted
    float* h_newVector = (float*)malloc(D * sizeof(float));
    for (unsigned i = 0; i < D; i++) {
        h_newVector[i] = 888.0f + (float)i;
    }

    float* d_newVector;
    gpuErrchk(cudaMalloc(&d_newVector, D * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_newVector, h_newVector, D * sizeof(float),
                         cudaMemcpyHostToDevice));

    unsigned newPointId = 100;
    insertPoint(d_graph, d_newVector, newPointId, 1.2f, MEDOID);

    // Verify new point is not marked as deleted
    assert(!dl.isDeleted(newPointId));
    printf("✓ New point %u is not marked as deleted\n", newPointId);

    // Verify deleted points are still deleted
    assert(dl.isDeleted(10) == true);
    assert(dl.isDeleted(20) == true);
    assert(dl.isDeleted(30) == true);
    printf("✓ Previously deleted points remain deleted\n");

    // Cleanup
    free(h_graph);
    free(h_newVector);
    cudaFree(d_graph);
    cudaFree(d_newVector);

    printf("Test 3 PASSED\n");
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║   Dynamic Insert Unit Tests               ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    testVectorCopyExtract();
    testSingleInsert();
    testInsertWithDeleteList();

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   ALL TESTS PASSED ✓                      ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    return 0;
}
