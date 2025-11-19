#ifndef LOCKFREE_GRAPH_CUH
#define LOCKFREE_GRAPH_CUH

#include <cuda_runtime.h>

/**
 * Lock-Free Graph Operations for Concurrent FreshDiskANN
 *
 * Technique: Version Numbers (Seqlock Pattern)
 *
 * How it works:
 * - Each vertex has a version counter
 * - Writers: increment version BEFORE and AFTER modification
 * - Readers: check version before and after read
 * - If versions differ OR version is odd (write in progress), retry
 *
 * Benefits:
 * - No locks, pure atomic operations
 * - Zero contention for readers (just read and check)
 * - Writers never block
 * - Retry only on actual conflict (very rare)
 */

// Version array - one per vertex
// Allocated alongside graph: cudaMalloc(&d_versions, N * sizeof(unsigned))
// Odd version = write in progress, Even = stable

/**
 * Graph entry layout WITH version:
 *
 * | Version (4B) | Vector (D*4B) | Degree (4B) | Neighbors (R*4B) |
 *
 * Total: 4 + 512 + 4 + 256 = 776 bytes (was 772)
 *
 * OR keep version in separate array (better cache behavior for reads)
 */

//=============================================================================
// WRITER FUNCTIONS (for INSERT operations)
//=============================================================================

/**
 * Begin write operation on vertex
 * Increments version to odd number (signals write in progress)
 */
__device__ __forceinline__
void beginWrite(unsigned* d_versions, unsigned vertexId) {
    atomicAdd(&d_versions[vertexId], 1);  // Even -> Odd
    __threadfence();  // Ensure version increment visible before writes
}

/**
 * End write operation on vertex
 * Increments version to even number (signals write complete)
 */
__device__ __forceinline__
void endWrite(unsigned* d_versions, unsigned vertexId) {
    __threadfence();  // Ensure all writes visible before version increment
    atomicAdd(&d_versions[vertexId], 1);  // Odd -> Even
}

/**
 * Atomically add a neighbor with version protection
 *
 * Usage:
 *   beginWrite(d_versions, vertexId);
 *   addNeighborVersioned(...);
 *   endWrite(d_versions, vertexId);
 */
__device__ __forceinline__
bool addNeighborVersioned(uint8_t* d_graph,
                          unsigned vertexId,
                          unsigned neighborId,
                          unsigned graphEntrySize,
                          unsigned dim,
                          unsigned maxDegree) {
    uint8_t* entry = d_graph + vertexId * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned* neighbors = degree + 1;

    unsigned currentDegree = *degree;
    if (currentDegree >= maxDegree) {
        return false;  // Full
    }

    // Write neighbor first
    neighbors[currentDegree] = neighborId;
    __threadfence();

    // Then increment degree
    *degree = currentDegree + 1;

    return true;
}

/**
 * Batch write all neighbors with version protection
 * Most efficient for RobustPrune output
 */
__device__ __forceinline__
void writeAllNeighborsVersioned(uint8_t* d_graph,
                                 unsigned* d_versions,
                                 unsigned vertexId,
                                 unsigned* newNeighbors,
                                 unsigned newDegree,
                                 unsigned graphEntrySize,
                                 unsigned dim,
                                 unsigned maxDegree) {
    beginWrite(d_versions, vertexId);

    uint8_t* entry = d_graph + vertexId * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned* neighbors = degree + 1;

    // Write all neighbors
    unsigned count = min(newDegree, maxDegree);
    for (unsigned i = 0; i < count; i++) {
        neighbors[i] = newNeighbors[i];
    }
    __threadfence();

    // Update degree
    *degree = count;

    endWrite(d_versions, vertexId);
}

//=============================================================================
// READER FUNCTIONS (for QUERY operations)
//=============================================================================

/**
 * Read neighbors with version check
 *
 * Returns: degree if successful, UINT_MAX if need retry
 *
 * Pattern:
 *   do {
 *       degree = readNeighborsVersioned(...);
 *   } while (degree == UINT_MAX);
 */
__device__ __forceinline__
unsigned readNeighborsVersioned(uint8_t* d_graph,
                                 unsigned* d_versions,
                                 unsigned vertexId,
                                 unsigned* localNeighbors,  // Output array
                                 unsigned maxNeighbors,
                                 unsigned graphEntrySize,
                                 unsigned dim) {
    // Read version (must be even = no write in progress)
    unsigned v1 = d_versions[vertexId];
    if (v1 & 1) {
        return UINT_MAX;  // Write in progress, retry
    }
    __threadfence();

    // Read data
    uint8_t* entry = d_graph + vertexId * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned* neighbors = degree + 1;

    unsigned deg = *degree;
    unsigned count = min(deg, maxNeighbors);

    for (unsigned i = 0; i < count; i++) {
        localNeighbors[i] = neighbors[i];
    }

    __threadfence();

    // Check version again
    unsigned v2 = d_versions[vertexId];
    if (v1 != v2) {
        return UINT_MAX;  // Data changed during read, retry
    }

    return count;
}

/**
 * Read degree only (faster than full neighbor read)
 */
__device__ __forceinline__
unsigned readDegreeVersioned(uint8_t* d_graph,
                              unsigned* d_versions,
                              unsigned vertexId,
                              unsigned graphEntrySize,
                              unsigned dim) {
    unsigned v1 = d_versions[vertexId];
    if (v1 & 1) return UINT_MAX;
    __threadfence();

    uint8_t* entry = d_graph + vertexId * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned deg = *degree;

    __threadfence();
    unsigned v2 = d_versions[vertexId];

    return (v1 == v2) ? deg : UINT_MAX;
}

/**
 * Optimistic read with bounded retries
 * Falls back to accepting potentially stale data after max retries
 */
__device__ __forceinline__
unsigned readNeighborsOptimistic(uint8_t* d_graph,
                                  unsigned* d_versions,
                                  unsigned vertexId,
                                  unsigned* localNeighbors,
                                  unsigned maxNeighbors,
                                  unsigned graphEntrySize,
                                  unsigned dim,
                                  unsigned maxRetries = 3) {
    for (unsigned retry = 0; retry < maxRetries; retry++) {
        unsigned result = readNeighborsVersioned(d_graph, d_versions, vertexId,
                                                  localNeighbors, maxNeighbors,
                                                  graphEntrySize, dim);
        if (result != UINT_MAX) {
            return result;
        }
        // Small backoff - busy wait
        for (volatile int k = 0; k < 100 * (retry + 1); k++) {}
    }

    // Fallback: read without version check (accept potentially stale)
    uint8_t* entry = d_graph + vertexId * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned* neighbors = degree + 1;

    unsigned deg = *degree;
    unsigned count = min(deg, maxNeighbors);
    for (unsigned i = 0; i < count; i++) {
        localNeighbors[i] = neighbors[i];
    }
    return count;
}

//=============================================================================
// REVERSE EDGE UPDATES (for INSERT's reverse edge phase)
//=============================================================================

/**
 * Add reverse edge with version protection
 *
 * When inserting P with neighbor N, we need to add P to N's adjacency list.
 * Multiple inserts may try to add edges to the same vertex N concurrently.
 *
 * This function handles that safely using version numbers.
 */
__device__ __forceinline__
bool addReverseEdgeVersioned(uint8_t* d_graph,
                              unsigned* d_versions,
                              unsigned targetVertex,    // target - receives the edge
                              unsigned sourceVertex,    // source - new point
                              unsigned graphEntrySize,
                              unsigned dim,
                              unsigned maxDegree) {
    // Begin write on target vertex
    beginWrite(d_versions, targetVertex);

    uint8_t* entry = d_graph + targetVertex * graphEntrySize;
    unsigned* degree = (unsigned*)(entry + dim * sizeof(float));
    unsigned* neighbors = degree + 1;

    bool success = false;
    unsigned currentDegree = *degree;

    if (currentDegree < maxDegree) {
        // Write new neighbor
        neighbors[currentDegree] = sourceVertex;
        __threadfence();

        // Increment degree
        *degree = currentDegree + 1;
        success = true;
    }

    endWrite(d_versions, targetVertex);
    return success;
}

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

/**
 * Check if vertex has neighbor (with version check)
 */
__device__ __forceinline__
bool hasNeighborVersioned(uint8_t* d_graph,
                           unsigned* d_versions,
                           unsigned vertexId,
                           unsigned neighborId,
                           unsigned graphEntrySize,
                           unsigned dim,
                           unsigned maxDegree) {
    unsigned localNeighbors[64];  // Assuming maxDegree <= 64
    unsigned count = readNeighborsOptimistic(d_graph, d_versions, vertexId,
                                              localNeighbors, maxDegree,
                                              graphEntrySize, dim);

    for (unsigned i = 0; i < count; i++) {
        if (localNeighbors[i] == neighborId) {
            return true;
        }
    }
    return false;
}

/**
 * Get vector pointer (no version needed - vectors don't change after insert)
 */
__device__ __forceinline__
float* getVector(uint8_t* d_graph, unsigned vertexId, unsigned graphEntrySize) {
    return (float*)(d_graph + vertexId * graphEntrySize);
}

/**
 * Compute L2 distance between query and vertex
 */
__device__ __forceinline__
float computeDistance(uint8_t* d_graph,
                      unsigned vertexId,
                      float* query,
                      unsigned graphEntrySize,
                      unsigned dim) {
    float* vec = getVector(d_graph, vertexId, graphEntrySize);
    float dist = 0.0f;
    for (unsigned i = 0; i < dim; i++) {
        float diff = query[i] - vec[i];
        dist += diff * diff;
    }
    return dist;
}

//=============================================================================
// HOST HELPER FUNCTIONS
//=============================================================================

/**
 * Allocate version array on GPU
 */
inline void allocateVersions(unsigned** d_versions, unsigned numPoints) {
    cudaMalloc(d_versions, numPoints * sizeof(unsigned));
    cudaMemset(*d_versions, 0, numPoints * sizeof(unsigned));  // All even = stable
}

/**
 * Free version array
 */
inline void freeVersions(unsigned* d_versions) {
    cudaFree(d_versions);
}

#endif // LOCKFREE_GRAPH_CUH
