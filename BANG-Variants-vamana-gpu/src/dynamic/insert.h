#ifndef INSERT_H
#define INSERT_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "../vamana.h"
#include "deleteList.h"

/**
 * FreshDiskANN Dynamic Insert
 *
 * Inserts new points into the graph using RobustPrune (α-RNG property)
 * to maintain graph connectivity and search quality.
 *
 * Key Features:
 * - Reuses existing VAMANA infrastructure (greedySearch + computeOutNeighbors)
 * - Maintains α-RNG property for stability
 * - GPU-accelerated batch inserts
 * - Automatically updates reverse edges
 */

/**
 * Insert a single point into the graph
 *
 * Algorithm:
 * 1. Run GreedySearch from medoid to build candidate set (visited set)
 * 2. Use existing computeOutNeighbors (calls pruneOutNeighbors internally)
 *    to select R neighbors with α-RNG property
 * 3. Update reverse edges to maintain bidirectional connectivity
 *
 * @param d_graph GPU graph structure
 * @param d_newVector New point vector (D dimensions)
 * @param newPointId Point ID for the new point
 * @param alpha α parameter for RobustPrune (typically 1.2)
 * @param medoid Medoid node ID for search starting point
 */
void insertPoint(uint8_t* d_graph,
                 float* d_newVector,
                 unsigned newPointId,
                 float alpha,
                 unsigned medoid = MEDOID);

/**
 * Insert a single point with version-based concurrency control
 *
 * Uses version numbers for lock-free concurrent access.
 * Multiple inserts can run in parallel safely.
 *
 * @param d_graph GPU graph structure
 * @param d_versions Version array for lock-free access
 * @param d_newVector New point vector (D dimensions)
 * @param newPointId Point ID for the new point
 * @param alpha α parameter for RobustPrune (typically 1.2)
 * @param medoid Medoid node ID for search starting point
 */
void insertPointVersioned(uint8_t* d_graph,
                          unsigned* d_versions,
                          float* d_newVector,
                          unsigned newPointId,
                          float alpha,
                          unsigned medoid = MEDOID);

/**
 * Pre-allocated buffers for insert operations (avoid cudaMalloc per insert)
 */
struct InsertBuffers {
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    GreedySearchBuffers gsBuffers;       // Pre-allocated greedy search buffers
    OutNeighborsBuffers outNbrsBuffers;  // Pre-allocated out-neighbors buffers
    bool allocated;
};

/**
 * Allocate insert buffers (call once at startup)
 */
void allocateInsertBuffers(InsertBuffers* buffers);

/**
 * Free insert buffers (call at shutdown)
 */
void freeInsertBuffers(InsertBuffers* buffers);

/**
 * Insert with pre-allocated buffers (much faster - no cudaMalloc overhead)
 */
void insertPointVersionedPrealloc(uint8_t* d_graph,
                                   unsigned* d_versions,
                                   float* d_newVector,
                                   unsigned newPointId,
                                   float alpha,
                                   InsertBuffers* buffers,
                                   cudaStream_t stream = 0,
                                   unsigned medoid = MEDOID);

/**
 * Insert multiple points in batch (more efficient)
 *
 * @param d_graph GPU graph structure
 * @param d_newVectors New point vectors (batchSize x D dimensions)
 * @param newPointIds Array of point IDs
 * @param batchSize Number of points to insert
 * @param alpha α parameter for RobustPrune (typically 1.2)
 * @param medoid Medoid node ID for search starting point
 */
void batchInsertPoints(uint8_t* d_graph,
                       float* d_newVectors,
                       unsigned* newPointIds,
                       unsigned batchSize,
                       float alpha,
                       unsigned medoid = MEDOID);

/**
 * Reinsert a deleted point (for consolidation)
 *
 * Similar to insertPoint but handles previously deleted points
 * during consolidation operations.
 *
 * @param d_graph GPU graph structure
 * @param pointId Point ID to reinsert
 * @param alpha α parameter for RobustPrune
 * @param deleteList DeleteList to check/update deletion status
 */
void reinsertPoint(uint8_t* d_graph,
                   unsigned pointId,
                   float alpha,
                   DeleteList* deleteList);

/**
 * Helper: Copy vector into graph at specified position
 *
 * @param d_graph GPU graph structure
 * @param d_vector Vector data (D dimensions)
 * @param pointId Target point ID
 */
void copyVectorToGraph(uint8_t* d_graph,
                       float* d_vector,
                       unsigned pointId);

/**
 * Helper: Extract vector from graph at specified position
 *
 * @param d_graph GPU graph structure
 * @param d_vector Output buffer (D dimensions)
 * @param pointId Source point ID
 */
void extractVectorFromGraph(uint8_t* d_graph,
                             float* d_vector,
                             unsigned pointId);

#endif // INSERT_H
