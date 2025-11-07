#ifndef CONSOLIDATE_H
#define CONSOLIDATE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "../vamana.h"
#include "deleteList.h"

/**
 * FreshDiskANN Consolidate Deletes (Algorithm 4)
 *
 * Batch updates the graph to remove deleted points when the deletion
 * threshold is reached (typically 5-10% of the index).
 *
 * Key Features:
 * - Identifies affected nodes (those with deleted neighbors)
 * - Rebuilds neighbor lists using RobustPrune
 * - Maintains α-RNG property for stability
 * - Clears DeleteList after consolidation
 *
 * Performance:
 * - Much faster than full rebuild
 * - Amortizes cost over many deletions
 * - GPU-accelerated for large batches
 */

/**
 * Consolidate all deletions in the DeleteList
 *
 * Algorithm:
 * 1. Find all affected nodes (non-deleted nodes with ≥1 deleted neighbor)
 * 2. For each affected node:
 *    a. Run GreedySearch to build candidate set (excluding deleted points)
 *    b. Call pruneOutNeighbors (RobustPrune) to select R neighbors
 *    c. Update reverse edges
 * 3. Clear DeleteList
 *
 * @param d_graph GPU graph structure
 * @param deleteList DeleteList with points to consolidate
 * @param alpha α parameter for RobustPrune (typically 1.2)
 * @param verbose Print progress information
 * @return Number of nodes that were updated
 */
unsigned consolidateDeletes(uint8_t* d_graph,
                            DeleteList* deleteList,
                            float alpha,
                            bool verbose = true);

/**
 * Check if consolidation should be triggered
 *
 * @param deleteList DeleteList to check
 * @param totalPoints Total number of points in index
 * @param thresholdPercent Threshold as percentage (default 5.0%)
 * @return true if deletion count >= threshold
 */
bool shouldConsolidate(DeleteList* deleteList,
                       unsigned totalPoints,
                       float thresholdPercent = 5.0f);

/**
 * Find all affected nodes (nodes with deleted neighbors)
 *
 * Scans the graph to find non-deleted nodes that have at least
 * one deleted neighbor in their adjacency list.
 *
 * @param d_graph GPU graph structure
 * @param deleteList DeleteList to check deletions
 * @param d_affectedNodes Output array of affected node IDs
 * @param d_affectedCount Output count of affected nodes
 */
void findAffectedNodes(uint8_t* d_graph,
                       DeleteList* deleteList,
                       unsigned* d_affectedNodes,
                       unsigned* d_affectedCount);

/**
 * GPU kernel to identify affected nodes
 *
 * Checks each node's neighbor list for deleted points.
 * If found, marks node as affected.
 *
 * @param d_graph GPU graph structure
 * @param d_deleted DeleteList bitvector
 * @param d_affectedNodes Output array for affected node IDs
 * @param d_affectedCount Atomic counter for affected nodes
 * @param numPoints Total number of points to check
 * @param graphEntrySize Size of each graph entry
 */
__global__ void findAffectedNodesKernel(uint8_t* d_graph,
                                         unsigned int* d_deleted,
                                         unsigned* d_affectedNodes,
                                         unsigned* d_affectedCount,
                                         unsigned numPoints,
                                         unsigned graphEntrySize);

/**
 * Rebuild neighbor lists for affected nodes
 *
 * For each affected node, runs GreedySearch and RobustPrune
 * to select new neighbors (excluding deleted points).
 *
 * @param d_graph GPU graph structure
 * @param d_affectedNodes Array of affected node IDs
 * @param affectedCount Number of affected nodes
 * @param deleteList DeleteList to filter deleted points
 * @param alpha α parameter for RobustPrune
 */
void rebuildAffectedNodes(uint8_t* d_graph,
                          unsigned* d_affectedNodes,
                          unsigned affectedCount,
                          DeleteList* deleteList,
                          float alpha);

#endif // CONSOLIDATE_H
