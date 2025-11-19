#ifndef BACKGROUND_CONSOLIDATE_H
#define BACKGROUND_CONSOLIDATE_H

#include "../vamana.h"
#include "deleteList.h"
#include "lockfree_graph.cuh"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>

/**
 * Background CPU Consolidation for FreshDiskANN
 *
 * Runs consolidation on CPU in background while GPU continues
 * processing INSERT/DELETE/QUERY operations.
 *
 * How it works:
 * 1. GPU detects consolidation threshold reached
 * 2. GPU copies affected nodes + graph to CPU
 * 3. CPU rebuilds edges in background thread
 * 4. GPU continues operations (queries filter deleted points)
 * 5. When CPU done, apply updates back to GPU with version protection
 *
 * Benefits:
 * - No stop-the-world pause
 * - GPU maintains high throughput
 * - CPU handles expensive edge rebuilding
 */

class BackgroundConsolidator {
public:
    BackgroundConsolidator(unsigned numPoints, unsigned dimensions,
                           unsigned maxDegree, float alpha);
    ~BackgroundConsolidator();

    /**
     * Start background consolidation
     *
     * @param d_graph GPU graph
     * @param d_versions Version array
     * @param deleteList Delete list with marked deletions
     * @return true if consolidation started, false if already running
     */
    bool startConsolidation(uint8_t* d_graph,
                            unsigned* d_versions,
                            DeleteList* deleteList);

    /**
     * Check if consolidation is complete
     */
    bool isComplete() const { return !consolidating.load(); }

    /**
     * Wait for current consolidation to complete
     */
    void waitForCompletion();

    /**
     * Get number of nodes rebuilt in last consolidation
     */
    unsigned getLastRebuiltCount() const { return lastRebuiltCount; }

    /**
     * Get time of last consolidation in milliseconds
     */
    double getLastConsolidationTimeMs() const { return lastConsolidationTimeMs; }

private:
    // Graph parameters
    unsigned numPoints, dimensions, maxDegree;
    float alpha;
    unsigned graphEntrySize;

    // CPU-side graph copy
    uint8_t* h_graph;
    unsigned* h_deleted;
    std::vector<unsigned> affectedNodes;

    // GPU pointers (set during consolidation)
    uint8_t* d_graph;
    unsigned* d_versions;
    DeleteList* deleteList;

    // Background thread
    std::thread consolidationThread;
    std::atomic<bool> consolidating{false};
    std::atomic<bool> shutdownRequested{false};
    std::mutex mutex;
    std::condition_variable cv;

    // Statistics
    unsigned lastRebuiltCount;
    double lastConsolidationTimeMs;

    // Worker function
    void consolidationWorker();

    // CPU-side algorithms
    void findAffectedNodesCPU();
    void rebuildNodeEdgesCPU(unsigned nodeId);
    float computeDistanceCPU(unsigned id1, unsigned id2);
    void robustPruneCPU(unsigned nodeId, std::vector<unsigned>& candidates,
                        std::vector<unsigned>& outNeighbors);
    void applyUpdatesToGPU();
};

/**
 * CPU-side distance computation
 */
inline float computeL2DistanceCPU(float* vec1, float* vec2, unsigned dim) {
    float dist = 0.0f;
    for (unsigned i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    return dist;
}

#endif // BACKGROUND_CONSOLIDATE_H
