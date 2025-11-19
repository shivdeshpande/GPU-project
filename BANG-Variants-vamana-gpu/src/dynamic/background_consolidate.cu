#include "background_consolidate.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <omp.h>

/**
 * Background CPU Consolidation Implementation
 */

BackgroundConsolidator::BackgroundConsolidator(unsigned numPoints, unsigned dimensions,
                                               unsigned maxDegree, float alpha)
    : numPoints(numPoints), dimensions(dimensions), maxDegree(maxDegree), alpha(alpha),
      d_graph(nullptr), d_versions(nullptr), deleteList(nullptr),
      lastRebuiltCount(0), lastConsolidationTimeMs(0.0) {

    graphEntrySize = this->dimensions * sizeof(float) + sizeof(unsigned) + this->maxDegree * sizeof(unsigned);

    // Allocate CPU-side graph copy
    h_graph = (uint8_t*)malloc(this->numPoints * graphEntrySize);
    h_deleted = (unsigned*)malloc(this->numPoints * sizeof(unsigned));

    if (!h_graph || !h_deleted) {
        fprintf(stderr, "Error: Failed to allocate CPU memory for consolidation\n");
        exit(1);
    }

    printf("BackgroundConsolidator initialized: %u points, %u dims, R=%u, α=%.2f\n",
           this->numPoints, this->dimensions, this->maxDegree, alpha);
}

BackgroundConsolidator::~BackgroundConsolidator() {
    // Signal shutdown
    shutdownRequested = true;
    cv.notify_all();

    // Wait for thread to finish
    if (consolidationThread.joinable()) {
        consolidationThread.join();
    }

    // Free CPU memory
    free(h_graph);
    free(h_deleted);
}

bool BackgroundConsolidator::startConsolidation(uint8_t* d_graph,
                                                 unsigned* d_versions,
                                                 DeleteList* deleteList) {
    // Check if already consolidating
    if (consolidating.load()) {
        return false;
    }

    // Store GPU pointers
    this->d_graph = d_graph;
    this->d_versions = d_versions;
    this->deleteList = deleteList;

    // Copy graph and delete list to CPU
    cudaMemcpy(h_graph, d_graph, numPoints * graphEntrySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_deleted, deleteList->getDevicePointer(),
               numPoints * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Start background thread
    consolidating = true;

    if (consolidationThread.joinable()) {
        consolidationThread.join();
    }

    consolidationThread = std::thread(&BackgroundConsolidator::consolidationWorker, this);

    return true;
}

void BackgroundConsolidator::waitForCompletion() {
    if (consolidationThread.joinable()) {
        consolidationThread.join();
    }
}

void BackgroundConsolidator::consolidationWorker() {
    auto start = std::chrono::high_resolution_clock::now();

    printf("\n[CPU Consolidation] Starting background consolidation...\n");

    // Step 1: Find affected nodes (nodes with deleted neighbors)
    findAffectedNodesCPU();

    printf("[CPU Consolidation] Found %zu affected nodes\n", affectedNodes.size());

    if (affectedNodes.empty()) {
        consolidating = false;
        lastRebuiltCount = 0;
        return;
    }

    // Step 2: Rebuild edges for affected nodes in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < affectedNodes.size(); i++) {
        if (shutdownRequested) continue;
        rebuildNodeEdgesCPU(affectedNodes[i]);
    }

    // Step 3: Apply updates back to GPU
    if (!shutdownRequested) {
        applyUpdatesToGPU();

        // Clear delete list on GPU
        deleteList->clear();
    }

    auto end = std::chrono::high_resolution_clock::now();
    lastConsolidationTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    lastRebuiltCount = affectedNodes.size();

    printf("[CPU Consolidation] Complete: %u nodes rebuilt in %.2f ms\n",
           lastRebuiltCount, lastConsolidationTimeMs);

    consolidating = false;
}

void BackgroundConsolidator::findAffectedNodesCPU() {
    affectedNodes.clear();

    for (unsigned nodeId = 0; nodeId < numPoints; nodeId++) {
        // Skip deleted nodes
        if (h_deleted[nodeId] != 0) continue;

        // Check if any neighbor is deleted
        uint8_t* entry = h_graph + nodeId * graphEntrySize;
        unsigned* degree = (unsigned*)(entry + dimensions * sizeof(float));
        unsigned* neighbors = degree + 1;

        bool hasDeletedNeighbor = false;
        for (unsigned i = 0; i < *degree && i < maxDegree; i++) {
            if (h_deleted[neighbors[i]] != 0) {
                hasDeletedNeighbor = true;
                break;
            }
        }

        if (hasDeletedNeighbor) {
            affectedNodes.push_back(nodeId);
        }
    }
}

float BackgroundConsolidator::computeDistanceCPU(unsigned id1, unsigned id2) {
    float* vec1 = (float*)(h_graph + id1 * graphEntrySize);
    float* vec2 = (float*)(h_graph + id2 * graphEntrySize);
    return computeL2DistanceCPU(vec1, vec2, dimensions);
}

void BackgroundConsolidator::rebuildNodeEdgesCPU(unsigned nodeId) {
    // Get node's vector
    uint8_t* entry = h_graph + nodeId * graphEntrySize;
    float* nodeVec = (float*)entry;

    // Build candidate set: current neighbors + their neighbors (2-hop)
    std::vector<unsigned> candidates;
    std::vector<bool> seen(numPoints, false);
    seen[nodeId] = true;

    // Add current neighbors (that aren't deleted)
    unsigned* degree = (unsigned*)(entry + dimensions * sizeof(float));
    unsigned* neighbors = degree + 1;

    for (unsigned i = 0; i < *degree && i < maxDegree; i++) {
        unsigned neighbor = neighbors[i];
        if (h_deleted[neighbor] == 0 && !seen[neighbor]) {
            candidates.push_back(neighbor);
            seen[neighbor] = true;
        }
    }

    // Add 2-hop neighbors
    size_t initialSize = candidates.size();
    for (size_t i = 0; i < initialSize; i++) {
        unsigned neighbor = candidates[i];
        uint8_t* neighborEntry = h_graph + neighbor * graphEntrySize;
        unsigned* neighborDegree = (unsigned*)(neighborEntry + dimensions * sizeof(float));
        unsigned* neighborNeighbors = neighborDegree + 1;

        for (unsigned j = 0; j < *neighborDegree && j < maxDegree; j++) {
            unsigned twoHop = neighborNeighbors[j];
            if (h_deleted[twoHop] == 0 && !seen[twoHop]) {
                candidates.push_back(twoHop);
                seen[twoHop] = true;

                // Limit candidates to avoid explosion
                if (candidates.size() >= maxDegree * 4) break;
            }
        }
        if (candidates.size() >= maxDegree * 4) break;
    }

    // If not enough candidates, do a small brute-force search
    if (candidates.size() < maxDegree) {
        // Find nearest non-deleted points
        std::vector<std::pair<float, unsigned>> distances;
        for (unsigned i = 0; i < numPoints; i++) {
            if (i != nodeId && h_deleted[i] == 0 && !seen[i]) {
                float dist = computeDistanceCPU(nodeId, i);
                distances.push_back({dist, i});
            }
        }

        // Partial sort to get top candidates
        size_t needed = maxDegree * 2 - candidates.size();
        if (needed > distances.size()) needed = distances.size();

        std::partial_sort(distances.begin(), distances.begin() + needed, distances.end());

        for (size_t i = 0; i < needed; i++) {
            candidates.push_back(distances[i].second);
        }
    }

    // Run RobustPrune on CPU
    std::vector<unsigned> newNeighbors;
    robustPruneCPU(nodeId, candidates, newNeighbors);

    // Update local copy (will be applied to GPU later)
    *degree = newNeighbors.size();
    for (size_t i = 0; i < newNeighbors.size(); i++) {
        neighbors[i] = newNeighbors[i];
    }
}

void BackgroundConsolidator::robustPruneCPU(unsigned nodeId,
                                            std::vector<unsigned>& candidates,
                                            std::vector<unsigned>& outNeighbors) {
    outNeighbors.clear();

    if (candidates.empty()) return;

    // Compute distances from node to all candidates
    std::vector<std::pair<float, unsigned>> distPairs;
    for (unsigned cand : candidates) {
        float dist = computeDistanceCPU(nodeId, cand);
        distPairs.push_back({dist, cand});
    }

    // Sort by distance
    std::sort(distPairs.begin(), distPairs.end());

    // RobustPrune: select maxDegree neighbors maintaining α-RNG property
    for (auto& [dist, cand] : distPairs) {
        if (outNeighbors.size() >= maxDegree) break;

        bool prune = false;

        // Check α-RNG condition against already selected neighbors
        for (unsigned selected : outNeighbors) {
            float distCandSelected = computeDistanceCPU(cand, selected);

            // Prune if α * dist(cand, selected) <= dist(node, cand)
            if (alpha * distCandSelected <= dist) {
                prune = true;
                break;
            }
        }

        if (!prune) {
            outNeighbors.push_back(cand);
        }
    }
}

void BackgroundConsolidator::applyUpdatesToGPU() {
    printf("[CPU Consolidation] Applying %zu updates to GPU...\n", affectedNodes.size());

    // For each affected node, copy its new adjacency list to GPU with version protection
    for (unsigned nodeId : affectedNodes) {
        uint8_t* cpuEntry = h_graph + nodeId * graphEntrySize;
        unsigned* cpuDegree = (unsigned*)(cpuEntry + dimensions * sizeof(float));
        unsigned* cpuNeighbors = cpuDegree + 1;

        // Prepare update data
        unsigned newDegree = *cpuDegree;
        unsigned newNeighbors[64];  // Use fixed size array (maxDegree assumed <= 64)
        for (unsigned i = 0; i < newDegree && i < maxDegree; i++) {
            newNeighbors[i] = cpuNeighbors[i];
        }

        // Copy to GPU with version protection
        // Increment version (begin write)
        unsigned one = 1;
        unsigned* versionPtr = d_versions + nodeId;

        // Use CUDA to atomically update
        // beginWrite equivalent
        cudaMemcpy(versionPtr, &one, sizeof(unsigned), cudaMemcpyHostToDevice);
        // Note: This is simplified - in production would use kernel for atomics

        // Copy new neighbors
        uint8_t* gpuEntry = d_graph + nodeId * graphEntrySize;
        unsigned* gpuDegree = (unsigned*)(gpuEntry + dimensions * sizeof(float));
        unsigned* gpuNeighbors = gpuDegree + 1;

        cudaMemcpy(gpuNeighbors, newNeighbors, newDegree * sizeof(unsigned),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpuDegree, &newDegree, sizeof(unsigned), cudaMemcpyHostToDevice);

        // endWrite equivalent - set version to even
        unsigned two = 2;
        cudaMemcpy(versionPtr, &two, sizeof(unsigned), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
}
