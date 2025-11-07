#include "../vamana.h"
#include "deleteList.h"
#include "insert.h"
#include "consolidate.h"
#include "workload.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

/**
 * FreshDiskANN Streaming Workload Executor
 *
 * Simplified version focusing on core functionality:
 * - Load initial graph
 * - Execute INSERT/DELETE/QUERY operations
 * - Trigger consolidation at threshold
 * - Track basic statistics
 */

struct Statistics {
    unsigned insertsProcessed;
    unsigned deletesProcessed;
    unsigned queriesProcessed;
    unsigned consolidationsTriggered;
    double totalInsertTime;
    double totalDeleteTime;
    double totalQueryTime;
    double totalConsolidateTime;
    double totalRecall5;
    double totalRecall10;

    Statistics() : insertsProcessed(0), deletesProcessed(0), queriesProcessed(0),
                   consolidationsTriggered(0), totalInsertTime(0.0),
                   totalDeleteTime(0.0), totalQueryTime(0.0), totalConsolidateTime(0.0),
                   totalRecall5(0.0), totalRecall10(0.0) {}

    void print() const {
        std::cout << "\n=== Workload Statistics ===\n";
        std::cout << "Inserts:        " << insertsProcessed << "\n";
        std::cout << "Deletes:        " << deletesProcessed << "\n";
        std::cout << "Queries:        " << queriesProcessed << "\n";
        std::cout << "Consolidations: " << consolidationsTriggered << "\n";
        std::cout << "\nTiming:\n";
        if (insertsProcessed > 0)
            std::cout << "  Insert avg:      " << (totalInsertTime / insertsProcessed) << " ms\n";
        if (deletesProcessed > 0)
            std::cout << "  Delete avg:      " << (totalDeleteTime / deletesProcessed) << " ms\n";
        if (queriesProcessed > 0) {
            std::cout << "  Query avg:       " << (totalQueryTime / queriesProcessed) << " ms\n";
            std::cout << "  Query QPS:       " << (queriesProcessed / (totalQueryTime / 1000.0)) << "\n";
            if (totalRecall5 > 0.0 || totalRecall10 > 0.0) {
                std::cout << "  5-recall@5:      " << (totalRecall5 / queriesProcessed) << "%\n";
                std::cout << "  10-recall@10:    " << (totalRecall10 / queriesProcessed) << "%\n";
            } else {
                std::cout << "  Recall:          N/A (no groundtruth)\n";
            }
        }
        if (consolidationsTriggered > 0)
            std::cout << "  Consolidate avg: " << (totalConsolidateTime / consolidationsTriggered) << " ms\n";
    }
};

/**
 * Load queries from binary file
 */
void loadQueries(const char* filename, float** queries, unsigned* numQueries, unsigned* dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open query file: %s\n", filename);
        exit(1);
    }

    unsigned nq, d;
    file.read((char*)&nq, sizeof(unsigned));
    file.read((char*)&d, sizeof(unsigned));

    *numQueries = nq;
    *dim = d;

    *queries = (float*)malloc(*numQueries * d * sizeof(float));
    file.read((char*)(*queries), nq * d * sizeof(float));
    file.close();

    printf("Loaded %u queries with dimension %u\n", *numQueries, *dim);
}

/**
 * Load groundtruth from binary file
 */
void loadGroundtruth(const char* filename, unsigned** groundtruth, unsigned* numQueries, unsigned* k) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open groundtruth file: %s\n", filename);
        exit(1);
    }

    unsigned nq, gtK;
    file.read((char*)&nq, sizeof(unsigned));
    file.read((char*)&gtK, sizeof(unsigned));

    *numQueries = nq;
    *k = gtK;

    *groundtruth = (unsigned*)malloc(*numQueries * gtK * sizeof(unsigned));
    file.read((char*)(*groundtruth), nq * gtK * sizeof(unsigned));
    file.close();

    printf("Loaded groundtruth: %u queries, k=%u\n", *numQueries, *k);
}

/**
 * Calculate recall for a single query
 */
double calculateRecall(unsigned* groundtruth, unsigned* results, unsigned gtK, unsigned k, unsigned recallK) {
    unsigned matches = 0;
    for (unsigned i = 0; i < recallK && i < k; i++) {
        for (unsigned j = 0; j < recallK && j < gtK; j++) {
            if (results[i] == groundtruth[j]) {
                matches++;
                break;
            }
        }
    }
    return (100.0 * matches) / recallK;
}

/**
 * GPU kernel: Compute L2 distances from query to all graph points
 */
__global__ void computeAllDistances(uint8_t* graph, float* queryVec,
                                    unsigned* deleted, float* distances, unsigned numPoints) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    // Check if point is deleted
    if (deleted != nullptr) {
        unsigned wordIdx = tid / 32;
        unsigned bitIdx = tid % 32;
        if (deleted[wordIdx] & (1u << bitIdx)) {
            distances[tid] = INFINITY;
            return;
        }
    }

    // Extract point vector from graph
    float* pointVec = (float*)(graph + tid * graphEntrySize);

    // Compute L2 distance
    float dist = 0.0f;
    for (unsigned d = 0; d < D; d++) {
        float diff = queryVec[d] - pointVec[d];
        dist += diff * diff;
    }
    distances[tid] = dist;
}

/**
 * CPU function: Find top-k nearest neighbors from distance array
 */
void findTopK(float* h_distances, unsigned* h_groundtruth, unsigned numPoints, unsigned k) {
    // Create array of (distance, id) pairs
    std::vector<std::pair<float, unsigned>> distPairs;
    distPairs.reserve(numPoints);

    for (unsigned i = 0; i < numPoints; i++) {
        if (h_distances[i] != INFINITY) {
            distPairs.push_back({h_distances[i], i});
        }
    }

    // Partial sort to get top-k
    unsigned actualK = std::min(k, (unsigned)distPairs.size());
    std::partial_sort(distPairs.begin(), distPairs.begin() + actualK, distPairs.end(),
                      [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                          return a.first < b.first;
                      });

    // Extract IDs
    for (unsigned i = 0; i < actualK; i++) {
        h_groundtruth[i] = distPairs[i].second;
    }

    // Fill remaining with 0 if needed
    for (unsigned i = actualK; i < k; i++) {
        h_groundtruth[i] = 0;
    }
}

/**
 * Compute exact groundtruth for a query via brute-force search
 */
void computeGroundtruth(uint8_t* d_graph, float* d_queryVec, DeleteList* deleteList,
                        unsigned* h_groundtruth, unsigned k) {
    // Allocate distance array on GPU
    float* d_distances;
    gpuErrchk(cudaMalloc(&d_distances, N * sizeof(float)));

    // Get delete bitvector pointer (if exists)
    unsigned* d_deleted = deleteList ? deleteList->getDevicePointer() : nullptr;

    // Compute distances to all points
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeAllDistances<<<numBlocks, threadsPerBlock>>>(d_graph, d_queryVec, d_deleted, d_distances, N);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy distances to host
    float* h_distances = (float*)malloc(N * sizeof(float));
    gpuErrchk(cudaMemcpy(h_distances, d_distances, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Find top-k on CPU
    findTopK(h_distances, h_groundtruth, N, k);

    // Cleanup
    free(h_distances);
    cudaFree(d_distances);
}

/**
 * Load graph from file
 */
void loadGraph(const char* filename, uint8_t** h_graph, uint8_t** d_graph) {
    *h_graph = (uint8_t*)malloc(N * graphEntrySize * sizeof(uint8_t));
    if (!*h_graph) {
        fprintf(stderr, "Error: Could not allocate host graph memory\n");
        exit(1);
    }

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open graph file: %s\n", filename);
        exit(1);
    }

    size_t read = fread(*h_graph, graphEntrySize, N, file);
    fclose(file);

    if (read != N) {
        fprintf(stderr, "Warning: Read %zu entries, expected %u\n", read, N);
    }

    // Copy to GPU
    gpuErrchk(cudaMalloc(d_graph, N * graphEntrySize * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(*d_graph, *h_graph, N * graphEntrySize * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    printf("Graph loaded: %u points, %u dimensions\n", N, D);
}

/**
 * Process INSERT event
 */
void processInsert(uint8_t* d_graph, const WorkloadEvent& event, float alpha, Statistics& stats) {
    CPUTimer timer;
    timer.Start();

    // Prepare vector data
    float* h_vector = (float*)malloc(D * sizeof(float));
    for (unsigned i = 0; i < D && i < event.vector.size(); i++) {
        h_vector[i] = event.vector[i];
    }
    // Fill remaining with zeros if vector is shorter
    for (unsigned i = event.vector.size(); i < D; i++) {
        h_vector[i] = 0.0f;
    }

    // Copy to GPU
    float* d_vector;
    gpuErrchk(cudaMalloc(&d_vector, D * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_vector, h_vector, D * sizeof(float), cudaMemcpyHostToDevice));

    // Insert point
    insertPoint(d_graph, d_vector, event.pointId, alpha);

    timer.Stop();
    stats.totalInsertTime += timer.Elapsed() * 1000.0; // Convert to ms
    stats.insertsProcessed++;

    free(h_vector);
    cudaFree(d_vector);
}

/**
 * Process DELETE event
 */
void processDelete(DeleteList* deleteList, const WorkloadEvent& event, Statistics& stats) {
    CPUTimer timer;
    timer.Start();

    deleteList->markDeleted(event.pointId);

    timer.Stop();
    stats.totalDeleteTime += timer.Elapsed() * 1000.0;
    stats.deletesProcessed++;
}

/**
 * Process QUERY event
 */
void processQuery(uint8_t* d_graph, const WorkloadEvent& event,
                  float* h_queries, unsigned* h_groundtruth,
                  unsigned gtK, unsigned k, unsigned searchL,
                  DeleteList* deleteList, Statistics& stats) {
    CPUTimer timer;
    timer.Start();

    // Check if query has embedded vector or uses query_id
    float* h_queryVec;
    float* h_embeddedVec = nullptr;
    unsigned* gtForQuery = nullptr;
    bool hasGroundtruth = false;

    if (!event.vector.empty()) {
        // Query has embedded vector (from workload generator)
        h_embeddedVec = (float*)malloc(D * sizeof(float));
        for (unsigned i = 0; i < D && i < event.vector.size(); i++) {
            h_embeddedVec[i] = event.vector[i];
        }
        // Fill remaining with zeros if needed
        for (unsigned i = event.vector.size(); i < D; i++) {
            h_embeddedVec[i] = 0.0f;
        }
        h_queryVec = h_embeddedVec;
        hasGroundtruth = false;  // Will compute groundtruth below
    } else {
        // Query uses query_id (references loaded queries)
        unsigned queryId = event.queryId;
        h_queryVec = h_queries + queryId * D;
        gtForQuery = h_groundtruth + queryId * gtK;
        hasGroundtruth = true;
    }

    // Copy query to GPU
    float* d_queryVec;
    gpuErrchk(cudaMalloc(&d_queryVec, D * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_queryVec, h_queryVec, D * sizeof(float), cudaMemcpyHostToDevice));

    // Compute groundtruth for embedded vectors
    unsigned* computedGroundtruth = nullptr;
    unsigned effectiveGtK = gtK;  // Use loaded gtK by default
    if (!hasGroundtruth && h_embeddedVec != nullptr) {
        computedGroundtruth = (unsigned*)malloc(k * sizeof(unsigned));
        computeGroundtruth(d_graph, d_queryVec, deleteList, computedGroundtruth, k);
        gtForQuery = computedGroundtruth;
        effectiveGtK = k;  // Computed groundtruth has k entries
        hasGroundtruth = true;  // Now we have groundtruth
    }

    // Allocate visited set
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    gpuErrchk(cudaMalloc(&d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, sizeof(unsigned)));

    // Run GreedySearch with delete filtering
    unsigned int* d_deleted = deleteList->getDevicePointer();
    greedySearch(d_graph, d_queryVec, d_visitedSet, d_visitedSetCount,
                 0, 1, searchL, d_deleted);

    // Allocate for distance computation and sorting
    float* d_visitedSetDists;
    unsigned* d_visitedSetAux;
    float* d_visitedSetDistsAux;
    gpuErrchk(cudaMalloc(&d_visitedSetDists, MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux, MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux, MAX_PARENTS_PERQUERY * sizeof(float)));

    // Compute distances
    computeDists<<<1, MAX_PARENTS_PERQUERY>>>(d_graph, d_visitedSet, d_visitedSetCount,
                                               d_queryVec, d_visitedSetDists, MAX_PARENTS_PERQUERY);

    // Sort by distance
    sortByDistance<<<1, MAX_PARENTS_PERQUERY, MAX_PARENTS_PERQUERY * sizeof(unsigned)>>>(
        d_visitedSet, d_visitedSetCount, d_visitedSetDists,
        d_visitedSetAux, d_visitedSetDistsAux, MAX_PARENTS_PERQUERY);

    // Copy results back
    unsigned* h_results = (unsigned*)malloc(k * sizeof(unsigned));
    gpuErrchk(cudaMemcpy(h_results, d_visitedSet, k * sizeof(unsigned), cudaMemcpyDeviceToHost));

    // Calculate recall if groundtruth is available
    double recall5 = 0.0, recall10 = 0.0;
    if (hasGroundtruth && gtForQuery != nullptr) {
        recall5 = calculateRecall(gtForQuery, h_results, effectiveGtK, k, std::min(k, 5u));
        recall10 = calculateRecall(gtForQuery, h_results, effectiveGtK, k, std::min(k, 10u));
        stats.totalRecall5 += recall5;
        stats.totalRecall10 += recall10;
    }

    timer.Stop();
    stats.totalQueryTime += timer.Elapsed() * 1000.0;
    stats.queriesProcessed++;

    // Cleanup
    free(h_results);
    if (h_embeddedVec) free(h_embeddedVec);
    if (computedGroundtruth) free(computedGroundtruth);
    cudaFree(d_queryVec);
    cudaFree(d_visitedSet);
    cudaFree(d_visitedSetCount);
    cudaFree(d_visitedSetDists);
    cudaFree(d_visitedSetAux);
    cudaFree(d_visitedSetDistsAux);
}

/**
 * Main execution
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <graph_file> <workload_file> [query_file] [groundtruth_file] [OPTIONS]\n", argv[0]);
        printf("\nArguments:\n");
        printf("  graph_file       Path to built VAMANA graph\n");
        printf("  workload_file    Path to JSONL workload file\n");
        printf("  query_file       (Optional) Query vectors for query_id-based queries\n");
        printf("  groundtruth_file (Optional) Groundtruth for query_id-based queries\n");
        printf("\nOptions:\n");
        printf("  --alpha ALPHA    Alpha parameter for RobustPrune (default: 1.2)\n");
        printf("  --thresh PERCENT Consolidation threshold percent (default: 5.0)\n");
        printf("  --searchL L      Search list length (default: 100)\n");
        printf("  --k K            Number of results to return (default: 10)\n");
        printf("\nNote: For workload generator queries with embedded vectors, groundtruth\n");
        printf("      is computed automatically via brute-force search.\n");
        return 1;
    }

    const char* graphFile = argv[1];
    const char* workloadFile = argv[2];

    // Determine if optional files are provided by checking if they don't start with "--"
    const char* queryFile = nullptr;
    const char* groundtruthFile = nullptr;
    int optionsStart = 3;  // Where options begin

    if (argc > 3 && argv[3][0] != '-') {
        queryFile = argv[3];
        optionsStart = 4;

        if (argc > 4 && argv[4][0] != '-') {
            groundtruthFile = argv[4];
            optionsStart = 5;
        }
    }

    // Parse options
    float alpha = 1.2f;
    float consolidateThresh = 5.0f;
    unsigned searchL = 100;
    unsigned k = 10;

    for (int i = optionsStart; i < argc - 1; i++) {
        if (strcmp(argv[i], "--alpha") == 0) {
            alpha = atof(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--thresh") == 0) {
            consolidateThresh = atof(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--searchL") == 0) {
            searchL = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--k") == 0) {
            k = atoi(argv[i+1]);
            i++;
        }
    }

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   FreshDiskANN Streaming Workload         ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    // Load graph
    uint8_t *h_graph, *d_graph;
    loadGraph(graphFile, &h_graph, &d_graph);

    // Load queries (optional - only needed for query_id-based queries)
    float* h_queries = nullptr;
    unsigned numQueries = 0, queryDim = 0;
    if (queryFile != nullptr) {
        loadQueries(queryFile, &h_queries, &numQueries, &queryDim);

        if (queryDim != D) {
            fprintf(stderr, "Error: Query dimension (%u) doesn't match graph dimension (%u)\n",
                    queryDim, D);
            exit(1);
        }
    } else {
        printf("No query file provided - using embedded query vectors from workload\n");
    }

    // Load groundtruth (optional - only needed for query_id-based queries)
    unsigned* h_groundtruth = nullptr;
    unsigned gtNumQueries = 0, gtK = 0;
    if (groundtruthFile != nullptr) {
        loadGroundtruth(groundtruthFile, &h_groundtruth, &gtNumQueries, &gtK);

        if (numQueries > 0 && gtNumQueries != numQueries) {
            fprintf(stderr, "Warning: Groundtruth queries (%u) != query file queries (%u)\n",
                    gtNumQueries, numQueries);
        }
    } else {
        printf("No groundtruth file provided - will compute groundtruth for embedded queries\n");
    }

    // Load workload
    printf("Loading workload: %s\n", workloadFile);
    Workload workload(workloadFile);
    workload.printSummary();

    printf("\nConfiguration:\n");
    printf("  Alpha (α): %.2f\n", alpha);
    printf("  Consolidation threshold: %.1f%%\n", consolidateThresh);
    printf("  Search list length (L): %u\n", searchL);
    printf("  Results per query (k): %u\n", k);

    // Initialize DeleteList
    DeleteList deleteList(N);
    Statistics stats;

    printf("\nStarting workload execution...\n\n");

    // Process events
    unsigned reportInterval = 1000;
    for (size_t i = 0; i < workload.size(); i++) {
        const WorkloadEvent& event = workload[i];

        switch (event.type) {
            case EVENT_INSERT:
                processInsert(d_graph, event, alpha, stats);
                break;

            case EVENT_DELETE:
                processDelete(&deleteList, event, stats);
                break;

            case EVENT_QUERY:
                processQuery(d_graph, event, h_queries, h_groundtruth,
                            gtK, k, searchL, &deleteList, stats);
                break;
        }

        // Check if consolidation needed
        if (shouldConsolidate(&deleteList, N, consolidateThresh)) {
            printf("\n⚠️  CONSOLIDATION TRIGGERED (Deleted: %u/%u = %.2f%%)\n",
                   deleteList.getDeleteCount(), N,
                   (float)deleteList.getDeleteCount() / N * 100.0f);

            CPUTimer timer;
            timer.Start();
            unsigned affectedNodes = consolidateDeletes(d_graph, &deleteList, alpha, true);
            timer.Stop();

            stats.totalConsolidateTime += timer.Elapsed() * 1000.0;
            stats.consolidationsTriggered++;

            printf("Consolidation time: %.2f ms\n\n", timer.Elapsed() * 1000.0);
        }

        // Progress report
        if ((i + 1) % reportInterval == 0 || i == workload.size() - 1) {
            printf("\n╭─────────────────────────────────────────────────╮\n");
            printf("│ Progress: %zu/%zu events (%.1f%%)                \n",
                   i + 1, workload.size(), (i + 1) * 100.0 / workload.size());
            printf("├─────────────────────────────────────────────────┤\n");
            printf("│ Operations:                                     │\n");
            printf("│   Inserts: %-6u  Deletes: %-6u  Queries: %-6u│\n",
                   stats.insertsProcessed, stats.deletesProcessed, stats.queriesProcessed);
            printf("│                                                 │\n");

            if (stats.queriesProcessed > 0) {
                double avgQueryTime = stats.totalQueryTime / stats.queriesProcessed;
                double qps = stats.queriesProcessed / (stats.totalQueryTime / 1000.0);

                printf("│ Query Performance:                              │\n");
                printf("│   Avg latency: %6.2f ms    QPS: %8.1f      │\n", avgQueryTime, qps);

                if (stats.totalRecall5 > 0.0 || stats.totalRecall10 > 0.0) {
                    double avgRecall5 = stats.totalRecall5 / stats.queriesProcessed;
                    double avgRecall10 = stats.totalRecall10 / stats.queriesProcessed;
                    printf("│   5-recall@5:  %6.2f%%     10-recall@10: %6.2f%%│\n", avgRecall5, avgRecall10);
                } else {
                    printf("│   Recall: N/A (no groundtruth)                 │\n");
                }
                printf("│                                                 │\n");
            }

            printf("│ Index State:                                    │\n");
            printf("│   Deleted (pending): %u (%.2f%% of index)       │\n",
                   deleteList.getDeleteCount(),
                   (float)deleteList.getDeleteCount() / N * 100.0f);
            printf("│   Consolidations: %u                            │\n", stats.consolidationsTriggered);
            printf("╰─────────────────────────────────────────────────╯\n\n");
        }
    }

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   Workload Complete!                      ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    stats.print();

    // Cleanup
    free(h_graph);
    free(h_queries);
    free(h_groundtruth);
    cudaFree(d_graph);

    return 0;
}
