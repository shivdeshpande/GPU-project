#include "concurrent_executor.h"
#include <iostream>
#include <fstream>
#include <cstring>

/**
 * Concurrent FreshDiskANN Workload Executor
 *
 * Processes INSERT, DELETE, and QUERY operations in parallel
 * using multiple CUDA streams and worker threads.
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
        printf("  --workers N      Number of worker threads (default: 4)\n");
        printf("  --batch SIZE     Batch size for concurrent operations (default: 100)\n");
        return 1;
    }

    const char* graphFile = argv[1];
    const char* workloadFile = argv[2];

    // Determine if optional files are provided
    const char* queryFile = nullptr;
    const char* groundtruthFile = nullptr;
    int optionsStart = 3;

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
    unsigned numWorkers = 8;   // Increased for better parallelism
    unsigned batchSize = 100;

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
        } else if (strcmp(argv[i], "--workers") == 0) {
            numWorkers = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--batch") == 0) {
            batchSize = atoi(argv[i+1]);
            i++;
        }
    }

    printf("\n╔════════════════════════════════════════════════════╗\n");
    printf("║   Concurrent FreshDiskANN Workload Executor        ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");

    // Load graph
    uint8_t *h_graph, *d_graph;
    loadGraph(graphFile, &h_graph, &d_graph);

    // Load queries (optional)
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

    // Load groundtruth (optional)
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
        gtK = k;  // Use k for computed groundtruth
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
    printf("  Worker threads: %u\n", numWorkers);
    printf("  Batch size: %u\n", batchSize);

    // Initialize DeleteList and Concurrent Executor
    DeleteList deleteList(N);
    ConcurrentExecutor executor(d_graph, &deleteList, h_queries, h_groundtruth,
                                gtK, k, searchL, alpha, consolidateThresh, numWorkers);

    printf("\nStarting concurrent workload execution...\n\n");

    auto totalStart = std::chrono::high_resolution_clock::now();

    // Submit all operations to the concurrent executor
    unsigned reportInterval = 1000;
    for (size_t i = 0; i < workload.size(); i++) {
        executor.submitOperation(workload[i]);

        // Periodic consolidation check and progress report
        if ((i + 1) % batchSize == 0) {
            executor.waitForCompletion();
            executor.checkConsolidation();
        }

        // Progress report
        if ((i + 1) % reportInterval == 0) {
            const auto& stats = executor.getStats();
            printf("Progress: %zu/%zu events (%.1f%%)\n",
                   i + 1, workload.size(), (i + 1) * 100.0 / workload.size());
            printf("  Processed - I:%u D:%u Q:%u\n",
                   stats.insertsProcessed.load(),
                   stats.deletesProcessed.load(),
                   stats.queriesProcessed.load());
        }
    }

    // Wait for all remaining operations
    executor.waitForCompletion();

    // Final consolidation check
    executor.checkConsolidation();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    printf("\n╔════════════════════════════════════════════════════╗\n");
    printf("║   Concurrent Workload Complete!                    ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");

    executor.getStats().print();

    printf("\nTotal execution time: %.2f ms\n", totalTimeMs);
    printf("Throughput: %.1f ops/sec\n", workload.size() / (totalTimeMs / 1000.0));

    // Cleanup
    executor.shutdown();
    free(h_graph);
    if (h_queries) free(h_queries);
    if (h_groundtruth) free(h_groundtruth);
    cudaFree(d_graph);

    return 0;
}
