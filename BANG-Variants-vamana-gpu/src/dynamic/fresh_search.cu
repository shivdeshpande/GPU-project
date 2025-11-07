#include "../vamana.h"
#include "deleteList.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

/**
 * FreshDiskANN Interactive Search (BANG-style)
 *
 * Similar to BANG search but for FreshDiskANN with dynamic operations
 */

/**
 * Load query vectors from binary file (.bin format)
 * Format: [num_queries][dim][vector1_data...][vector2_data...]...
 */
void loadQueries(const char* filename, float** queries, unsigned* numQueries, unsigned* dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open query file: %s\n", filename);
        exit(1);
    }

    // Read number of queries and dimension
    unsigned nq, d;
    file.read((char*)&nq, sizeof(unsigned));
    file.read((char*)&d, sizeof(unsigned));

    *numQueries = nq;
    *dim = d;

    printf("Loading %u queries with dimension %u\n", *numQueries, *dim);

    // Allocate memory
    *queries = (float*)malloc(*numQueries * d * sizeof(float));

    // Read all vector data
    file.read((char*)(*queries), nq * d * sizeof(float));

    file.close();
}

/**
 * Load groundtruth from binary file (.bin format)
 * Format: [num_queries][k][id1][id2]...[idk] (all IDs for all queries)
 */
void loadGroundtruth(const char* filename, unsigned** groundtruth, unsigned* numQueries, unsigned* k) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open groundtruth file: %s\n", filename);
        exit(1);
    }

    // Read number of queries and k
    unsigned nq, kval;
    file.read((char*)&nq, sizeof(unsigned));
    file.read((char*)&kval, sizeof(unsigned));

    *numQueries = nq;
    *k = kval;

    printf("Loading groundtruth: %u queries, k=%u\n", *numQueries, *k);

    // Allocate memory
    *groundtruth = (unsigned*)malloc(*numQueries * (*k) * sizeof(unsigned));

    // Read all groundtruth IDs
    file.read((char*)(*groundtruth), nq * kval * sizeof(unsigned));

    file.close();
}

/**
 * Calculate k-recall@k
 * Computes intersection between groundtruth and results
 * groundtruth: array with stride gtK
 * results: array with stride resultsK
 * recallAt: number of results to consider for recall
 */
double calculateRecall(unsigned numQueries, unsigned* groundtruth, unsigned* results,
                       unsigned gtK, unsigned resultsK, unsigned recallAt) {
    double totalRecall = 0.0;

    for (unsigned i = 0; i < numQueries; i++) {
        std::set<unsigned> gt, res;

        // Insert groundtruth IDs (top recallAt from groundtruth)
        for (unsigned j = 0; j < recallAt && j < gtK; j++) {
            gt.insert(groundtruth[i * gtK + j]);
        }

        // Insert result IDs (top recallAt from results)
        for (unsigned j = 0; j < recallAt && j < resultsK; j++) {
            res.insert(results[i * resultsK + j]);
        }

        // Count intersection
        unsigned matched = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                matched++;
            }
        }

        totalRecall += matched;
    }

    return (totalRecall / numQueries) * (100.0 / recallAt);
}

/**
 * Run search on GPU for all queries
 * Returns results array: results[query_id * k + rank] = point_id
 */
void runSearch(uint8_t* d_graph, float* h_queries, unsigned numQueries, unsigned dim,
               unsigned searchL, unsigned k, unsigned** results,
               DeleteList* deleteList, double* timeMs) {

    CPUTimer timer;
    timer.Start();

    // Allocate memory for results
    *results = (unsigned*)malloc(numQueries * k * sizeof(unsigned));

    // Copy queries to GPU
    float* d_queries;
    gpuErrchk(cudaMalloc(&d_queries, numQueries * dim * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_queries, h_queries, numQueries * dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Allocate visited sets for all queries
    unsigned* d_visitedSets;
    unsigned* d_visitedSetCount;
    gpuErrchk(cudaMalloc(&d_visitedSets, numQueries * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, numQueries * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, numQueries * sizeof(unsigned)));

    // Run GreedySearch for all queries
    unsigned int* d_deleted = deleteList ? deleteList->getDevicePointer() : nullptr;
    greedySearch(d_graph, d_queries, d_visitedSets, d_visitedSetCount,
                 0, numQueries, searchL, d_deleted);

    // Allocate memory for distance computation and sorting
    float* d_visitedSetDists;
    unsigned* d_visitedSetAux;
    float* d_visitedSetDistsAux;

    gpuErrchk(cudaMalloc(&d_visitedSetDists, numQueries * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux, numQueries * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux, numQueries * MAX_PARENTS_PERQUERY * sizeof(float)));

    // Compute distances from queries to visited nodes
    computeDists<<<numQueries, 1024>>>(d_graph,
                                       d_visitedSets,
                                       d_visitedSetCount,
                                       d_queries,
                                       d_visitedSetDists,
                                       MAX_PARENTS_PERQUERY);
    gpuErrchk(cudaDeviceSynchronize());

    // Sort visited sets by distance
    sortByDistance<<<numQueries, MAX_PARENTS_PERQUERY,
                     MAX_PARENTS_PERQUERY*sizeof(unsigned)>>>(d_visitedSets,
                                                              d_visitedSetCount,
                                                              d_visitedSetDists,
                                                              d_visitedSetAux,
                                                              d_visitedSetDistsAux,
                                                              MAX_PARENTS_PERQUERY);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy results back to host
    unsigned* h_visitedSets = (unsigned*)malloc(numQueries * MAX_PARENTS_PERQUERY * sizeof(unsigned));
    unsigned* h_visitedSetCount = (unsigned*)malloc(numQueries * sizeof(unsigned));

    gpuErrchk(cudaMemcpy(h_visitedSets, d_visitedSets,
                         numQueries * MAX_PARENTS_PERQUERY * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_visitedSetCount, d_visitedSetCount,
                         numQueries * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

    // Free distance computation memory
    cudaFree(d_visitedSetDists);
    cudaFree(d_visitedSetAux);
    cudaFree(d_visitedSetDistsAux);

    // Extract top-k results for each query
    for (unsigned i = 0; i < numQueries; i++) {
        unsigned count = std::min(h_visitedSetCount[i], (unsigned)MAX_PARENTS_PERQUERY);
        unsigned numToCopy = std::min(count, k);

        for (unsigned j = 0; j < numToCopy; j++) {
            (*results)[i * k + j] = h_visitedSets[i * MAX_PARENTS_PERQUERY + j];
        }

        // Fill remaining with 0 if needed
        for (unsigned j = numToCopy; j < k; j++) {
            (*results)[i * k + j] = 0;
        }

    }

    timer.Stop();
    *timeMs = timer.Elapsed() * 1000.0; // Convert to ms

    // Cleanup
    free(h_visitedSets);
    free(h_visitedSetCount);
    cudaFree(d_queries);
    cudaFree(d_visitedSets);
    cudaFree(d_visitedSetCount);
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
 * Main interactive search
 */
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <graph_file> <query_file> <groundtruth_file> [k]\n", argv[0]);
        printf("\nInteractive FreshDiskANN Search\n");
        printf("  Similar to BANG search but for dynamic graphs\n");
        return 1;
    }

    const char* graphFile = argv[1];
    const char* queryFile = argv[2];
    const char* groundtruthFile = argv[3];
    unsigned k = (argc > 4) ? atoi(argv[4]) : 10;

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   FreshDiskANN Interactive Search         ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    // Load graph
    uint8_t *h_graph, *d_graph;
    loadGraph(graphFile, &h_graph, &d_graph);

    // Load queries
    float* h_queries;
    unsigned numQueries, queryDim;
    loadQueries(queryFile, &h_queries, &numQueries, &queryDim);

    if (queryDim != D) {
        fprintf(stderr, "Error: Query dimension (%u) doesn't match graph dimension (%u)\n",
                queryDim, D);
        exit(1);
    }

    // Load groundtruth
    unsigned* groundtruth;
    unsigned gtNumQueries, gtK;
    loadGroundtruth(groundtruthFile, &groundtruth, &gtNumQueries, &gtK);

    if (gtNumQueries != numQueries) {
        fprintf(stderr, "Warning: Groundtruth queries (%u) != query file queries (%u)\n",
                gtNumQueries, numQueries);
    }

    // Initialize DeleteList (empty for now)
    DeleteList deleteList(N);

    printf("\n=== Index Status ===\n");
    printf("Active points: %u\n", N);
    printf("Deleted points: %u\n", deleteList.getDeleteCount());
    printf("Queries loaded: %u\n", numQueries);
    printf("\n");

    // Interactive loop
    char input[256];
    while (true) {
        printf("Enter value of Search List Length (L) or 'q' to quit\n> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        // Trim newline
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "q") == 0 || strcmp(input, "Q") == 0) {
            break;
        }

        unsigned searchL = atoi(input);
        if (searchL == 0) {
            printf("Invalid input. Please enter a number or 'q' to quit.\n");
            continue;
        }

        printf("\nRunning with L=%u...\n\n", searchL);

        // Run search multiple times for stable timing
        const unsigned numRuns = 5;
        double totalTime = 0.0;
        double recall5 = 0.0, recall10 = 0.0;

        printf("L    Time(ms)  QPS        %u-r@%u   %u-r@%u  Deleted%%\n",
               std::min(k, 5u), std::min(k, 5u),
               std::min(k, 10u), std::min(k, 10u));
        printf("--   --------  ---        ------  -------  --------\n");

        for (unsigned run = 0; run < numRuns; run++) {
            unsigned* results;
            double timeMs;

            runSearch(d_graph, h_queries, numQueries, D, searchL, k,
                      &results, &deleteList, &timeMs);

            // Calculate recall
            double r5 = calculateRecall(numQueries, groundtruth, results, gtK, k, std::min(k, 5u));
            double r10 = calculateRecall(numQueries, groundtruth, results, gtK, k, std::min(k, 10u));

            double qps = (numQueries / timeMs) * 1000.0;
            double deletedPercent = (deleteList.getDeleteCount() / (double)N) * 100.0;

            printf("%-4u %-9.2f %-10.2f %-7.2f %-8.2f %-8.2f\n",
                   searchL, timeMs, qps, r5, r10, deletedPercent);

            totalTime += timeMs;
            recall5 += r5;
            recall10 += r10;

            free(results);
        }

        printf("\nAverage: L=%u, Time=%.2fms, QPS=%.2f, %u-recall@%u=%.2f%%, Active=%u\n",
               searchL,
               totalTime / numRuns,
               (numQueries / (totalTime / numRuns)) * 1000.0,
               std::min(k, 5u), std::min(k, 5u),
               recall5 / numRuns,
               N - deleteList.getDeleteCount());

        printf("\nTry Next run? [y|n]\n> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "n") == 0 || strcmp(input, "N") == 0) {
            break;
        }
    }

    printf("\n=== Session Summary ===\n");
    printf("Queries processed: %u\n", numQueries);
    printf("\nGoodbye!\n\n");

    // Cleanup
    free(h_graph);
    free(h_queries);
    free(groundtruth);
    cudaFree(d_graph);

    return 0;
}
