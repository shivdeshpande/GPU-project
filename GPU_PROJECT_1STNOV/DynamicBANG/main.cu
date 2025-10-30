#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <string.h>
#include <assert.h>
#include "utils/timer.h"
#include "utils/utils.h"
#include "dynamicBANG.h"
#include "file_loaders.h"
#include <cassert>
using namespace std;

// Forward declarations of kernel functions (defined in dynamicBANG.cu)
__global__ void neighbor_filtering_dual(unsigned* d_neighbors,
                                        unsigned* d_neighbors_temp,
                                        unsigned* d_numNeighbors_query,
                                        unsigned* d_numNeighbors_query_temp,
                                        bool* d_processed_bit_vec,
                                        unsigned* d_parents,
                                        uint8_t* d_pIndex_static,
                                        uint8_t* d_pIndex_fresh,
                                        uint32_t* d_fresh_count,
                                        uint32_t* d_delete_bitmap,
                                        uint32_t static_size,
                                        unsigned iter,
                                        bool* d_nextIter);

__global__ void compute_neighborDist_par_dual(unsigned* d_neighbors,
                                              unsigned* d_numNeighbors_query,
                                              float*  d_neighborsDist_query,
                                              datatype_t* d_queriesFP,
                                              uint8_t* d_pIndex_static,
                                              uint8_t* d_pIndex_fresh,
                                              uint32_t static_size);

__global__ void compute_BestLSets_par_sort_msort_new(unsigned* d_neighbors,
                                                     unsigned* d_numNeighbors_query,
                                                     float* d_neighborsDist_query,
                                                     unsigned* d_BestLSets,
                                                     float* d_BestLSetsDist,
                                                     bool* d_BestLSets_visited,
                                                     unsigned* d_parents,
                                                     unsigned iter,
                                                     bool* d_nextIter,
                                                     unsigned* d_BestLSets_count,
                                                     unsigned* d_L2ParentIds,
                                                     unsigned* d_FPSetCoordsList_Counts,
                                                     unsigned* d_numQueries);

__global__ void compute_NearestNeighbours(unsigned* d_BestLSets,
                                         unsigned* d_nearestNeighbours,
                                         unsigned* d_numQueries,
                                         unsigned* d_recall);

// Forward declarations of utility functions
void printIndexStats(const StaticIndex* static_idx, const FreshIndex* fresh_idx,
                     const DeleteBuffer* del_buf);
void saveMetricsToFile(const PerformanceMetrics* metrics, const char* filename);
void freeWorkload(std::vector<WorkloadEvent>& events);

// ============================================================================
// DUAL-INDEX SEARCH IMPLEMENTATION
// ============================================================================

void searchDualIndex(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     datatype_t* h_queries, uint32_t* h_results,
                     uint32_t num_queries, uint32_t recall_at) {

    // Allocate device memory for search
    datatype_t* d_queriesFP;
    unsigned* d_neighbors;
    unsigned* d_numNeighbors_query;
    float* d_neighborsDist_query;
    unsigned* d_BestLSets;
    float* d_BestLSetsDist;
    bool* d_BestLSets_visited;
    unsigned* d_parents;
    bool* d_nextIter;
    unsigned* d_BestLSets_count;
    bool* d_processed_bit_vec;
    unsigned* d_nearestNeighbours;
    unsigned* d_numQueries;
    unsigned* d_recall;
    unsigned* d_L2ParentIds;
    unsigned* d_FPSetCoordsList_Counts;

    // Allocations
    gpuErrchk(cudaMalloc(&d_queriesFP, sizeof(datatype_t) * (num_queries*D)));
    gpuErrchk(cudaMalloc(&d_neighbors, sizeof(unsigned) * (num_queries*(R+1))));
    gpuErrchk(cudaMalloc(&d_numNeighbors_query, sizeof(unsigned) * num_queries));
    gpuErrchk(cudaMalloc(&d_neighborsDist_query, sizeof(float) * (num_queries*(R+1))));
    gpuErrchk(cudaMalloc(&d_BestLSets, sizeof(unsigned) * (num_queries*L)));
    gpuErrchk(cudaMalloc(&d_BestLSetsDist, sizeof(float) * (num_queries*L)));
    gpuErrchk(cudaMalloc(&d_BestLSets_visited, sizeof(bool) * (num_queries*L)));
    gpuErrchk(cudaMalloc(&d_parents, sizeof(unsigned) * (num_queries*SIZEPARENTLIST)));
    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_BestLSets_count, sizeof(unsigned) * num_queries));
    gpuErrchk(cudaMalloc(&d_processed_bit_vec, sizeof(bool)*BF_MEMORY*num_queries));
    gpuErrchk(cudaMalloc(&d_nearestNeighbours, (recall_at * num_queries) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_numQueries, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_recall, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_L2ParentIds, (MAX_PARENTS_PERQUERY * num_queries) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_FPSetCoordsList_Counts, num_queries * sizeof(unsigned)));

    // Initialize
    gpuErrchk(cudaMemcpy(d_queriesFP, h_queries, sizeof(datatype_t) * (D*num_queries), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_processed_bit_vec, 0, sizeof(bool)*BF_MEMORY*num_queries));
    gpuErrchk(cudaMemset(d_parents, 1, sizeof(unsigned)*(num_queries*SIZEPARENTLIST)));
    gpuErrchk(cudaMemset(d_BestLSets_count, 0, sizeof(unsigned)*num_queries));
    gpuErrchk(cudaMemcpy(d_recall, &recall_at, sizeof(unsigned), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_numQueries, &num_queries, sizeof(unsigned), cudaMemcpyHostToDevice));

    // Initialize parent IDs with MEDOID
    unsigned* L2ParentIds = (unsigned*)malloc(sizeof(unsigned) * num_queries);
    unsigned* FPSetCoordsList_Counts = (unsigned*)malloc(sizeof(unsigned) * num_queries);
    for (int i = 0; i < num_queries; i++) {
        L2ParentIds[i] = MEDOID;
        FPSetCoordsList_Counts[i] = 1;
    }
    gpuErrchk(cudaMemcpy(d_L2ParentIds, L2ParentIds, sizeof(unsigned) * num_queries, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_FPSetCoordsList_Counts, FPSetCoordsList_Counts, sizeof(unsigned) * num_queries, cudaMemcpyHostToDevice));

    // Search iterations
    unsigned iter = 1;
    bool nextIter = false;
    unsigned numThreads_K2 = 512;  // For distance computation
    unsigned numThreads_K3 = max(R+1, 2*L);  // For merge
    unsigned numThreads_K5 = 256;  // For neighbor filtering

    do {
        gpuErrchk(cudaMemset(d_numNeighbors_query, 0, sizeof(unsigned)*num_queries));

        // Filter neighbors from both static and fresh indices
        neighbor_filtering_dual<<<num_queries, numThreads_K5>>>(
            d_neighbors,
            nullptr,  // d_neighbors_temp (unused in this version)
            d_numNeighbors_query,
            nullptr,  // d_numNeighbors_query_temp
            d_processed_bit_vec,
            d_parents,
            static_idx->d_pIndex,
            fresh->d_pIndex,
            fresh->d_count,
            del_buf->d_bitmap,
            static_idx->num_nodes,
            iter,
            d_nextIter);

        gpuErrchk(cudaDeviceSynchronize());

        // Compute distances
        compute_neighborDist_par_dual<<<num_queries, numThreads_K2>>>(
            d_neighbors,
            d_numNeighbors_query,
            d_neighborsDist_query,
            d_queriesFP,
            static_idx->d_pIndex,
            fresh->d_pIndex,
            static_idx->num_nodes);

        gpuErrchk(cudaDeviceSynchronize());

        // Sort and merge into Best-L set
        compute_BestLSets_par_sort_msort_new<<<num_queries, numThreads_K3>>>(
            d_neighbors,
            d_numNeighbors_query,
            d_neighborsDist_query,
            d_BestLSets,
            d_BestLSetsDist,
            d_BestLSets_visited,
            d_parents,
            iter,
            d_nextIter,
            d_BestLSets_count,
            d_L2ParentIds,
            d_FPSetCoordsList_Counts,
            d_numQueries);

        gpuErrchk(cudaDeviceSynchronize());

        // Check if next iteration needed
        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));

        iter++;
        if (iter == MAX_PARENTS_PERQUERY-1) {
            printf("Warning: Max iterations reached\n");
            break;
        }

    } while(nextIter);

    // Select final top-K neighbors
    compute_NearestNeighbours<<<num_queries, MAX_PARENTS_PERQUERY>>>(
        d_BestLSets,
        d_nearestNeighbours,
        d_numQueries,
        d_recall);

    gpuErrchk(cudaDeviceSynchronize());

    // Copy results back (convert from column-major to row-major)
    unsigned* temp_results = (unsigned*)malloc(sizeof(unsigned) * recall_at * num_queries);
    gpuErrchk(cudaMemcpy(temp_results, d_nearestNeighbours,
                        sizeof(unsigned) * (recall_at * num_queries),
                        cudaMemcpyDeviceToHost));

    // Transpose results
    for(unsigned i = 0; i < num_queries; i++) {
        for(unsigned j = 0; j < recall_at; j++) {
            h_results[i*recall_at + j] = temp_results[num_queries*j + i];
        }
    }

    // Cleanup
    free(temp_results);
    free(L2ParentIds);
    free(FPSetCoordsList_Counts);

    cudaFree(d_queriesFP);
    cudaFree(d_neighbors);
    cudaFree(d_numNeighbors_query);
    cudaFree(d_neighborsDist_query);
    cudaFree(d_BestLSets);
    cudaFree(d_BestLSetsDist);
    cudaFree(d_BestLSets_visited);
    cudaFree(d_parents);
    cudaFree(d_nextIter);
    cudaFree(d_BestLSets_count);
    cudaFree(d_processed_bit_vec);
    cudaFree(d_nearestNeighbours);
    cudaFree(d_numQueries);
    cudaFree(d_recall);
    cudaFree(d_L2ParentIds);
    cudaFree(d_FPSetCoordsList_Counts);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

int main(int argc, char** argv) {
    if(argc < 7) {
        cerr << "Usage: " << argv[0] << " <index_file> <query_file> <ground_truth_file> "
             << "<workload_jsonl> <recall_at> <num_threads>" << endl;
        exit(1);
    }

    string index_file = string(argv[1]);
    string query_file = string(argv[2]);
    string truthset_file = string(argv[3]);
    string workload_file = string(argv[4]);
    unsigned recall_at = atoi(argv[5]);
    unsigned num_threads = atoi(argv[6]);

    printf("================================================================================\n");
    printf("                         DynamicBANG - GPU FreshDiskANN                        \n");
    printf("================================================================================\n\n");

    printf("Configuration:\n");
    printf("  Dataset:        %s\n", "SIFT10K");
    printf("  Dimensions:     %d\n", D);
    printf("  L (search):     %d\n", L);
    printf("  R (degree):     %d\n", R);
    printf("  Recall@:        %u\n", recall_at);
    printf("  Fresh Capacity: %u\n", FRESH_INDEX_CAPACITY);
    printf("\n");

    // Initialize indices
    StaticIndex static_idx;
    FreshIndex fresh_idx;
    DeleteBuffer del_buf;

    initStaticIndex(&static_idx, index_file.c_str());
    initFreshIndex(&fresh_idx);
    initDeleteBuffer(&del_buf, static_idx.capacity + fresh_idx.capacity);

    // Load workload
    std::vector<WorkloadEvent> workload = loadWorkload(workload_file.c_str(), 1000000);

    // Load ground truth (if available)
    uint32_t* gt_ids = nullptr;
    float* gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    if (file_exists(truthset_file)) {
        load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        printf("[Ground Truth] Loaded %lu queries, dimension %lu\n", gt_num, gt_dim);
    }

    // Process workload
    PerformanceMetrics metrics = {0};
    processWorkload(&static_idx, &fresh_idx, &del_buf, workload, &metrics,
                    gt_ids, gt_dim, recall_at);

    // Print results
    printIndexStats(&static_idx, &fresh_idx, &del_buf);
    printMetrics(&metrics);
    saveMetricsToFile(&metrics, "dynamicBANG_metrics.txt");

    // Cleanup
    freeWorkload(workload);
    freeStaticIndex(&static_idx);
    freeFreshIndex(&fresh_idx);
    freeDeleteBuffer(&del_buf);

    if (gt_ids) delete[] gt_ids;
    if (gt_dists) delete[] gt_dists;

    printf("DynamicBANG completed successfully!\n");

    return 0;
}
