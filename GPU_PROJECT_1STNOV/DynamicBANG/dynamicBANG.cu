#include <cassert>
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <unordered_set>
#include <assert.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "dynamicBANG.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include <chrono>

using namespace std;

// ============================================================================
// DEVICE FUNCTIONS - Bloom Filter (From BANG_Exactdistance)
// ============================================================================

/*Hash functions used for bloom filters */
__device__ unsigned hashFn1_d(unsigned x) {
    // FNV-1a hash
    uint64_t hash = 0xcbf29ce4;
    hash = (hash ^ (x & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;
    return hash % (BF_ENTRIES);
}

__device__ unsigned hashFn2_d(unsigned x) {
    // FNV-1a hash
    uint64_t hash = 0x84222325;
    hash = (hash ^ (x & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
    return hash % (BF_ENTRIES);
}

// ============================================================================
// DEVICE FUNCTIONS - Delete Buffer
// ============================================================================

__device__ inline bool isDeleted_d(const uint32_t* d_bitmap, uint32_t node_id) {
    uint32_t word_idx = node_id / 32;
    uint32_t bit_idx = node_id % 32;
    return (d_bitmap[word_idx] & (1U << bit_idx)) != 0;
}

// ============================================================================
// DEVICE FUNCTIONS - Binary Search (From BANG_Exactdistance)
// ============================================================================

__device__ unsigned lower_bound_d(float arr[], unsigned lo, unsigned hi, float target) {
    unsigned mid;
    while(lo < hi) {
        mid = (lo + hi)/2;
        float val = arr[mid];
        if (target <= val)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

__device__ unsigned upper_bound_d(float arr[], unsigned lo, unsigned hi, float target) {
    unsigned mid;
    while(lo < hi) {
        mid = (lo + hi)/2;
        float val = arr[mid];
        if (target >= val)
            lo = mid+1;
        else
            hi = mid;
    }
    return lo;
}

// ============================================================================
// KERNELS - Neighbor Filtering with Dual Index Support
// ============================================================================

/**
 * Filters neighbors using bloom filter, checks delete buffer, supports dual index
 * Adapted from BANG_Exactdistance neighbor_filtering_new
 */
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
                                        bool* d_nextIter) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if(d_parents[queryID*(SIZEPARENTLIST)]==0)
        return;

    *d_nextIter = false;

    unsigned offset_neighbors = queryID * (R+1);
    unsigned offset_bit_vec = queryID*BF_MEMORY;
    bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;
    unsigned long long parentID;

    if(iter==1){
        // First iteration: set MEDOID bits
        parentID = MEDOID;
        if(tid==0){
            if(!(d_processed_bit_vec_start[hashFn1_d(MEDOID)])) {
                d_processed_bit_vec_start[hashFn1_d(MEDOID)] = true;

                // Check if MEDOID is deleted
                if (!isDeleted_d(d_delete_bitmap, MEDOID)) {
                    unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
                    d_neighbors[offset_neighbors + old] = MEDOID;
                }
            }
        }
    }
    else parentID = d_parents[queryID*(SIZEPARENTLIST)+1];

    // Determine if parent is in static or fresh index
    uint8_t* d_pIndex;
    uint32_t fresh_count = *d_fresh_count;

    if (parentID < static_size) {
        // Parent is in static index
        d_pIndex = d_pIndex_static;
    } else {
        // Parent is in fresh index
        d_pIndex = d_pIndex_fresh;
        parentID = parentID - static_size;  // Adjust to fresh index offset
    }

    unsigned* bound = (unsigned*)(d_pIndex + ((unsigned long long)INDEX_ENTRY_LEN*parentID) + D*sizeof(datatype_t));

    // Process each neighbor
    for(unsigned ii=tid; ii < *bound; ii += blockDim.x ) {
        unsigned nbr = *(bound+1+ii);

        // Check if already visited using bloom filter
        if(!(d_processed_bit_vec_start[hashFn1_d(nbr)])) {
            d_processed_bit_vec_start[hashFn1_d(nbr)] = true;

            // Check if deleted
            if (!isDeleted_d(d_delete_bitmap, nbr)) {
                unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
                d_neighbors[offset_neighbors + old] = nbr;
            }
        }
    }
}

// ============================================================================
// KERNELS - Distance Computation with Dual Index Support
// ============================================================================

/**
 * Computes exact L2 distance for neighbors from dual index
 * Adapted from BANG_Exactdistance compute_neighborDist_par
 */
__global__ void compute_neighborDist_par_dual(unsigned* d_neighbors,
                                              unsigned* d_numNeighbors_query,
                                              float*  d_neighborsDist_query,
                                              datatype_t* d_queriesFP,
                                              uint8_t* d_pIndex_static,
                                              uint8_t* d_pIndex_fresh,
                                              uint32_t static_size) {

    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;

    unsigned numNeighbors = d_numNeighbors_query[queryID];
    unsigned queryNeighbors_start  = queryID * (R+1);
    float* d_neighborsDist_query_start = d_neighborsDist_query + queryNeighbors_start;
    datatype_t* d_queriesFP_start = d_queriesFP+(queryID*D);

    #define THREADS_PER_NEIGHBOR 8
    typedef cub::WarpReduce<float,THREADS_PER_NEIGHBOR> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[R];

    // 8 threads cooperate to compute distance for each neighbor
    for(unsigned j = tid/THREADS_PER_NEIGHBOR; j < numNeighbors; j += (blockDim.x)/THREADS_PER_NEIGHBOR) {
        unsigned long long myNeighbor = d_neighbors[queryNeighbors_start + j];

        // Determine if neighbor is in static or fresh index
        datatype_t* pBase;
        if (myNeighbor < static_size) {
            pBase = (datatype_t*)(d_pIndex_static+(myNeighbor*INDEX_ENTRY_LEN));
        } else {
            uint32_t fresh_id = myNeighbor - static_size;
            pBase = (datatype_t*)(d_pIndex_fresh+(fresh_id*INDEX_ENTRY_LEN));
        }

        // Compute L2 distance (exact, no PQ compression)
        float sum = 0.0f;
        for(unsigned i = tid%THREADS_PER_NEIGHBOR; i < D; i += THREADS_PER_NEIGHBOR){
            float diff = (float)(*(pBase+i)) - (float)d_queriesFP_start[i];
            sum += diff*diff;
        }
        d_neighborsDist_query_start[j] = WarpReduce(temp_storage[j]).Sum(sum);
    }
}

// ============================================================================
// KERNELS - Best-L Set Computation (From BANG_Exactdistance)
// ============================================================================

/**
 * Sorts neighbors and merges with Best-L set
 * Directly from BANG_Exactdistance compute_BestLSets_par_sort_msort_new
 */
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
                                                     unsigned* d_numQueries) {

    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    __shared__ unsigned shm_pos[R+1];
    unsigned offset = queryID*(R+1);

    __shared__ float shm_neighborsDist_query_aux[R+1];
    __shared__ unsigned shm_neighbors_aux[R+1];

    __shared__ float shm_neighborsDist_query[R];
    __shared__ float shm_currBestLSetsDist[L];
    __shared__ float shm_BestLSetsDist[L];
    __shared__ unsigned shm_pos1[R+L+1];
    __shared__ unsigned shm_BestLSets[L];
    __shared__ bool shm_BestLSets_visited[L];
    __shared__ unsigned Temp;

    // Parallel merge sort
    for(unsigned subArraySize=2; subArraySize< 2*numNeighbors; subArraySize *= 2){
        unsigned subArrayID = tid/subArraySize;
        unsigned start = subArrayID * subArraySize;
        unsigned mid = min(start + subArraySize/2, numNeighbors);
        unsigned end = min(start + subArraySize, numNeighbors);

        if(tid >= start && tid < mid){
            unsigned lowerBound = lower_bound_d(&d_neighborsDist_query[offset + mid], 0, end-mid, d_neighborsDist_query[offset + tid]);
            shm_pos[tid] = lowerBound + tid;
        }

        if(tid >= mid && tid < end)  {
            unsigned upperBound = upper_bound_d(&d_neighborsDist_query[offset + start], 0, mid-start, d_neighborsDist_query[offset + tid]);
            shm_pos[tid] = start + (upperBound + tid-mid);
        }
        __syncthreads();
        __threadfence_block();

        for(int i=tid; i < numNeighbors; i += blockDim.x) {
            shm_neighborsDist_query_aux[shm_pos[i]] = d_neighborsDist_query[offset+i];
            shm_neighbors_aux[shm_pos[i]] = d_neighbors[offset+i];
        }
        __syncthreads();

        for(int i=tid; i < numNeighbors; i += blockDim.x) {
            d_neighborsDist_query[offset + i] = shm_neighborsDist_query_aux[i];
            d_neighbors[offset + i] = shm_neighbors_aux[i];
        }
        __syncthreads();
    }

    __syncthreads();
    for(int i=tid; i < numNeighbors; i += blockDim.x) {
        shm_neighborsDist_query_aux[i] = d_neighborsDist_query[offset + i];
        shm_neighbors_aux[i] = d_neighbors[offset + i];
    }
    __syncthreads();

    unsigned Best_L_Set_size;
    unsigned newBest_L_Set_size;
    __shared__ unsigned nbrsBound;

    if(numNeighbors > 0){
        if(iter==1){
            nbrsBound = min(numNeighbors,L);
            for(unsigned ii=tid; ii < nbrsBound; ii += blockDim.x) {
                unsigned nbr = shm_neighbors_aux[ii];
                d_BestLSets[queryID*L + tid] = nbr;
                d_BestLSetsDist[queryID*L + tid] = shm_neighborsDist_query_aux[ii];
                d_BestLSets_visited[queryID*L + tid] = (nbr == MEDOID);
            }
            __syncthreads();
            newBest_L_Set_size = nbrsBound;
            d_BestLSets_count[queryID] = nbrsBound;
        }
        else {
            Best_L_Set_size = d_BestLSets_count[queryID];
            float maxBestLSetDist = d_BestLSetsDist[L*queryID+Best_L_Set_size-1];
            Temp = min(L,numNeighbors);

            if (tid == 0) {
                for(nbrsBound = 0; nbrsBound < Temp; ++nbrsBound) {
                    if(shm_neighborsDist_query_aux[nbrsBound] >= maxBestLSetDist){
                        break;
                    }
                }
            }
            __syncthreads();

            nbrsBound = max(nbrsBound, min(L-Best_L_Set_size, numNeighbors));
            newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, L);
            d_BestLSets_count[queryID] = newBest_L_Set_size;

            for(int i=tid; i < Best_L_Set_size; i += blockDim.x) {
                shm_currBestLSetsDist[i] = d_BestLSetsDist[L*queryID+i];
            }
            __syncthreads();

            if(tid < nbrsBound) {
                shm_pos1[tid] = lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size, shm_neighborsDist_query_aux[tid]) + tid;
            }
            if(tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size)) {
                shm_pos1[tid] = upper_bound_d(shm_neighborsDist_query_aux, 0, nbrsBound, shm_currBestLSetsDist[tid-nbrsBound]) + (tid-nbrsBound);
            }

            __syncthreads();
            __threadfence_block();

            if(tid < nbrsBound && shm_pos1[tid] < newBest_L_Set_size)  {
                shm_BestLSetsDist[shm_pos1[tid]] = shm_neighborsDist_query_aux[tid];
                shm_BestLSets[shm_pos1[tid]] = shm_neighbors_aux[tid];
                shm_BestLSets_visited[shm_pos1[tid]] = false;
            }
            Temp = (nbrsBound + Best_L_Set_size);
            if(tid >= nbrsBound && tid < Temp && shm_pos1[tid] < newBest_L_Set_size) {
                shm_BestLSetsDist[shm_pos1[tid]] = shm_currBestLSetsDist[tid-nbrsBound];
                shm_BestLSets[shm_pos1[tid]] = d_BestLSets[queryID*L+(tid-nbrsBound)];
                shm_BestLSets_visited[shm_pos1[tid]] = d_BestLSets_visited[queryID*L+(tid-nbrsBound)];
            }
            __syncthreads();
            __threadfence_block();

            if (tid < newBest_L_Set_size) {
                d_BestLSetsDist[L*queryID+tid] = shm_BestLSetsDist[tid];
                d_BestLSets[L*queryID+tid] = shm_BestLSets[tid];
                d_BestLSets_visited[L*queryID+tid] = shm_BestLSets_visited[tid];
            }
            __syncthreads();
        }
    }

    if(tid == 0) {
        unsigned parentIndex = 0;
        for(unsigned ii=0; ii < newBest_L_Set_size; ++ii) {
            if(!d_BestLSets_visited[L*queryID + ii]) {
                parentIndex++;
                d_BestLSets_visited[L*queryID + ii] = true;
                d_parents[queryID*(SIZEPARENTLIST)] = parentIndex;
                d_parents[queryID*(SIZEPARENTLIST)+parentIndex] = d_BestLSets[L*queryID + ii];
                *d_nextIter = true;
                break;
            }
        }
    }
}

// ============================================================================
// KERNELS - Final Nearest Neighbors Selection
// ============================================================================

/**
 * Selects top-K nearest neighbors from Best-L set
 * Adapted from BANG_Exactdistance compute_NearestNeighbours
 */
__global__ void compute_NearestNeighbours(unsigned* d_BestLSets,
                                         unsigned* d_nearestNeighbours,
                                         unsigned* d_numQueries,
                                         unsigned* d_recall) {
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;

    for(unsigned ii = tid; ii < *d_recall; ii += blockDim.x) {
        d_nearestNeighbours[((*d_numQueries) * ii) + queryID] = d_BestLSets[(L * queryID) + ii];
    }
}
