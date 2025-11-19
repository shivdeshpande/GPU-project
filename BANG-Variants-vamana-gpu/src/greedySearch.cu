#ifndef VAMANA_H
#include "vamana.h"
#endif

#include "dynamic/lockfree_graph.cuh"

__device__ bool contains(unsigned *set, unsigned count, unsigned el) {
    for (int i = 0; i < count; i++) {
        if (set[i] == el) {
            return true;
        }
    }

    return false;
}

__global__ void initializeParents(bool *d_hasParent, unsigned *d_parents) {
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (tid == 0) {
        d_hasParent[queryID] = true;
        d_parents[queryID] = MEDOID;
    }
}

__global__ void initializeWorklist(uint8_t *d_graph,
                                   float *d_queryVecs,
                                   unsigned *d_worklist,
                                   unsigned *d_worklistCount,
                                   float *d_worklistDist,
                                   bool *d_worklistVisited,
                                   unsigned searchL) {
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned worklistOffset = searchL * queryID;

    float *queryVec = d_queryVecs + D * queryID;
    float *medoidVec = (float*)(d_graph + graphEntrySize * MEDOID);

    if (tid == 0) {
        d_worklist[worklistOffset] = MEDOID;
        d_worklistCount[queryID] = 1;
        d_worklistVisited[worklistOffset] = true;

        float dist = 0;
        for (int i = 0; i < D; i++) {
            float diff = queryVec[i] - medoidVec[i];
            dist += diff * diff;
        }
        d_worklistDist[worklistOffset] = dist;
    }
}

/* Adds unvisited neighbours of nodes in d_parents to d_neighbors
 *
 * d_graph           - The graph
 * d_hasParent       - Whether there is a node to be visited (a parent) for a query
 * d_parents         - If the query has a parent, then the index of the parent
 * d_bloomFilters    - The bloom filters for checking if a node has been visited
 * d_neighbors       - Array for putting unvisited neighbors into
 * d_neighborsCount  - No. of unvisited neighbors for each query
 * d_visitedSet      - Array of visited points for each query
 * d_visitedSetCount - No. of visited points for each query
 */
__global__ void filterNeighbors(uint8_t *d_graph,
                                bool *d_hasParent,
                                unsigned *d_parents,
                                bool *d_bloomFilters,
                                unsigned *d_neighbors,
                                unsigned *d_neighborsCount,
                                unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount,
                                unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY); // Get the bloom filter for this query
    unsigned parent = d_parents[queryID];                       // Get the parent for this query

    // Get the pointers to the degree and neighbors of the parent
    unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(float));
    unsigned *neighborPtr = degreePtr + 1;

    __syncthreads();

    // Initialize neighbor count to zero, reset hasParent
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }
    }

    __syncthreads();

    // Loop over each neighbor
    unsigned degree = *degreePtr;

    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = neighborPtr[ii];

        if (neighbor == queryID)
            continue;

        // FreshDiskANN: Skip deleted neighbors if deletion filtering is enabled
        if (d_deleted != nullptr && d_deleted[neighbor] != 0)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}

/**
 * Version-based filterNeighbors kernel for concurrent safety
 *
 * Uses version numbers to ensure consistent reads of neighbor lists
 * even when other threads are modifying them concurrently.
 */
__global__ void filterNeighborsVersioned(uint8_t *d_graph,
                                          unsigned *d_versions,
                                          bool *d_hasParent,
                                          unsigned *d_parents,
                                          bool *d_bloomFilters,
                                          unsigned *d_neighbors,
                                          unsigned *d_neighborsCount,
                                          unsigned *d_visitedSet,
                                          unsigned *d_visitedSetCount,
                                          unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY);
    unsigned parent = d_parents[queryID];

    // Shared memory for neighbor snapshot
    __shared__ unsigned localNeighbors[R];
    __shared__ unsigned localDegree;

    __syncthreads();

    // Thread 0 takes the snapshot with version check
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }

        // Version-based read with retry
        unsigned retries = 0;
        const unsigned maxRetries = 5;

        while (retries < maxRetries) {
            unsigned v1 = d_versions[parent];
            if (v1 & 1) {
                // Write in progress, small backoff
                retries++;
                continue;
            }
            __threadfence();

            // Read degree and neighbors
            unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(float));
            unsigned *neighborPtr = degreePtr + 1;

            localDegree = *degreePtr;
            unsigned count = min(localDegree, (unsigned)R);

            for (unsigned i = 0; i < count; i++) {
                localNeighbors[i] = neighborPtr[i];
            }

            __threadfence();
            unsigned v2 = d_versions[parent];

            if (v1 == v2) {
                break;  // Consistent read
            }
            retries++;
        }

        // If max retries hit, use last read (bounded staleness)
        if (retries >= maxRetries) {
            unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(float));
            unsigned *neighborPtr = degreePtr + 1;
            localDegree = *degreePtr;
            for (unsigned i = 0; i < min(localDegree, (unsigned)R); i++) {
                localNeighbors[i] = neighborPtr[i];
            }
        }
    }

    __syncthreads();

    // All threads process from the snapshot
    unsigned degree = localDegree;

    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = localNeighbors[ii];

        if (neighbor == queryID)
            continue;

        // Skip deleted neighbors
        if (d_deleted != nullptr && d_deleted[neighbor] != 0)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}


__global__ void mergeIntoWorklist(unsigned *d_worklistCount,
                                  unsigned *d_worklist,
                                  float *d_worklistDist,
                                  bool *d_worklistVisited,

                                  unsigned *d_neighborsCount,
                                  unsigned *d_neighbors,
                                  float *d_neighborsDist,

                                  bool *d_hasParent,
                                  unsigned *d_parents,
                                  bool *d_nextIter,
                                  unsigned searchL) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned neighborsOffset = queryID * (R + 1);
    unsigned worklistOffset = queryID * searchL;

    unsigned numNeighbors = d_neighborsCount[queryID];
    unsigned worklistSize = d_worklistCount[queryID];
    unsigned newWorklistSize = min(numNeighbors + worklistSize, searchL);

    __shared__ unsigned sortedPositions[R + MAX_L + 1];

    unsigned id;
    float dist;
    bool visited;
    unsigned newPos = searchL;

    if (tid < worklistSize) {
        // First searchL threads find new position for worklist elements
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_worklistDist[worklistOffset + tid]);
        id = d_worklist[worklistOffset + tid];
        dist = d_worklistDist[worklistOffset + tid];
        visited = d_worklistVisited[worklistOffset + tid];
        newPos = before + tid;
        sortedPositions[tid] = newPos;
    } else if (tid >= MAX_L && tid < MAX_L + numNeighbors) {
        // Next R + 1 threads find new position for neighbors
        unsigned idx = tid - MAX_L; // Index into the neighbors array
        unsigned before = upperBound(&d_worklistDist[worklistOffset], 0, worklistSize, d_neighborsDist[neighborsOffset + idx]);
        id = d_neighbors[neighborsOffset + idx];
        dist = d_neighborsDist[neighborsOffset + idx];
        visited = false;
        newPos = before + idx;
        sortedPositions[tid] = newPos;
    }
    
    __syncthreads();
    __threadfence_block();

    if (newPos < newWorklistSize) {
        d_worklist[worklistOffset + newPos] = id;
        d_worklistDist[worklistOffset + newPos] = dist;
        d_worklistVisited[worklistOffset + newPos] = visited;
    }

    __syncthreads();
    __threadfence_block();
    
    if (tid == 0) {
        d_worklistCount[queryID] = newWorklistSize;
        
        for (unsigned ii = 0; ii < newWorklistSize; ii++) {
            // Find the closest unvisited node, set it as the parent for the next iteration, and mark it as visited.
            if (!d_worklistVisited[worklistOffset + ii]) {
                *d_nextIter = true; 
                d_hasParent[queryID] = true;
                d_parents[queryID] = d_worklist[worklistOffset + ii];
                d_worklistVisited[worklistOffset + ii] = true;

                break;
            }
        }
    }   
}


// Performs greedy search and returns the visited sets
void greedySearch(uint8_t *d_graph,
                  float *d_queryVecs,
                  unsigned *d_visitedSets,
                  unsigned *d_visitedSetCount,
                  unsigned batchStart,
                  unsigned batchSize,
                  unsigned searchL,
                  unsigned int *d_deleted) {
 
    bool *d_hasParent;
    unsigned *d_parents;
    bool *d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMemset(d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(unsigned)));
    
    unsigned *d_worklist;
    unsigned *d_worklistCount;
    float *d_worklistDist;
    bool *d_worklistVisited;

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * MAX_L * sizeof(bool)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(unsigned)));    

    bool nextIter;
    bool *d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<<<batchSize, 1>>>(d_graph,
                                         d_queryVecs,
                                         d_worklist,
                                         d_worklistCount,
                                         d_worklistDist,
                                         d_worklistVisited,
                                         searchL);
    // cudaDeviceSynchronize();

    unsigned *neighbors = (unsigned*)malloc((R+1) * sizeof(unsigned));
    float *neighborsDist = (float*)malloc((R+1) * sizeof(float));
    unsigned *worklist = (unsigned*)malloc(MAX_L * sizeof(unsigned));
    
    int iter = 0;
    do {    
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));
        
        filterNeighbors<<<batchSize, R>>>(d_graph,
                                            d_hasParent,
                                            d_parents,
                                            d_bloomFilters,
                                            d_neighbors,
                                            d_neighborsCount,
                                            d_visitedSets,
                                            d_visitedSetCount,
                                            d_deleted);

        // gpuErrchk(cudaDeviceSynchronize());

        computeDists<<<batchSize, R*8>>>(d_graph,
                                           d_neighbors,
                                           d_neighborsCount,
                                           d_queryVecs,
                                           d_neighborDists,
                                           (R+1));
        // gpuErrchk(cudaDeviceSynchronize());

        sortByDistance<<<batchSize, R, R*sizeof(unsigned)>>>(d_neighbors,
                                                                     d_neighborsCount,
                                                                     d_neighborDists,
                                                                     d_neighborsAux,
                                                                     d_neighborDistsAux,
                                                                     R+1);
        // gpuErrchk(cudaDeviceSynchronize());
       
        mergeIntoWorklist<<<batchSize, R+MAX_L>>>(d_worklistCount,
                                                d_worklist,
                                                d_worklistDist,
                                                d_worklistVisited,

                                                d_neighborsCount,
                                                d_neighbors,
                                                d_neighborDists,

                                                d_hasParent,
                                                d_parents,
                                                d_nextIter,
                                                searchL);
        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    }  while (nextIter);
        
    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));
    
    gpuErrchk(cudaFree(d_nextIter));

    printf("Greedy search finished in %d iterations.\n", iter);
}

/**
 * Version-based greedy search for concurrent safety
 *
 * Uses version numbers to ensure consistent reads of neighbor lists
 * even when other threads are modifying them concurrently.
 */
void greedySearchVersioned(uint8_t *d_graph,
                           unsigned *d_versions,
                           float *d_queryVecs,
                           unsigned *d_visitedSets,
                           unsigned *d_visitedSetCount,
                           unsigned batchStart,
                           unsigned batchSize,
                           unsigned searchL,
                           unsigned int *d_deleted) {

    bool *d_hasParent;
    unsigned *d_parents;
    bool *d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMemset(d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(unsigned)));

    unsigned *d_worklist;
    unsigned *d_worklistCount;
    float *d_worklistDist;
    bool *d_worklistVisited;

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * MAX_L * sizeof(bool)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(unsigned)));

    bool nextIter;
    bool *d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<<<batchSize, 1>>>(d_graph,
                                         d_queryVecs,
                                         d_worklist,
                                         d_worklistCount,
                                         d_worklistDist,
                                         d_worklistVisited,
                                         searchL);

    int iter = 0;
    do {
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));

        // Use versioned filterNeighbors for concurrent safety
        filterNeighborsVersioned<<<batchSize, R>>>(d_graph,
                                                    d_versions,
                                                    d_hasParent,
                                                    d_parents,
                                                    d_bloomFilters,
                                                    d_neighbors,
                                                    d_neighborsCount,
                                                    d_visitedSets,
                                                    d_visitedSetCount,
                                                    d_deleted);

        computeDists<<<batchSize, R*8>>>(d_graph,
                                           d_neighbors,
                                           d_neighborsCount,
                                           d_queryVecs,
                                           d_neighborDists,
                                           (R+1));

        sortByDistance<<<batchSize, R, R*sizeof(unsigned)>>>(d_neighbors,
                                                                     d_neighborsCount,
                                                                     d_neighborDists,
                                                                     d_neighborsAux,
                                                                     d_neighborDistsAux,
                                                                     R+1);

        mergeIntoWorklist<<<batchSize, R+MAX_L>>>(d_worklistCount,
                                                d_worklist,
                                                d_worklistDist,
                                                d_worklistVisited,

                                                d_neighborsCount,
                                                d_neighbors,
                                                d_neighborDists,

                                                d_hasParent,
                                                d_parents,
                                                d_nextIter,
                                                searchL);
        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    }  while (nextIter);

    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));

    gpuErrchk(cudaFree(d_nextIter));
}

// Allocate pre-allocated buffers for greedySearchVersioned
void allocateGreedySearchBuffers(GreedySearchBuffers* buffers, unsigned batchSize) {
    buffers->batchSize = batchSize;

    gpuErrchk(cudaMalloc(&buffers->d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&buffers->d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));

    gpuErrchk(cudaMalloc(&buffers->d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));

    gpuErrchk(cudaMalloc(&buffers->d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistVisited, batchSize * MAX_L * sizeof(bool)));

    gpuErrchk(cudaMalloc(&buffers->d_nextIter, sizeof(bool)));
    gpuErrchk(cudaMallocHost(&buffers->h_nextIter, sizeof(bool)));  // Pinned memory
}

// Free pre-allocated buffers
void freeGreedySearchBuffers(GreedySearchBuffers* buffers) {
    gpuErrchk(cudaFree(buffers->d_hasParent));
    gpuErrchk(cudaFree(buffers->d_parents));
    gpuErrchk(cudaFree(buffers->d_bloomFilters));

    gpuErrchk(cudaFree(buffers->d_neighbors));
    gpuErrchk(cudaFree(buffers->d_neighborsCount));
    gpuErrchk(cudaFree(buffers->d_neighborDists));
    gpuErrchk(cudaFree(buffers->d_neighborsAux));
    gpuErrchk(cudaFree(buffers->d_neighborDistsAux));

    gpuErrchk(cudaFree(buffers->d_worklist));
    gpuErrchk(cudaFree(buffers->d_worklistCount));
    gpuErrchk(cudaFree(buffers->d_worklistDist));
    gpuErrchk(cudaFree(buffers->d_worklistVisited));

    gpuErrchk(cudaFree(buffers->d_nextIter));
    gpuErrchk(cudaFreeHost(buffers->h_nextIter));  // Free pinned memory
}

// Version using pre-allocated buffers (much faster)
void greedySearchVersionedPrealloc(uint8_t *d_graph,
                                    unsigned *d_versions,
                                    float *d_queryVecs,
                                    unsigned *d_visitedSets,
                                    unsigned *d_visitedSetCount,
                                    unsigned batchStart,
                                    unsigned batchSize,
                                    unsigned searchL,
                                    unsigned int *d_deleted,
                                    GreedySearchBuffers* buffers,
                                    cudaStream_t stream) {

    // Reset buffers for this query (async on stream)
    gpuErrchk(cudaMemsetAsync(buffers->d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighborsCount, 0, batchSize * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_worklistCount, 0, batchSize * sizeof(unsigned), stream));

    initializeParents<<<batchSize, 1, 0, stream>>>(buffers->d_hasParent, buffers->d_parents);
    initializeWorklist<<<batchSize, 1, 0, stream>>>(d_graph,
                                         d_queryVecs,
                                         buffers->d_worklist,
                                         buffers->d_worklistCount,
                                         buffers->d_worklistDist,
                                         buffers->d_worklistVisited,
                                         searchL);

    // Use pinned memory for truly async copy
    bool* h_nextIter = buffers->h_nextIter;
    *h_nextIter = true;

    int iter = 0;
    const int CHECK_INTERVAL = 10;  // Check termination every N iterations

    while (*h_nextIter && iter < searchL * 2) {  // Upper bound to prevent infinite loops
        // Run multiple iterations before checking termination
        for (int i = 0; i < CHECK_INTERVAL && iter < searchL * 2; i++) {
            iter++;
            gpuErrchk(cudaMemsetAsync(buffers->d_nextIter, false, sizeof(bool), stream));

            filterNeighborsVersioned<<<batchSize, R, 0, stream>>>(d_graph,
                                                        d_versions,
                                                        buffers->d_hasParent,
                                                        buffers->d_parents,
                                                        buffers->d_bloomFilters,
                                                        buffers->d_neighbors,
                                                        buffers->d_neighborsCount,
                                                        d_visitedSets,
                                                        d_visitedSetCount,
                                                        d_deleted);

            computeDists<<<batchSize, R*8, 0, stream>>>(d_graph,
                                               buffers->d_neighbors,
                                               buffers->d_neighborsCount,
                                               d_queryVecs,
                                               buffers->d_neighborDists,
                                               (R+1));

            sortByDistance<<<batchSize, R, R*sizeof(unsigned), stream>>>(buffers->d_neighbors,
                                                                         buffers->d_neighborsCount,
                                                                         buffers->d_neighborDists,
                                                                         buffers->d_neighborsAux,
                                                                         buffers->d_neighborDistsAux,
                                                                         R+1);

            mergeIntoWorklist<<<batchSize, R+MAX_L, 0, stream>>>(buffers->d_worklistCount,
                                                    buffers->d_worklist,
                                                    buffers->d_worklistDist,
                                                    buffers->d_worklistVisited,

                                                    buffers->d_neighborsCount,
                                                    buffers->d_neighbors,
                                                    buffers->d_neighborDists,

                                                    buffers->d_hasParent,
                                                    buffers->d_parents,
                                                    buffers->d_nextIter,
                                                    searchL);
        }

        // Check termination condition (only once per CHECK_INTERVAL iterations)
        gpuErrchk(cudaMemcpyAsync(h_nextIter, buffers->d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    // printf("Greedy search finished in %d iterations.\n", iter);
}
