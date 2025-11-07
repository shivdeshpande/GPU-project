#include <time.h>
#include <stdlib.h>

#ifndef VAMANA_H
#include "vamana.h"
#endif

void generateRandomGraph(uint8_t *graph, unsigned batchStart, unsigned batchSize) {
    srand(0);
    for (int i = batchStart; i < batchSize; i++) {
        unsigned* neighbors = (unsigned*)(graph + (i * graphEntrySize + D * sizeof(float))) + 1;
        for (int j = 0; j < R; j++) {
            neighbors[j] = (rand() % batchSize) + batchStart;
        }
    }
}

__global__ void computeDists(uint8_t *d_graph,
                             unsigned *d_nodes,
                             unsigned *d_nodeCount,
                             float *d_queryVecs,
                             float *d_dists,
                             unsigned rowSize) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    float *queryVec = d_queryVecs + D*queryID;          // Pointer to query vector
    unsigned offset = rowSize * queryID;
    unsigned numNodes = d_nodeCount[queryID];

    // Initialize distances to zero
    for (unsigned i = tid; i < numNodes; i += blockDim.x) {
        d_dists[offset + i] = 0;
    }

    __syncthreads();

    // if (queryID == 0 & tid == 0) printf("NumNodes: %d\n", numNodes);

    // Assign 8 threads to each node
    for (unsigned j = tid/8; j < numNodes; j += (blockDim.x + 7) / 8) {
        unsigned node = d_nodes[offset + j];
        float *nodeVec = (float*)(d_graph + graphEntrySize*node); // Pointer to node vector
        float sum = 0;

        // Sum up 8 dimensions in parallel
        for (unsigned i = tid%8; i < D; i += 8) {
            float diff = nodeVec[i] - queryVec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }

    /*
    for (unsigned j = tid; j < numNodes; j += blockDim.x) {
        unsigned node = d_nodes[offset + j];
        float *nodeVec = (float*)(d_graph + graphEntrySize*node); // Pointer to node vector
        float sum = 0;
        for (int i = 0; i < D; i++) {
            float diff = nodeVec[i] - queryVec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }
    */    
}

__device__ unsigned lowerBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target > arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ unsigned upperBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target >= arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void sortByDistance(unsigned *d_items,
                               unsigned *d_itemCount,
                               float *d_dists,
                               unsigned *d_itemsAux,
                               float *d_distsAux,
                               unsigned rowSize) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned numItems = d_itemCount[queryID];
    unsigned offset = queryID * rowSize;

    extern __shared__ unsigned sortedPositions[];

    for (unsigned subarraySize = 2; subarraySize < 2 * numItems; subarraySize *= 2) {
        unsigned subarrayID = tid / subarraySize;
        unsigned start = subarrayID * subarraySize;
        unsigned mid = min(start + subarraySize / 2, numItems);
        unsigned end = min(start + subarraySize, numItems);        

        unsigned before;

        if (tid >= start && tid < mid) {
            // If current thread corresponds to lower half, find the no. of elements before this element from the upper half
            before = lowerBound(&d_dists[offset + mid], 0, end-mid, d_dists[offset + tid]);
            sortedPositions[tid] = tid + before;
        } else if (tid >= mid && tid < end) {
            // If current thread corresponds to upper half, find the no. of elements before this element from the lower half
            before = upperBound(&d_dists[offset + start], 0, mid-start, d_dists[offset + tid]);
            sortedPositions[tid] = before + (tid - mid + start); 
        }
       
        __syncthreads();
        __threadfence_block();

        // Copy the neigbors to correct positions in auxiliary array
        for (int i = tid; i < numItems; i += blockDim.x) {
            d_itemsAux[offset + sortedPositions[i]] = d_items[offset + i];
            d_distsAux[offset + sortedPositions[i]] = d_dists[offset + i];
        }

        __syncthreads();
        __threadfence_block();

        // Copy from auxiliary array back into original array
        for (int i = tid; i < numItems; i += blockDim.x) {
            d_items[offset + i] = d_itemsAux[offset + i];
            d_dists[offset + i] = d_distsAux[offset + i];
        }

        __syncthreads();
        __threadfence_block();   
    }
}
