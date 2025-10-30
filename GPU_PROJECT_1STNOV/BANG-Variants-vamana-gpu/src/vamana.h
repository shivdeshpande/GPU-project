#define VAMANA_H

#include <stdint.h>
#include <stdio.h>
#include "timer.h"

#define BF_ENTRIES 399887U   	 // per query, max entries in BF, (prime number)
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); // 4-byte mem aligned size for actual allocation

typedef enum { INIT, PRUNED, NEIGHBOR } NodeState;

// Number of vertices in graph
#define N 10000
#define NUM_QUERIES 10000

// Number of dimensions of a vector
#define D 128
// Maximum degree of a vertex
#define R 64
// Worklist size
#define L 150

// Maximum no. of parents to keep track of
#define MAX_PARENTS_PERQUERY 600
// Maximum no. of reverse index entries to keep track of
#define MAX_REVERSE_INDEX_ENTRIES 500
// Index of medioid node
#define MEDOID 5000

#define VISITED_NODES_SIZE 10000

// Size of one vector in the graph
const unsigned graphEntrySize = D*sizeof(float) + sizeof(unsigned) + R*sizeof(unsigned);
const unsigned reverseIndexEntrySize = (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned);

// Bloom Filter
__device__ unsigned bf_hashFn1(unsigned x);
__device__ unsigned bf_hashFn2(unsigned x);
__device__ bool bf_check(bool *bf, unsigned x);
__device__ void bf_set(bool *bf, unsigned x);

void generateRandomGraph(uint8_t *graph, unsigned batchStart, unsigned batchSize);

// Greedy Search
__global__ void filterNeighbors(uint8_t *d_graph,
                                bool *d_hasParent,
                                unsigned *d_parents,
                                bool *d_bloomFilters,
                                unsigned *d_neighbors,
                                unsigned *d_neighborsCount,
                                unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount);

__global__ void computeNeighborDists(uint8_t *d_graph,
                                     unsigned *d_neighbors,
                                     unsigned *d_neighborsCount,
                                     float *d_queryVecs,
                                     float *d_neighborDists);

__global__ void sortNeighborDists(unsigned *d_neighbors,
                                  unsigned *d_neighborsCount,
                                  float *d_neighborsDist,
                                  unsigned *d_neighborsAux,
                                  float *d_neighborsDistAux);

// Given a sorted list, find the no. of elements less than or equal to target
__device__ unsigned lowerBound(float arr[], unsigned lo, unsigned hi, float target);
__device__ unsigned upperBound(float arr[], unsigned lo, unsigned hi, float target);

__global__ void mergeIntoWorklist(unsigned *d_worklistCount,
                                  unsigned *d_worklist,
                                  float *d_worklistDist,
                                  unsigned *d_worklistVisited,

                                  unsigned d_neighborsCount,
                                  unsigned *d_neighbors,
                                  float *d_neighborsDist,

                                  bool *d_hasParent,
                                  unsigned *d_parents,
                                  bool *d_nextIter);


// Out Neighbor Computation
__global__ void getNeighbors(uint8_t *d_graph,
                             unsigned batchStart,
                             unsigned *d_neighbors,
                             unsigned *d_neighborsCount);

__global__ void computeVisitedSetDists(uint8_t *d_graph,
                                       unsigned *d_visitedSet,
                                       unsigned *d_visitedSetCount,
                                       float *d_queryVecs,
                                       float *d_visitedSetDists);
  
__global__ void sortVisitedSets(unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount,
                                float *d_visitedSetDists,
                                unsigned *d_visitedSetAux,
                                float *d_visitedSetDistsAux);

__global__ void mergeNeighborsIntoVisitedSet(unsigned *d_visitedSetCount,
                                             unsigned *d_visitedSet,
                                             float *d_visitedSetDists,

                                             unsigned *d_neighborsCount,
                                             unsigned *d_neighbors,
                                             float *d_neighborsDist);


__global__ void pruneOutNeighbors(uint8_t *d_graph,
                                    unsigned batchStart,
                                    unsigned *d_visitedSet,
                                    unsigned *d_visitedSetCount,
                                    float *d_visitedSetDists,
                                    NodeState *d_visitedSetStatus,

                                    float *d_queryVecs,
                                    uint8_t *d_reverseEdgeIndex,
                                    float alpha,
                                    unsigned iter,
                                    bool *d_nextIter);

void greedySearch(uint8_t *d_graph,
                  float *d_queryVecs,
                  unsigned *d_visitedSets,
                  unsigned *d_visitedSetCount,
                  unsigned batchStart,
                  unsigned batchSize);


void computeOutNeighbors(uint8_t *d_graph,
                         float *d_queryVecs,
                         unsigned *d_visitedSets,
                         unsigned *d_visitedSetCount,
                         float alpha,
                         uint8_t *d_reverseEdgeIndex,
                         unsigned batchStart,
                         unsigned batchSize);


void computeReverseEdges(uint8_t *d_graph,
                         uint8_t *d_reverseEdgeIndex,
                         float alpha);

__global__ void computeDists(uint8_t *d_graph,
                             unsigned *d_nodes,
                             unsigned *d_nodeCount,
                             float *d_queryVecs,
                             float *d_dists,
                             unsigned rowSize);


__global__ void sortByDistance(unsigned *d_items,
                               unsigned *d_itemCount,
                               float *d_dists,
                               unsigned *d_itemsAux,
                               float *d_distsAux,
                               unsigned rowSize);

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
