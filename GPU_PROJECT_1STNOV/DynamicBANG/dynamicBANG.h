#include <cassert>
#ifndef DYNAMICBANG_H_
#define DYNAMICBANG_H_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <sys/stat.h>
#include <vector>
#include <map>
#include <string>

// ============================================================================
// DATASET CONFIGURATION (Following BANG_Exactdistance pattern)
// ============================================================================

#define ROUND_UP(X, Y) \
	((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

using std::string;

// Default dataset: SIFT10K for testing
#ifndef DATASET_DEF
#define SIFT10K_DATASET
#endif

#ifdef SIFT10K_DATASET
typedef float datatype_t;
#define INDEX_ENTRY_LEN (772)  // 128*4 (vector) + 4 (degree) + 64*4 (neighbors)
#define D 128                   // Vector dimensions
#define L 100                   // L_search (worklist size)
#define CHUNKS 128
#define MEDOID 0                // Default starting node
#define N 10000                 // Number of base vectors
#define NUMTHREADS_COMPUTEPARENT 1
#endif

#ifdef SIFT1M_DATASET
typedef float datatype_t;
#define INDEX_ENTRY_LEN (772)
#define D 128
#define L 100
#define CHUNKS 128
#define MEDOID 123742
#define N 1000000
#define NUMTHREADS_COMPUTEPARENT 1
#endif

#ifdef SIFT100M_DATASET
typedef uint8_t datatype_t;
#define INDEX_ENTRY_LEN (388)
#define D 128
#define L 40
#define CHUNKS 64
#define MEDOID 59689614
#define N 100000000
#define NUMTHREADS_COMPUTEPARENT 1
#endif

// ============================================================================
// GRAPH & SEARCH PARAMETERS
// ============================================================================

#define R 64                     // Max node degree
#define K 100                    // Top-K neighbors to return

// ============================================================================
// FRESHDISKANN PARAMETERS
// ============================================================================

// Fresh index sizing (10% of static index capacity)
#define FRESH_INDEX_CAPACITY (N / 10)
#define FRESH_INDEX_THRESHOLD 0.08f    // Trigger consolidation at 8% full

// Consolidation triggers (hybrid strategy)
#define CONSOLIDATE_TIME_THRESHOLD 60.0f    // 60 seconds
#define CONSOLIDATE_SIZE_THRESHOLD (uint32_t)(FRESH_INDEX_CAPACITY * FRESH_INDEX_THRESHOLD)

// Batch processing
#define INSERT_BATCH_SIZE 1000
#define QUERY_BATCH_SIZE 1000
#define DELETE_BATCH_SIZE 1000

// ============================================================================
// GPU KERNEL PARAMETERS
// ============================================================================

// Bloom filter for visited tracking (same as BANG_Exactdistance)
#define BF_ENTRIES 399887U   // Prime number per query
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned);

// Thread configuration
#define THREADS_PER_BLOCK 256
#define THREADS_PER_VECTOR 8
#define WARP_SIZE 32

// Memory limits
#define MAX_PARENTS_PERQUERY (4*L+20)
#define SIZEPARENTLIST (1+1)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Static index (read-only during normal operation)
struct StaticIndex {
    uint8_t* d_pIndex;          // Device graph: [vector][degree][neighbors]
    uint8_t* h_pIndex;          // Host mirror (optional, for consolidation)
    uint32_t num_nodes;         // Current number of nodes
    uint32_t capacity;          // Maximum capacity
    size_t total_size_bytes;    // Total index size
};

// Fresh index (mutable for insertions)
struct FreshIndex {
    uint8_t* d_pIndex;          // Device graph: [vector][degree][neighbors]
    uint8_t* h_pIndex;          // Host mirror
    uint32_t* d_count;          // Atomic counter (device)
    uint32_t* h_count;          // Host mirror of count
    uint32_t capacity;          // Maximum capacity
    size_t total_size_bytes;    // Total index size
};

// Delete buffer (lazy deletion bitmap)
struct DeleteBuffer {
    uint32_t* d_bitmap;         // Device bitmap: bit_i = 1 if node i deleted
    uint32_t* h_bitmap;         // Host mirror
    uint32_t total_nodes;       // Total nodes (static + fresh)
    uint32_t num_deleted;       // Count of deleted nodes
    size_t bitmap_size_bytes;   // Total bitmap size
};

// Workload event types
enum EventType {
    EVENT_INSERT = 0,
    EVENT_DELETE = 1,
    EVENT_QUERY = 2,
    EVENT_METADATA = 3
};

// Workload event structure
struct WorkloadEvent {
    uint64_t timestamp;         // Event timestamp
    EventType type;             // Insert, delete, or query
    uint32_t id;                // Node ID (for insert/delete)
    datatype_t* vector;         // Vector data (for insert/query)
    std::string scenario;       // Scenario name
};

// Performance metrics
struct PerformanceMetrics {
    // Operation counts
    uint64_t total_inserts;
    uint64_t total_deletes;
    uint64_t total_queries;

    // Throughput
    double insert_qps;
    double delete_qps;
    double query_qps;
    double overall_throughput;

    // Latency (milliseconds)
    double insert_latency_avg;
    double insert_latency_p50;
    double insert_latency_p99;
    double delete_latency_avg;
    double delete_latency_p50;
    double delete_latency_p99;
    double query_latency_avg;
    double query_latency_p50;
    double query_latency_p99;

    // Accuracy
    double recall_at_1;
    double recall_at_10;
    double recall_at_100;

    // Index statistics
    uint32_t num_consolidations;
    double consolidation_time_total;
    double consolidation_time_avg;
    uint32_t static_index_size;
    uint32_t fresh_index_size;
    uint32_t num_deleted;

    // Memory usage
    size_t gpu_memory_used_bytes;
    size_t cpu_memory_used_bytes;

    // Timing
    double total_elapsed_time;
};

// Search result
struct SearchResult {
    uint32_t* node_ids;         // Top-K node IDs
    float* distances;           // Top-K distances
    uint32_t k;                 // Number of results
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
void initStaticIndex(StaticIndex* index, const char* index_file);
void initFreshIndex(FreshIndex* index);
void initDeleteBuffer(DeleteBuffer* buffer, uint32_t total_capacity);

// Core operations
void insertBatch(FreshIndex* fresh, StaticIndex* static_idx, DeleteBuffer* del_buf,
                 datatype_t* h_vectors, uint32_t* h_ids, uint32_t batch_size);
void deleteBatch(DeleteBuffer* del_buf, uint32_t* h_ids, uint32_t batch_size);
void searchDualIndex(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     datatype_t* h_queries, uint32_t* h_results, uint32_t num_queries,
                     uint32_t recall_at);

// Consolidation
bool shouldConsolidate(FreshIndex* fresh, DeleteBuffer* del_buf, double elapsed_time);
double consolidateIndices(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf);

// Workload processing
std::vector<WorkloadEvent> loadWorkload(const char* jsonl_file, uint32_t max_events);
void processWorkload(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     const std::vector<WorkloadEvent>& workload, PerformanceMetrics* metrics,
                     uint32_t* ground_truth, uint32_t gt_dim, uint32_t recall_at);

// Metrics
double calculate_recall(unsigned num_queries, unsigned *gold_std,
                       float *gs_dist, unsigned dim_gs,
                       unsigned *our_results, unsigned dim_or,
                       unsigned recall_at);
void printMetrics(const PerformanceMetrics* metrics);

// Utilities (from BANG_Exactdistance)
inline bool file_exists(const std::string& name) {
    struct stat buffer;
    auto val = stat(name.c_str(), &buffer);
    return (val == 0);
}

inline void alloc_aligned(void** ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align);
#endif
    assert(*ptr != nullptr);
}

// Template functions for binary file loading
template<typename T>
inline void load_bin(const std::string& bin_file, T*& data, size_t& npts, size_t& dim);

template<typename T>
inline void load_aligned_bin(const std::string& bin_file, T*& data,
                            size_t& npts, size_t& dim, size_t& rounded_dim);

// Helper functions
bool isNodeDeleted(const DeleteBuffer* buffer, uint32_t node_id);
void clearFreshIndex(FreshIndex* index);
void clearDeleteBuffer(DeleteBuffer* buffer);

// Cleanup
void freeStaticIndex(StaticIndex* index);
void freeFreshIndex(FreshIndex* index);
void freeDeleteBuffer(DeleteBuffer* buffer);

// GPU Device functions
__device__ unsigned hashFn1_d(unsigned x);
__device__ unsigned hashFn2_d(unsigned x);
__device__ void bloomInsert_d(uint32_t* bloom_filter, uint32_t value);
__device__ bool bloomContains_d(const uint32_t* bloom_filter, uint32_t value);
__device__ bool isDeleted_d(const uint32_t* d_bitmap, uint32_t node_id);

#endif // DYNAMICBANG_H_
