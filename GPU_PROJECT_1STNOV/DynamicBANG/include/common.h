#ifndef DYNAMICBANG_COMMON_H
#define DYNAMICBANG_COMMON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// DATASET CONFIGURATION
// ============================================================================

// Default: SIFT10K (can be overridden via compile flags)
#ifndef DATASET
#define DATASET SIFT10K
#endif

// Dataset-specific configurations
#if DATASET == SIFT10K
    #define N 10000              // Number of initial base vectors
    #define D 128                // Vector dimensions
    #define INITIAL_CAPACITY 10000
#elif DATASET == SIFT1M
    #define N 1000000
    #define D 128
    #define INITIAL_CAPACITY 1000000
#elif DATASET == DEEP100M
    #define N 100000000
    #define D 96
    #define INITIAL_CAPACITY 100000000
#else
    #error "Unknown dataset configuration"
#endif

#define K 100                    // Top-K neighbors to return per query

// ============================================================================
// GRAPH PARAMETERS
// ============================================================================

#define R 64                     // Max degree (out-neighbors per node)
#define L_SEARCH 100             // Search list size (L parameter)
#define ALPHA 1.5f              // Alpha for robust prune
#define MEDOID_ID 0             // Starting node for greedy search

// ============================================================================
// FRESHDISKANN PARAMETERS
// ============================================================================

// Fresh index sizing
#define FRESH_INDEX_CAPACITY (N / 10)  // 10% of static index size
#define FRESH_INDEX_THRESHOLD 0.08f    // Trigger consolidation at 8% full

// Consolidation triggers (hybrid strategy)
#define CONSOLIDATE_TIME_THRESHOLD 60.0f    // 60 seconds
#define CONSOLIDATE_SIZE_THRESHOLD (FRESH_INDEX_CAPACITY * FRESH_INDEX_THRESHOLD)

// Batch processing (for GPU efficiency)
#define INSERT_BATCH_SIZE 1000       // Process inserts in batches
#define QUERY_BATCH_SIZE 1000        // Process queries in batches
#define DELETE_BATCH_SIZE 1000       // Process deletes in batches

// ============================================================================
// GPU KERNEL PARAMETERS
// ============================================================================

#define THREADS_PER_BLOCK 256
#define THREADS_PER_VECTOR 8         // Threads cooperating for distance computation
#define WARP_SIZE 32
#define MAX_BLOCKS 65535

// Bloom filter for visited nodes (per query)
#define BF_ENTRIES 399887U           // Prime number for hash distribution
#define BF_HASH1_SEED 0x811c9dc5U    // FNV-1a seed
#define BF_HASH2_SEED 0x01000193U

// Memory limits
#define MAX_PARENTS_PER_QUERY (4 * L_SEARCH + 20)
#define MAX_REVERSE_INDEX_ENTRIES 500

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Graph node structure (static and fresh indices)
struct GraphNode {
    float* vector;               // D-dimensional vector (device pointer)
    uint32_t degree;            // Current number of neighbors
    uint32_t neighbors[R];      // Neighbor node IDs
    uint32_t id;                // Global node ID
    bool is_active;             // False if deleted
};

// Static index (read-only during search)
struct StaticIndex {
    float* d_vectors;           // All vectors: [N][D]
    uint32_t* d_graph;          // Graph structure: [N][1+R] (degree + neighbors)
    uint32_t num_nodes;         // Current number of nodes
    uint32_t capacity;          // Maximum capacity
    size_t vector_size_bytes;   // D * sizeof(float)
    size_t node_size_bytes;     // Size of one graph node entry
};

// Fresh index (mutable for insertions)
struct FreshIndex {
    float* d_vectors;           // Fresh vectors: [capacity][D]
    uint32_t* d_graph;          // Fresh graph: [capacity][1+R]
    uint32_t* d_count;          // Current count (device atomic counter)
    uint32_t* h_count;          // Host mirror of count
    uint32_t capacity;          // Maximum capacity
    size_t vector_size_bytes;
    size_t node_size_bytes;
};

// Delete buffer (lazy deletion bitmap)
struct DeleteBuffer {
    uint32_t* d_bitmap;         // Bitmap on device: bit_i = 1 if node i deleted
    uint32_t* h_bitmap;         // Host mirror for updates
    uint32_t total_nodes;       // Total nodes (static + fresh)
    uint32_t num_deleted;       // Count of deleted nodes
    size_t bitmap_size_bytes;   // Total bitmap size in bytes
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
    float* vector;              // Vector data (for insert/query)
    std::string scenario;       // Scenario name
};

// Performance metrics
struct PerformanceMetrics {
    // Operation counts (dynamic, tracked during workload processing)
    uint64_t total_inserts;
    uint64_t total_deletes;
    uint64_t total_queries;

    // Throughput
    double insert_qps;          // Insertions per second
    double delete_qps;          // Deletions per second
    double query_qps;           // Queries per second
    double overall_throughput;  // Total operations per second

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

    // Accuracy (computed against ground truth when available)
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
    double total_elapsed_time;  // Total workload processing time
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
void initStaticIndex(StaticIndex* index, const char* vector_file, const char* graph_file);
void initFreshIndex(FreshIndex* index);
void initDeleteBuffer(DeleteBuffer* buffer, uint32_t total_capacity);

// Core operations
void insertBatch(FreshIndex* fresh, StaticIndex* static_idx, DeleteBuffer* del_buf,
                 float* h_vectors, uint32_t* h_ids, uint32_t batch_size);
void deleteBatch(DeleteBuffer* del_buf, uint32_t* h_ids, uint32_t batch_size);
void searchDualIndex(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     float* h_queries, SearchResult* results, uint32_t num_queries);

// Consolidation
bool shouldConsolidate(FreshIndex* fresh, DeleteBuffer* del_buf, double elapsed_time);
void consolidateIndices(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf);

// Workload processing
std::vector<WorkloadEvent> loadWorkload(const char* jsonl_file, uint32_t max_events);
void processWorkload(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     const std::vector<WorkloadEvent>& workload, PerformanceMetrics* metrics);

// Metrics
void computeRecall(SearchResult* computed, uint32_t* ground_truth, uint32_t num_queries,
                   uint32_t k, PerformanceMetrics* metrics);
void printMetrics(const PerformanceMetrics* metrics);

// Utilities
void loadFvecsFile(const char* filename, float** vectors, uint32_t* num_vectors, uint32_t* dim);
void loadIvecsFile(const char* filename, uint32_t** data, uint32_t* num_vectors, uint32_t* dim);
void saveGraphBinary(const char* filename, uint32_t* graph, uint32_t num_nodes);
void loadGraphBinary(const char* filename, uint32_t** graph, uint32_t* num_nodes);

// Memory management
void checkCudaError(cudaError_t err, const char* msg);
void printGPUMemoryUsage();

// Cleanup
void freeStaticIndex(StaticIndex* index);
void freeFreshIndex(FreshIndex* index);
void freeDeleteBuffer(DeleteBuffer* buffer);

#endif // DYNAMICBANG_COMMON_H
