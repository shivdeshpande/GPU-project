#ifndef CONCURRENT_EXECUTOR_H
#define CONCURRENT_EXECUTOR_H

#include "../vamana.h"
#include "deleteList.h"
#include "insert.h"
#include "consolidate.h"
#include "workload.h"
#include "lockfree_graph.cuh"
#include "background_consolidate.h"
#include "lockfree_queue.h"
#include <cuda_runtime.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <vector>
#include <functional>

// Lock-free queue capacity
#define QUEUE_CAPACITY 4096

// Number of streams per operation type for parallelism
#define NUM_QUERY_STREAMS 8
#define NUM_INSERT_STREAMS 4
#define BATCH_SIZE 32  // Number of queries to batch together

// Pre-allocated resources per stream to avoid malloc overhead
struct QueryStreamResources {
    float* d_queryVec;           // Query vector on GPU
    float* h_queryVec;           // Pinned host memory for query
    unsigned* d_visitedSet;
    unsigned* d_visitedSetCount;
    float* d_visitedSetDists;
    unsigned* d_visitedSetAux;
    float* d_visitedSetDistsAux;
    unsigned* h_results;         // Pinned host memory for results
    GreedySearchBuffers gsBuffers;  // Pre-allocated greedy search buffers
    bool inUse;                  // Flag to track if stream is busy
};

struct InsertStreamResources {
    float* d_vector;             // Vector on GPU
    float* h_vector;             // Pinned host memory
    InsertBuffers insertBuffers; // Pre-allocated insert buffers
    bool inUse;
};

// Operation wrapper for queue (defined before BatchQuery which uses it)
struct Operation {
    WorkloadEvent event;
    unsigned operationId;

    Operation() : operationId(0) {}
    Operation(const WorkloadEvent& e, unsigned id) : event(e), operationId(id) {}
};

// Batch query structure for processing multiple queries together
struct BatchQuery {
    std::vector<Operation> ops;
    unsigned count;
};

// Pre-allocated buffers for batch processing (avoid cudaMalloc per batch)
struct BatchBuffers {
    float* d_batchQueries;
    unsigned* d_batchVisitedSets;
    unsigned* d_batchVisitedCounts;
    float* d_batchDists;
    unsigned* d_batchVisitedAux;
    float* d_batchDistsAux;
    float* h_batchQueries;      // Pinned host memory
    unsigned* h_batchResults;   // Pinned host memory
    unsigned allocatedSize;     // Size these were allocated for
};

/**
 * Concurrent FreshDiskANN Executor
 *
 * Supports parallel execution of INSERT, DELETE, and QUERY operations
 * while maintaining correctness through:
 * - Read-write locks for graph access
 * - Multiple CUDA streams for concurrent kernel execution
 * - Atomic operations for delete list
 * - Exclusive locking for consolidation
 */

// Result structure for completed operations
struct OperationResult {
    unsigned operationId;
    EventType type;
    double timeMs;
    double recall5;
    double recall10;
    bool success;

    OperationResult() : operationId(0), type(EVENT_QUERY), timeMs(0),
                        recall5(0), recall10(0), success(false) {}
};

// Concurrent statistics with atomic counters
struct ConcurrentStatistics {
    std::atomic<unsigned> insertsProcessed{0};
    std::atomic<unsigned> deletesProcessed{0};
    std::atomic<unsigned> queriesProcessed{0};
    std::atomic<unsigned> consolidationsTriggered{0};

    std::atomic<double> totalInsertTime{0.0};
    std::atomic<double> totalDeleteTime{0.0};
    std::atomic<double> totalQueryTime{0.0};
    std::atomic<double> totalConsolidateTime{0.0};
    std::atomic<double> totalRecall5{0.0};
    std::atomic<double> totalRecall10{0.0};

    void addInsert(double timeMs) {
        insertsProcessed++;
        double old = totalInsertTime.load();
        while (!totalInsertTime.compare_exchange_weak(old, old + timeMs));
    }

    void addDelete(double timeMs) {
        deletesProcessed++;
        double old = totalDeleteTime.load();
        while (!totalDeleteTime.compare_exchange_weak(old, old + timeMs));
    }

    void addQuery(double timeMs, double r5, double r10) {
        queriesProcessed++;
        double old = totalQueryTime.load();
        while (!totalQueryTime.compare_exchange_weak(old, old + timeMs));
        old = totalRecall5.load();
        while (!totalRecall5.compare_exchange_weak(old, old + r5));
        old = totalRecall10.load();
        while (!totalRecall10.compare_exchange_weak(old, old + r10));
    }

    void addConsolidate(double timeMs) {
        consolidationsTriggered++;
        double old = totalConsolidateTime.load();
        while (!totalConsolidateTime.compare_exchange_weak(old, old + timeMs));
    }

    void print() const {
        printf("\n=== Concurrent Workload Statistics ===\n");
        printf("Inserts:        %u\n", insertsProcessed.load());
        printf("Deletes:        %u\n", deletesProcessed.load());
        printf("Queries:        %u\n", queriesProcessed.load());
        printf("Consolidations: %u\n", consolidationsTriggered.load());
        printf("\nTiming:\n");

        unsigned ins = insertsProcessed.load();
        unsigned del = deletesProcessed.load();
        unsigned qry = queriesProcessed.load();
        unsigned con = consolidationsTriggered.load();

        if (ins > 0)
            printf("  Insert avg:      %.3f ms\n", totalInsertTime.load() / ins);
        if (del > 0)
            printf("  Delete avg:      %.3f ms\n", totalDeleteTime.load() / del);
        if (qry > 0) {
            printf("  Query avg:       %.3f ms\n", totalQueryTime.load() / qry);
            printf("  Query QPS:       %.1f\n", qry / (totalQueryTime.load() / 1000.0));
            printf("  5-recall@5:      %.2f%%\n", totalRecall5.load() / qry);
            printf("  10-recall@10:    %.2f%%\n", totalRecall10.load() / qry);
        }
        if (con > 0)
            printf("  Consolidate avg: %.3f ms\n", totalConsolidateTime.load() / con);
    }
};

class ConcurrentExecutor {
public:
    ConcurrentExecutor(uint8_t* d_graph,
                       DeleteList* deleteList,
                       float* h_queries,
                       unsigned* h_groundtruth,
                       unsigned gtK,
                       unsigned k,
                       unsigned searchL,
                       float alpha,
                       float consolidateThresh,
                       unsigned numWorkers = 4,
                       bool useBackgroundConsolidation = true);

    ~ConcurrentExecutor();

    // Submit operation for async execution
    void submitOperation(const WorkloadEvent& event);

    // Wait for all pending operations to complete
    void waitForCompletion();

    // Check and trigger consolidation if needed
    bool checkConsolidation();

    // Get statistics
    const ConcurrentStatistics& getStats() const { return stats; }

    // Shutdown executor
    void shutdown();

private:
    // Graph and data structures
    uint8_t* d_graph;
    unsigned* d_versions;     // Version array for lock-free access
    DeleteList* deleteList;
    float* h_queries;
    unsigned* h_groundtruth;
    unsigned gtK, k, searchL;
    float alpha;
    float consolidateThresh;

    // CUDA stream pools for concurrent execution
    cudaStream_t queryStreams[NUM_QUERY_STREAMS];
    cudaStream_t insertStreams[NUM_INSERT_STREAMS];
    cudaStream_t deleteStream;
    std::atomic<unsigned> queryStreamIdx{0};
    std::atomic<unsigned> insertStreamIdx{0};

    // Pre-allocated resources per stream (memory pools)
    QueryStreamResources queryResources[NUM_QUERY_STREAMS];
    InsertStreamResources insertResources[NUM_INSERT_STREAMS];
    std::mutex resourceMutex[NUM_QUERY_STREAMS];  // Protect resource allocation

    // Batch processing
    std::vector<Operation> queryBatch;
    std::mutex batchMutex;
    std::condition_variable batchCV;
    std::atomic<bool> batchReady{false};
    BatchBuffers batchBuffers;  // Pre-allocated buffers for batch queries

    // Synchronization primitives
    std::mutex queueMutex;                  // Protects legacy queue operations
    std::condition_variable_any queueCV;    // Signal for new operations
    std::mutex consolidateMutex;            // Exclusive lock for consolidation
    std::atomic<bool> consolidating{false}; // Flag to pause ops during consolidation
    std::atomic<unsigned> activeOps{0};     // Count of active operations

    // Background consolidation
    BackgroundConsolidator* backgroundConsolidator;
    bool useBackgroundConsolidation;

    // Lock-free operation queues (no mutex needed for push/pop)
    LockFreeQueue<Operation>* insertQueueLF;
    LockFreeQueue<Operation>* deleteQueueLF;
    LockFreeQueue<Operation>* queryQueueLF;

    // Worker threads
    std::vector<std::thread> workers;
    std::atomic<bool> running{true};
    std::atomic<unsigned> pendingOps{0};
    std::atomic<unsigned> nextOpId{0};

    // Statistics
    ConcurrentStatistics stats;

    // Worker thread functions
    void insertWorker();
    void deleteWorker();
    void queryWorker();

    // Process individual operations
    void processInsert(const Operation& op);
    void processDelete(const Operation& op);
    void processQuery(const Operation& op);

    // Helper functions
    void computeGroundtruthForQuery(float* d_queryVec, unsigned* h_gt, unsigned k, cudaStream_t stream);
    double calculateRecall(unsigned* groundtruth, unsigned* results,
                          unsigned gtK, unsigned k, unsigned recallK);

    // Resource management
    void initializeStreamResources();
    void freeStreamResources();
    int acquireQueryStream();
    void releaseQueryStream(int idx);
    int acquireInsertStream();
    void releaseInsertStream(int idx);

    // Batch processing
    void processBatchQueries(std::vector<Operation>& batch);
    void batchWorker();
};

#endif // CONCURRENT_EXECUTOR_H
