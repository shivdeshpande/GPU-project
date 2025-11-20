#include "concurrent_executor.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

/**
 * Concurrent FreshDiskANN Executor Implementation
 *
 * Uses multiple CUDA streams and worker threads to process
 * INSERT, DELETE, and QUERY operations in parallel while
 * maintaining correctness.
 */

// GPU kernel: Compute L2 distances from query to all graph points
__global__ void computeAllDistancesKernel(uint8_t* graph, float* queryVec,
                                          unsigned* deleted, float* distances,
                                          unsigned numPoints, unsigned graphEntrySize) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    // Check if point is deleted
    if (deleted != nullptr && deleted[tid] != 0) {
        distances[tid] = INFINITY;
        return;
    }

    // Extract point vector from graph
    float* pointVec = (float*)(graph + tid * graphEntrySize);

    // Compute L2 distance
    float dist = 0.0f;
    for (unsigned d = 0; d < D; d++) {
        float diff = queryVec[d] - pointVec[d];
        dist += diff * diff;
    }
    distances[tid] = dist;
}

ConcurrentExecutor::ConcurrentExecutor(uint8_t* d_graph,
                                       DeleteList* deleteList,
                                       float* h_queries,
                                       unsigned* h_groundtruth,
                                       unsigned gtK,
                                       unsigned k,
                                       unsigned searchL,
                                       float alpha,
                                       float consolidateThresh,
                                       unsigned numWorkers,
                                       bool useBackgroundConsolidation)
    : d_graph(d_graph), deleteList(deleteList), h_queries(h_queries),
      h_groundtruth(h_groundtruth), gtK(gtK), k(k), searchL(searchL),
      alpha(alpha), consolidateThresh(consolidateThresh),
      useBackgroundConsolidation(useBackgroundConsolidation) {

    // Allocate version array for lock-free access
    allocateVersions(&d_versions, N);

    // Initialize background consolidator
    if (useBackgroundConsolidation) {
        backgroundConsolidator = new BackgroundConsolidator(N, D, R, alpha);
    } else {
        backgroundConsolidator = nullptr;
    }

    // Create CUDA stream pools for concurrent execution
    for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
        cudaStreamCreate(&queryStreams[i]);
    }
    for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
        cudaStreamCreate(&insertStreams[i]);
    }
    cudaStreamCreate(&deleteStream);

    // Initialize pre-allocated memory pools
    initializeStreamResources();

    // Pre-allocate batch buffers (avoid cudaMalloc per batch)
    batchBuffers.allocatedSize = BATCH_SIZE;
    cudaMalloc(&batchBuffers.d_batchQueries, BATCH_SIZE * D * sizeof(float));
    cudaMalloc(&batchBuffers.d_batchVisitedSets, BATCH_SIZE * MAX_PARENTS_PERQUERY * sizeof(unsigned));
    cudaMalloc(&batchBuffers.d_batchVisitedCounts, BATCH_SIZE * sizeof(unsigned));
    cudaMalloc(&batchBuffers.d_batchDists, BATCH_SIZE * MAX_PARENTS_PERQUERY * sizeof(float));
    cudaMalloc(&batchBuffers.d_batchVisitedAux, BATCH_SIZE * MAX_PARENTS_PERQUERY * sizeof(unsigned));
    cudaMalloc(&batchBuffers.d_batchDistsAux, BATCH_SIZE * MAX_PARENTS_PERQUERY * sizeof(float));
    cudaMallocHost(&batchBuffers.h_batchQueries, BATCH_SIZE * D * sizeof(float));
    cudaMallocHost(&batchBuffers.h_batchResults, BATCH_SIZE * k * sizeof(unsigned));
    printf("  - Pre-allocated batch buffers for %d queries\n", BATCH_SIZE);

    // Initialize lock-free queues
    insertQueueLF = new LockFreeQueue<Operation>(QUEUE_CAPACITY);
    deleteQueueLF = new LockFreeQueue<Operation>(QUEUE_CAPACITY);
    queryQueueLF = new LockFreeQueue<Operation>(QUEUE_CAPACITY);

    printf("Concurrent executor initialized with %u workers\n", numWorkers);
    printf("  - Version array allocated for %u vertices\n", N);
    printf("  - %d query streams with memory pools\n", NUM_QUERY_STREAMS);
    printf("  - %d insert streams with memory pools\n", NUM_INSERT_STREAMS);
    printf("  - Batch size: %d queries\n", BATCH_SIZE);
    printf("  - Lock-free queues (capacity: %d)\n", QUEUE_CAPACITY);

    // Start worker threads
    // Calculate worker distribution based on numWorkers
    unsigned insertWorkers = std::max(1u, numWorkers / 4);      // ~25% for inserts
    unsigned deleteWorkers = 1;                                   // 1 for deletes (fast)
    unsigned queryWorkers = numWorkers - insertWorkers - deleteWorkers;  // Rest for queries

    // Insert workers - can run in parallel for different point IDs
    for (unsigned i = 0; i < insertWorkers; i++) {
        workers.emplace_back(&ConcurrentExecutor::insertWorker, this);
    }

    // Delete worker - uses atomic operations on bitvector
    for (unsigned i = 0; i < deleteWorkers; i++) {
        workers.emplace_back(&ConcurrentExecutor::deleteWorker, this);
    }

    // Batch worker - collects queries and processes them in batches for better GPU utilization
    // Use single batchWorker since it batches queries for parallel GPU execution
    workers.emplace_back(&ConcurrentExecutor::batchWorker, this);
    printf("  (Using batch processing with size %d for queries)\n", BATCH_SIZE);

    printf("Started %zu workers: %u insert, %u delete, %u query\n",
           workers.size(), insertWorkers, deleteWorkers, queryWorkers);
}

ConcurrentExecutor::~ConcurrentExecutor() {
    shutdown();

    // Wait for any background consolidation to complete
    if (backgroundConsolidator) {
        backgroundConsolidator->waitForCompletion();
        delete backgroundConsolidator;
    }

    // Free lock-free queues
    delete insertQueueLF;
    delete deleteQueueLF;
    delete queryQueueLF;

    // Free pre-allocated memory pools
    freeStreamResources();

    // Free batch buffers
    cudaFree(batchBuffers.d_batchQueries);
    cudaFree(batchBuffers.d_batchVisitedSets);
    cudaFree(batchBuffers.d_batchVisitedCounts);
    cudaFree(batchBuffers.d_batchDists);
    cudaFree(batchBuffers.d_batchVisitedAux);
    cudaFree(batchBuffers.d_batchDistsAux);
    cudaFreeHost(batchBuffers.h_batchQueries);
    cudaFreeHost(batchBuffers.h_batchResults);

    // Free version array
    freeVersions(d_versions);

    // Destroy CUDA stream pools
    for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
        cudaStreamDestroy(queryStreams[i]);
    }
    for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
        cudaStreamDestroy(insertStreams[i]);
    }
    cudaStreamDestroy(deleteStream);
}

void ConcurrentExecutor::shutdown() {
    running = false;
    queueCV.notify_all();

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers.clear();
}

void ConcurrentExecutor::submitOperation(const WorkloadEvent& event) {
    Operation op(event, nextOpId++);
    pendingOps++;

    // Use lock-free push - spin if queue is full
    bool pushed = false;
    switch (event.type) {
        case EVENT_INSERT:
            while (!(pushed = insertQueueLF->tryPush(op))) {
                std::this_thread::yield();
            }
            break;
        case EVENT_DELETE:
            while (!(pushed = deleteQueueLF->tryPush(op))) {
                std::this_thread::yield();
            }
            break;
        case EVENT_QUERY:
            while (!(pushed = queryQueueLF->tryPush(op))) {
                std::this_thread::yield();
            }
            break;
    }
}

void ConcurrentExecutor::waitForCompletion() {
    while (pendingOps > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
        cudaStreamSynchronize(queryStreams[i]);
    }
    for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
        cudaStreamSynchronize(insertStreams[i]);
    }
    cudaStreamSynchronize(deleteStream);
}

bool ConcurrentExecutor::checkConsolidation() {
    unsigned deleteCount = deleteList->getDeleteCount();
    unsigned threshold = (unsigned)((float)N * consolidateThresh / 100.0f);

    if (deleteCount >= threshold) {
        // Use background consolidation if enabled
        if (useBackgroundConsolidation && backgroundConsolidator) {
            // Check if previous consolidation is still running
            if (!backgroundConsolidator->isComplete()) {
                printf("[Info] Previous consolidation still running, skipping...\n");
                return false;
            }

            printf("\n⚠️  BACKGROUND CONSOLIDATION TRIGGERED (Deleted: %u/%u = %.2f%%)\n",
                   deleteCount, N, (float)deleteCount / N * 100.0f);
            printf("[Info] GPU continues processing while CPU rebuilds edges...\n\n");

            // Start background consolidation - GPU continues working!
            backgroundConsolidator->startConsolidation(d_graph, d_versions, deleteList);

            // Record stats when complete (check in next iteration)
            stats.addConsolidate(0);  // Time will be updated when complete

            return true;
        }

        // Fallback: stop-the-world consolidation
        consolidating = true;

        // Wait for all active operations to complete
        while (activeOps > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // Synchronize all streams
        for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
            cudaStreamSynchronize(queryStreams[i]);
        }
        for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
            cudaStreamSynchronize(insertStreams[i]);
        }
        cudaStreamSynchronize(deleteStream);

        std::unique_lock<std::mutex> consoleLock(consolidateMutex);

        printf("\n⚠️  STOP-THE-WORLD CONSOLIDATION (Deleted: %u/%u = %.2f%%)\n",
               deleteCount, N, (float)deleteCount / N * 100.0f);

        auto start = std::chrono::high_resolution_clock::now();

        unsigned affectedNodes = consolidateDeletes(d_graph, deleteList, alpha, true);

        auto end = std::chrono::high_resolution_clock::now();
        double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

        stats.addConsolidate(timeMs);

        printf("Consolidation time: %.2f ms, affected nodes: %u\n\n", timeMs, affectedNodes);

        // Resume operations
        consolidating = false;

        return true;
    }

    return false;
}

void ConcurrentExecutor::insertWorker() {
    while (running) {
        Operation op;

        // Try to pop from lock-free queue
        if (insertQueueLF->tryPop(op)) {
            // Wait if consolidation is in progress
            while (consolidating) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            activeOps++;
            processInsert(op);
            activeOps--;
            pendingOps--;
        } else {
            // No work, yield to avoid busy-waiting
            std::this_thread::yield();
        }
    }
}

void ConcurrentExecutor::deleteWorker() {
    while (running) {
        Operation op;

        // Try to pop from lock-free queue
        if (deleteQueueLF->tryPop(op)) {
            // Wait if consolidation is in progress
            while (consolidating) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            activeOps++;
            processDelete(op);
            activeOps--;
            pendingOps--;
        } else {
            // No work, yield to avoid busy-waiting
            std::this_thread::yield();
        }
    }
}

void ConcurrentExecutor::queryWorker() {
    while (running) {
        Operation op;

        // Try to pop from lock-free queue
        if (queryQueueLF->tryPop(op)) {
            // Wait if consolidation is in progress
            while (consolidating) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            activeOps++;
            processQuery(op);
            activeOps--;
            pendingOps--;
        } else {
            // No work, yield to avoid busy-waiting
            std::this_thread::yield();
        }
    }
}

void ConcurrentExecutor::processInsert(const Operation& op) {
    auto start = std::chrono::high_resolution_clock::now();

    // Acquire stream with pre-allocated resources
    int streamIdx = acquireInsertStream();
    cudaStream_t stream = insertStreams[streamIdx];
    InsertStreamResources& res = insertResources[streamIdx];

    // Copy vector data to pinned memory (faster than pageable)
    for (unsigned i = 0; i < D && i < op.event.vector.size(); i++) {
        res.h_vector[i] = op.event.vector[i];
    }
    for (unsigned i = op.event.vector.size(); i < D; i++) {
        res.h_vector[i] = 0.0f;
    }

    // Async copy using pinned memory (truly async)
    cudaMemcpyAsync(res.d_vector, res.h_vector, D * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Insert point with pre-allocated buffers (avoids cudaMalloc overhead!)
    insertPointVersionedPrealloc(d_graph, d_versions, res.d_vector, op.event.pointId,
                                  alpha, &res.insertBuffers, stream);

    auto end = std::chrono::high_resolution_clock::now();
    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    stats.addInsert(timeMs);

    releaseInsertStream(streamIdx);
}

void ConcurrentExecutor::processDelete(const Operation& op) {
    auto start = std::chrono::high_resolution_clock::now();

    // Delete uses atomic operations, no lock needed
    deleteList->markDeleted(op.event.pointId);

    auto end = std::chrono::high_resolution_clock::now();
    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    stats.addDelete(timeMs);
}

void ConcurrentExecutor::processQuery(const Operation& op) {
    auto start = std::chrono::high_resolution_clock::now();

    // Acquire stream with pre-allocated resources
    int streamIdx = acquireQueryStream();
    cudaStream_t stream = queryStreams[streamIdx];
    QueryStreamResources& res = queryResources[streamIdx];

    // Determine query vector source
    unsigned* gtForQuery = nullptr;
    bool hasGroundtruth = false;

    if (!op.event.vector.empty()) {
        // Query has embedded vector - copy to pinned memory
        for (unsigned i = 0; i < D && i < op.event.vector.size(); i++) {
            res.h_queryVec[i] = op.event.vector[i];
        }
        for (unsigned i = op.event.vector.size(); i < D; i++) {
            res.h_queryVec[i] = 0.0f;
        }
    } else if (h_queries != nullptr) {
        // Query uses query_id - copy from query set to pinned memory
        unsigned queryId = op.event.queryId;
        memcpy(res.h_queryVec, h_queries + queryId * D, D * sizeof(float));
        if (h_groundtruth != nullptr) {
            gtForQuery = h_groundtruth + queryId * gtK;
            hasGroundtruth = true;
        }
    } else {
        // No query vector available
        stats.addQuery(0, 0, 0);
        releaseQueryStream(streamIdx);
        return;
    }

    // Async copy using pinned memory (truly async)
    cudaMemcpyAsync(res.d_queryVec, res.h_queryVec, D * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Compute groundtruth for embedded vectors
    unsigned* computedGroundtruth = nullptr;
    unsigned effectiveGtK = gtK;
    if (!hasGroundtruth && !op.event.vector.empty()) {
        computedGroundtruth = (unsigned*)malloc(k * sizeof(unsigned));
        computeGroundtruthForQuery(res.d_queryVec, computedGroundtruth, k, stream);
        gtForQuery = computedGroundtruth;
        effectiveGtK = k;
        hasGroundtruth = true;
    }

    // Reset visited set count
    cudaMemsetAsync(res.d_visitedSetCount, 0, sizeof(unsigned), stream);

    // Run GreedySearch with pre-allocated buffers on this stream (enables parallelism)
    unsigned* d_deleted = deleteList->getDevicePointer();
    greedySearchVersionedPrealloc(d_graph, d_versions, res.d_queryVec, res.d_visitedSet,
                                   res.d_visitedSetCount, 0, 1, searchL, d_deleted,
                                   &res.gsBuffers, stream);

    // Compute distances using pre-allocated buffers
    computeDists<<<1, MAX_PARENTS_PERQUERY, 0, stream>>>(
        d_graph, res.d_visitedSet, res.d_visitedSetCount,
        res.d_queryVec, res.d_visitedSetDists, MAX_PARENTS_PERQUERY);

    // Sort by distance using pre-allocated buffers
    sortByDistance<<<1, MAX_PARENTS_PERQUERY, MAX_PARENTS_PERQUERY * sizeof(unsigned), stream>>>(
        res.d_visitedSet, res.d_visitedSetCount, res.d_visitedSetDists,
        res.d_visitedSetAux, res.d_visitedSetDistsAux, MAX_PARENTS_PERQUERY);

    // Copy results back to pinned memory (faster)
    cudaMemcpyAsync(res.h_results, res.d_visitedSet, k * sizeof(unsigned),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate recall
    double recall5 = 0.0, recall10 = 0.0;
    if (hasGroundtruth && gtForQuery != nullptr) {
        recall5 = calculateRecall(gtForQuery, res.h_results, effectiveGtK, k, std::min(k, 5u));
        recall10 = calculateRecall(gtForQuery, res.h_results, effectiveGtK, k, std::min(k, 10u));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    stats.addQuery(timeMs, recall5, recall10);

    // Cleanup only dynamic allocations
    if (computedGroundtruth) free(computedGroundtruth);

    releaseQueryStream(streamIdx);
}

void ConcurrentExecutor::computeGroundtruthForQuery(float* d_queryVec, unsigned* h_gt, unsigned k, cudaStream_t stream) {
    // Allocate distance array on GPU
    float* d_distances;
    cudaMalloc(&d_distances, N * sizeof(float));

    // Get delete bitvector pointer
    unsigned* d_deleted = deleteList->getDevicePointer();

    // Compute distances to all points
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeAllDistancesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_graph, d_queryVec, d_deleted, d_distances, N, graphEntrySize);
    cudaStreamSynchronize(stream);

    // Copy distances to host
    float* h_distances = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_distances, d_distances, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Find top-k on CPU
    std::vector<std::pair<float, unsigned>> distPairs;
    distPairs.reserve(N);

    for (unsigned i = 0; i < N; i++) {
        if (h_distances[i] != INFINITY) {
            distPairs.push_back({h_distances[i], i});
        }
    }

    unsigned actualK = std::min(k, (unsigned)distPairs.size());
    std::partial_sort(distPairs.begin(), distPairs.begin() + actualK, distPairs.end(),
                      [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                          return a.first < b.first;
                      });

    for (unsigned i = 0; i < actualK; i++) {
        h_gt[i] = distPairs[i].second;
    }
    for (unsigned i = actualK; i < k; i++) {
        h_gt[i] = 0;
    }

    free(h_distances);
    cudaFree(d_distances);
}

double ConcurrentExecutor::calculateRecall(unsigned* groundtruth, unsigned* results,
                                           unsigned gtK, unsigned k, unsigned recallK) {
    unsigned matches = 0;
    for (unsigned i = 0; i < recallK && i < k; i++) {
        for (unsigned j = 0; j < recallK && j < gtK; j++) {
            if (results[i] == groundtruth[j]) {
                matches++;
                break;
            }
        }
    }
    return (100.0 * matches) / recallK;
}

// ============== Resource Management ==============

void ConcurrentExecutor::initializeStreamResources() {
    // Initialize query stream resources with pinned memory
    for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
        QueryStreamResources& res = queryResources[i];

        // GPU allocations
        cudaMalloc(&res.d_queryVec, D * sizeof(float));
        cudaMalloc(&res.d_visitedSet, MAX_PARENTS_PERQUERY * sizeof(unsigned));
        cudaMalloc(&res.d_visitedSetCount, sizeof(unsigned));
        cudaMalloc(&res.d_visitedSetDists, MAX_PARENTS_PERQUERY * sizeof(float));
        cudaMalloc(&res.d_visitedSetAux, MAX_PARENTS_PERQUERY * sizeof(unsigned));
        cudaMalloc(&res.d_visitedSetDistsAux, MAX_PARENTS_PERQUERY * sizeof(float));

        // Pre-allocate greedy search buffers (avoids cudaMalloc per query)
        allocateGreedySearchBuffers(&res.gsBuffers, 1);

        // Pinned host memory for faster async transfers
        cudaMallocHost(&res.h_queryVec, D * sizeof(float));
        cudaMallocHost(&res.h_results, k * sizeof(unsigned));

        res.inUse = false;
    }

    // Initialize insert stream resources with pinned memory
    for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
        InsertStreamResources& res = insertResources[i];

        cudaMalloc(&res.d_vector, D * sizeof(float));
        cudaMallocHost(&res.h_vector, D * sizeof(float));

        // Pre-allocate insert buffers (avoids ~20MB cudaMalloc per insert!)
        allocateInsertBuffers(&res.insertBuffers);

        res.inUse = false;
    }
}

void ConcurrentExecutor::freeStreamResources() {
    // Free query stream resources
    for (int i = 0; i < NUM_QUERY_STREAMS; i++) {
        QueryStreamResources& res = queryResources[i];

        cudaFree(res.d_queryVec);
        cudaFree(res.d_visitedSet);
        cudaFree(res.d_visitedSetCount);
        cudaFree(res.d_visitedSetDists);
        cudaFree(res.d_visitedSetAux);
        cudaFree(res.d_visitedSetDistsAux);

        // Free pre-allocated greedy search buffers
        freeGreedySearchBuffers(&res.gsBuffers);

        cudaFreeHost(res.h_queryVec);
        cudaFreeHost(res.h_results);
    }

    // Free insert stream resources
    for (int i = 0; i < NUM_INSERT_STREAMS; i++) {
        InsertStreamResources& res = insertResources[i];

        cudaFree(res.d_vector);
        cudaFreeHost(res.h_vector);

        // Free pre-allocated insert buffers
        freeInsertBuffers(&res.insertBuffers);
    }
}

int ConcurrentExecutor::acquireQueryStream() {
    // Try to find an available stream
    for (int attempts = 0; attempts < NUM_QUERY_STREAMS * 2; attempts++) {
        unsigned idx = queryStreamIdx.fetch_add(1) % NUM_QUERY_STREAMS;

        std::lock_guard<std::mutex> lock(resourceMutex[idx]);
        if (!queryResources[idx].inUse) {
            queryResources[idx].inUse = true;
            return idx;
        }
    }

    // All streams busy, wait for first available
    unsigned idx = queryStreamIdx.fetch_add(1) % NUM_QUERY_STREAMS;
    std::lock_guard<std::mutex> lock(resourceMutex[idx]);
    cudaStreamSynchronize(queryStreams[idx]);
    queryResources[idx].inUse = true;
    return idx;
}

void ConcurrentExecutor::releaseQueryStream(int idx) {
    std::lock_guard<std::mutex> lock(resourceMutex[idx]);
    queryResources[idx].inUse = false;
}

int ConcurrentExecutor::acquireInsertStream() {
    // Round-robin with simple availability check
    for (int attempts = 0; attempts < NUM_INSERT_STREAMS * 2; attempts++) {
        unsigned idx = insertStreamIdx.fetch_add(1) % NUM_INSERT_STREAMS;

        if (!insertResources[idx].inUse) {
            insertResources[idx].inUse = true;
            return idx;
        }
    }

    // All streams busy, wait for first
    unsigned idx = insertStreamIdx.fetch_add(1) % NUM_INSERT_STREAMS;
    cudaStreamSynchronize(insertStreams[idx]);
    insertResources[idx].inUse = true;
    return idx;
}

void ConcurrentExecutor::releaseInsertStream(int idx) {
    insertResources[idx].inUse = false;
}

// ============== Batch Processing ==============

void ConcurrentExecutor::processBatchQueries(std::vector<Operation>& batch) {
    if (batch.empty()) return;

    auto start = std::chrono::high_resolution_clock::now();

    unsigned batchSize = batch.size();

    // Use pre-allocated buffers (no cudaMalloc overhead!)
    float* d_batchQueries = batchBuffers.d_batchQueries;
    unsigned* d_batchVisitedSets = batchBuffers.d_batchVisitedSets;
    unsigned* d_batchVisitedCounts = batchBuffers.d_batchVisitedCounts;
    float* d_batchDists = batchBuffers.d_batchDists;
    unsigned* d_batchVisitedAux = batchBuffers.d_batchVisitedAux;
    float* d_batchDistsAux = batchBuffers.d_batchDistsAux;
    float* h_batchQueries = batchBuffers.h_batchQueries;
    unsigned* h_batchResults = batchBuffers.h_batchResults;

    // Prepare all queries in batch
    for (unsigned i = 0; i < batchSize; i++) {
        const Operation& op = batch[i];
        float* queryDst = h_batchQueries + i * D;

        if (!op.event.vector.empty()) {
            for (unsigned j = 0; j < D && j < op.event.vector.size(); j++) {
                queryDst[j] = op.event.vector[j];
            }
            for (unsigned j = op.event.vector.size(); j < D; j++) {
                queryDst[j] = 0.0f;
            }
        } else if (h_queries != nullptr) {
            memcpy(queryDst, h_queries + op.event.queryId * D, D * sizeof(float));
        }
    }

    // Single large transfer instead of many small ones
    cudaMemcpy(d_batchQueries, h_batchQueries, batchSize * D * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_batchVisitedCounts, 0, batchSize * sizeof(unsigned));

    unsigned* d_deleted = deleteList->getDevicePointer();

    // TRUE GPU BATCHING: Single call with batchSize blocks - all queries process in parallel!
    greedySearchVersioned(d_graph, d_versions, d_batchQueries, d_batchVisitedSets,
                          d_batchVisitedCounts, 0, batchSize, searchL, d_deleted);

    // Batched distance computation and sorting - all queries in parallel
    computeDists<<<batchSize, R*8>>>(
        d_graph, d_batchVisitedSets, d_batchVisitedCounts,
        d_batchQueries, d_batchDists, MAX_PARENTS_PERQUERY);

    sortByDistance<<<batchSize, MAX_PARENTS_PERQUERY, MAX_PARENTS_PERQUERY * sizeof(unsigned)>>>(
        d_batchVisitedSets, d_batchVisitedCounts, d_batchDists,
        d_batchVisitedAux, d_batchDistsAux, MAX_PARENTS_PERQUERY);

    cudaDeviceSynchronize();

    // Copy all results back in one transfer
    for (unsigned i = 0; i < batchSize; i++) {
        unsigned* d_visitedSet = d_batchVisitedSets + i * MAX_PARENTS_PERQUERY;
        cudaMemcpy(h_batchResults + i * k, d_visitedSet, k * sizeof(unsigned),
                   cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgTimeMs = totalTimeMs / batchSize;

    // Calculate recall for each query and update stats
    for (unsigned i = 0; i < batchSize; i++) {
        const Operation& op = batch[i];
        unsigned* results = h_batchResults + i * k;

        double recall5 = 0.0, recall10 = 0.0;
        if (h_groundtruth != nullptr && op.event.vector.empty()) {
            unsigned* gtForQuery = h_groundtruth + op.event.queryId * gtK;
            recall5 = calculateRecall(gtForQuery, results, gtK, k, std::min(k, 5u));
            recall10 = calculateRecall(gtForQuery, results, gtK, k, std::min(k, 10u));
        }

        stats.addQuery(avgTimeMs, recall5, recall10);
    }

    // No cleanup needed - using pre-allocated buffers
}

void ConcurrentExecutor::batchWorker() {
    std::vector<Operation> localBatch;
    localBatch.reserve(BATCH_SIZE);

    while (running) {
        // Collect queries from lock-free queue until batch is full or timeout
        Operation op;
        auto batchStart = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::milliseconds(2);  // 2ms timeout for batching

        while (localBatch.size() < BATCH_SIZE) {
            if (queryQueueLF->tryPop(op)) {
                localBatch.push_back(op);
            } else {
                // Check timeout
                auto elapsed = std::chrono::steady_clock::now() - batchStart;
                if (elapsed > timeout && !localBatch.empty()) {
                    break;  // Process what we have
                }
                if (elapsed > timeout && localBatch.empty() && !running) {
                    break;  // Exit condition
                }
                std::this_thread::yield();
            }
        }

        if (!running && localBatch.empty()) break;

        if (!localBatch.empty()) {
            // Wait if consolidation is in progress
            while (consolidating) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            activeOps += localBatch.size();
            processBatchQueries(localBatch);
            activeOps -= localBatch.size();
            pendingOps -= localBatch.size();

            localBatch.clear();
        }
    }
}
