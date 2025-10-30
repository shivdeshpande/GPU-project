#include "dynamicBANG.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

// ============================================================================
// WORKLOAD LOADING FROM JSONL FILES
// ============================================================================

std::vector<WorkloadEvent> loadWorkload(const char* jsonl_file, uint32_t max_events) {
    printf("[Workload] Loading from %s...\n", jsonl_file);

    std::vector<WorkloadEvent> events;
    std::ifstream file(jsonl_file);

    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open workload file %s\n", jsonl_file);
        return events;
    }

    std::string line;
    uint32_t line_num = 0;
    uint32_t insert_count = 0;
    uint32_t delete_count = 0;
    uint32_t query_count = 0;

    while (std::getline(file, line) && events.size() < max_events) {
        line_num++;

        if (line.empty()) continue;

        try {
            json j = json::parse(line);

            // Skip metadata lines
            if (j.contains("type") && j["type"] == "metadata") {
                printf("[Workload] Metadata: scenario=%s, total_events=%d\n",
                       j["scenario"].get<string>().c_str(),
                       j["total_events"].get<int>());
                continue;
            }

            WorkloadEvent event;
            event.timestamp = j["t"].get<uint64_t>();
            std::string type_str = j["type"].get<string>();

            if (type_str == "insert") {
                event.type = EVENT_INSERT;
                event.id = j["id"].get<uint32_t>();

                // Allocate and copy vector
                auto vec_json = j["vec"];
                event.vector = (datatype_t*)malloc(D * sizeof(datatype_t));
                for (size_t i = 0; i < D && i < vec_json.size(); i++) {
                    event.vector[i] = (datatype_t)vec_json[i].get<double>();
                }
                insert_count++;

            } else if (type_str == "delete") {
                event.type = EVENT_DELETE;
                event.id = j["id"].get<uint32_t>();
                event.vector = nullptr;
                delete_count++;

            } else if (type_str == "query") {
                event.type = EVENT_QUERY;
                event.id = 0;  // Queries don't have IDs

                // Allocate and copy vector
                auto vec_json = j["vec"];
                event.vector = (datatype_t*)malloc(D * sizeof(datatype_t));
                for (size_t i = 0; i < D && i < vec_json.size(); i++) {
                    event.vector[i] = (datatype_t)vec_json[i].get<double>();
                }
                query_count++;

            } else {
                continue;  // Unknown type
            }

            if (j.contains("scenario")) {
                event.scenario = j["scenario"].get<string>();
            }

            events.push_back(event);

        } catch (const std::exception& e) {
            fprintf(stderr, "Warning: Failed to parse line %u: %s\n", line_num, e.what());
            continue;
        }
    }

    file.close();

    printf("[Workload] Loaded %lu events: %u inserts, %u deletes, %u queries\n",
           events.size(), insert_count, delete_count, query_count);

    return events;
}

void freeWorkload(std::vector<WorkloadEvent>& events) {
    for (auto& event : events) {
        if (event.vector) {
            free(event.vector);
            event.vector = nullptr;
        }
    }
    events.clear();
}

// ============================================================================
// WORKLOAD PROCESSING
// ============================================================================

void processWorkload(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf,
                     const std::vector<WorkloadEvent>& workload, PerformanceMetrics* metrics,
                     uint32_t* ground_truth, uint32_t gt_dim, uint32_t recall_at) {

    printf("[Workload] Processing %lu events...\n", workload.size());

    CPUTimer total_timer;
    total_timer.Start();

    // Timing accumulators
    std::vector<double> insert_latencies;
    std::vector<double> delete_latencies;
    std::vector<double> query_latencies;

    // Batch buffers
    std::vector<datatype_t*> insert_vectors;
    std::vector<uint32_t> insert_ids;
    std::vector<uint32_t> delete_ids;
    std::vector<datatype_t*> query_vectors;

    double last_consolidation_time = 0.0;
    uint32_t consolidation_count = 0;

    for (size_t i = 0; i < workload.size(); i++) {
        const WorkloadEvent& event = workload[i];

        CPUTimer event_timer;
        event_timer.Start();

        switch (event.type) {
            case EVENT_INSERT:
                insert_vectors.push_back(event.vector);
                insert_ids.push_back(event.id);

                // Process batch
                if (insert_vectors.size() >= INSERT_BATCH_SIZE) {
                    // Flatten vectors
                    datatype_t* h_vectors = (datatype_t*)malloc(insert_vectors.size() * D * sizeof(datatype_t));
                    for (size_t j = 0; j < insert_vectors.size(); j++) {
                        memcpy(h_vectors + j * D, insert_vectors[j], D * sizeof(datatype_t));
                    }

                    uint32_t* h_ids = (uint32_t*)malloc(insert_vectors.size() * sizeof(uint32_t));

                    insertBatch(fresh, static_idx, del_buf, h_vectors, h_ids, insert_vectors.size());

                    event_timer.Stop();
                    insert_latencies.push_back(event_timer.Elapsed() * 1000.0);  // Convert to ms

                    free(h_vectors);
                    free(h_ids);
                    insert_vectors.clear();
                    insert_ids.clear();

                    metrics->total_inserts += insert_vectors.size();
                }
                break;

            case EVENT_DELETE:
                delete_ids.push_back(event.id);

                // Process batch
                if (delete_ids.size() >= DELETE_BATCH_SIZE) {
                    deleteBatch(del_buf, delete_ids.data(), delete_ids.size());

                    event_timer.Stop();
                    delete_latencies.push_back(event_timer.Elapsed() * 1000.0);

                    delete_ids.clear();
                    metrics->total_deletes += delete_ids.size();
                }
                break;

            case EVENT_QUERY:
                query_vectors.push_back(event.vector);

                // Process batch
                if (query_vectors.size() >= QUERY_BATCH_SIZE) {
                    // Flatten vectors
                    datatype_t* h_queries = (datatype_t*)malloc(query_vectors.size() * D * sizeof(datatype_t));
                    for (size_t j = 0; j < query_vectors.size(); j++) {
                        memcpy(h_queries + j * D, query_vectors[j], D * sizeof(datatype_t));
                    }

                    uint32_t* h_results = (uint32_t*)malloc(query_vectors.size() * recall_at * sizeof(uint32_t));

                    searchDualIndex(static_idx, fresh, del_buf, h_queries, h_results,
                                   query_vectors.size(), recall_at);

                    event_timer.Stop();
                    query_latencies.push_back(event_timer.Elapsed() * 1000.0);

                    free(h_queries);
                    free(h_results);
                    query_vectors.clear();

                    metrics->total_queries += query_vectors.size();
                }
                break;

            default:
                break;
        }

        // Check for consolidation
        total_timer.Stop();
        double elapsed = total_timer.Elapsed() - last_consolidation_time;

        if (shouldConsolidate(fresh, del_buf, elapsed)) {
            consolidateIndices(static_idx, fresh, del_buf);
            last_consolidation_time = total_timer.Elapsed();
            consolidation_count++;
        }
    }

    // Process remaining batches
    if (!insert_vectors.empty()) {
        datatype_t* h_vectors = (datatype_t*)malloc(insert_vectors.size() * D * sizeof(datatype_t));
        for (size_t j = 0; j < insert_vectors.size(); j++) {
            memcpy(h_vectors + j * D, insert_vectors[j], D * sizeof(datatype_t));
        }
        uint32_t* h_ids = (uint32_t*)malloc(insert_vectors.size() * sizeof(uint32_t));
        insertBatch(fresh, static_idx, del_buf, h_vectors, h_ids, insert_vectors.size());
        free(h_vectors);
        free(h_ids);
        metrics->total_inserts += insert_vectors.size();
    }

    if (!delete_ids.empty()) {
        deleteBatch(del_buf, delete_ids.data(), delete_ids.size());
        metrics->total_deletes += delete_ids.size();
    }

    if (!query_vectors.empty()) {
        datatype_t* h_queries = (datatype_t*)malloc(query_vectors.size() * D * sizeof(datatype_t));
        for (size_t j = 0; j < query_vectors.size(); j++) {
            memcpy(h_queries + j * D, query_vectors[j], D * sizeof(datatype_t));
        }
        uint32_t* h_results = (uint32_t*)malloc(query_vectors.size() * recall_at * sizeof(uint32_t));
        searchDualIndex(static_idx, fresh, del_buf, h_queries, h_results,
                       query_vectors.size(), recall_at);
        free(h_queries);
        free(h_results);
        metrics->total_queries += query_vectors.size();
    }

    total_timer.Stop();

    // Compute metrics
    metrics->total_elapsed_time = total_timer.Elapsed();
    metrics->num_consolidations = consolidation_count;

    // Calculate latency statistics
    if (!insert_latencies.empty()) {
        double sum = 0;
        for (double lat : insert_latencies) sum += lat;
        metrics->insert_latency_avg = sum / insert_latencies.size();
        metrics->insert_qps = metrics->total_inserts / metrics->total_elapsed_time;
    }

    if (!delete_latencies.empty()) {
        double sum = 0;
        for (double lat : delete_latencies) sum += lat;
        metrics->delete_latency_avg = sum / delete_latencies.size();
        metrics->delete_qps = metrics->total_deletes / metrics->total_elapsed_time;
    }

    if (!query_latencies.empty()) {
        double sum = 0;
        for (double lat : query_latencies) sum += lat;
        metrics->query_latency_avg = sum / query_latencies.size();
        metrics->query_qps = metrics->total_queries / metrics->total_elapsed_time;
    }

    metrics->overall_throughput = (metrics->total_inserts + metrics->total_deletes + metrics->total_queries) / metrics->total_elapsed_time;
    metrics->static_index_size = static_idx->num_nodes;
    metrics->fresh_index_size = *fresh->h_count;
    metrics->num_deleted = del_buf->num_deleted;

    printf("[Workload] Processing complete in %.2f seconds\n", metrics->total_elapsed_time);
}
