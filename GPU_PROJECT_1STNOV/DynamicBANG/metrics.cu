#include "dynamicBANG.h"
#include <set>
#include <cstdio>
#include <cmath>

// ============================================================================
// RECALL CALCULATION (From BANG_Exactdistance)
// ============================================================================

double calculate_recall(unsigned num_queries, unsigned *gold_std,
                       float *gs_dist, unsigned dim_gs,
                       unsigned *our_results, unsigned dim_or,
                       unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        unsigned *gt_vec = gold_std + dim_gs * i;
        unsigned *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;

        if (gs_dist != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec, res_vec + recall_at);

        unsigned cur_recall = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }

    return total_recall / (num_queries) * (100.0 / recall_at);
}

// ============================================================================
// METRICS COMPUTATION
// ============================================================================

void computeRecall(SearchResult* computed, uint32_t* ground_truth,
                   uint32_t num_queries, uint32_t k,
                   PerformanceMetrics* metrics) {
    // Recall@1
    double recall_1 = calculate_recall(num_queries, ground_truth, nullptr,
                                       k, computed->node_ids, k, 1);
    metrics->recall_at_1 = recall_1;

    // Recall@10
    double recall_10 = calculate_recall(num_queries, ground_truth, nullptr,
                                        k, computed->node_ids, k, 10);
    metrics->recall_at_10 = recall_10;

    // Recall@100
    double recall_100 = calculate_recall(num_queries, ground_truth, nullptr,
                                         k, computed->node_ids, k, 100);
    metrics->recall_at_100 = recall_100;
}

void printMetrics(const PerformanceMetrics* metrics) {
    printf("\n");
    printf("================================================================================\n");
    printf("                          PERFORMANCE METRICS                                   \n");
    printf("================================================================================\n\n");

    printf("--- Operation Counts ---\n");
    printf("  Total Inserts:  %lu\n", metrics->total_inserts);
    printf("  Total Deletes:  %lu\n", metrics->total_deletes);
    printf("  Total Queries:  %lu\n", metrics->total_queries);
    printf("  Total Ops:      %lu\n\n",
           metrics->total_inserts + metrics->total_deletes + metrics->total_queries);

    printf("--- Throughput (Ops/Second) ---\n");
    printf("  Insert QPS:     %.2f\n", metrics->insert_qps);
    printf("  Delete QPS:     %.2f\n", metrics->delete_qps);
    printf("  Query QPS:      %.2f\n", metrics->query_qps);
    printf("  Overall:        %.2f\n\n", metrics->overall_throughput);

    printf("--- Latency (milliseconds) ---\n");
    printf("  Insert Avg:     %.3f ms\n", metrics->insert_latency_avg);
    printf("  Delete Avg:     %.3f ms\n", metrics->delete_latency_avg);
    printf("  Query Avg:      %.3f ms\n\n", metrics->query_latency_avg);

    printf("--- Accuracy ---\n");
    printf("  Recall@1:       %.2f%%\n", metrics->recall_at_1);
    printf("  Recall@10:      %.2f%%\n", metrics->recall_at_10);
    printf("  Recall@100:     %.2f%%\n\n", metrics->recall_at_100);

    printf("--- Index Statistics ---\n");
    printf("  Static Size:    %u nodes\n", metrics->static_index_size);
    printf("  Fresh Size:     %u nodes\n", metrics->fresh_index_size);
    printf("  Deleted:        %u nodes\n", metrics->num_deleted);
    printf("  Active Nodes:   %u\n\n",
           metrics->static_index_size + metrics->fresh_index_size - metrics->num_deleted);

    printf("--- Consolidation ---\n");
    printf("  Count:          %u\n", metrics->num_consolidations);
    printf("  Total Time:     %.2f seconds\n", metrics->consolidation_time_total);
    if (metrics->num_consolidations > 0) {
        printf("  Avg Time:       %.2f seconds\n",
               metrics->consolidation_time_total / metrics->num_consolidations);
    }
    printf("\n");

    printf("--- Overall ---\n");
    printf("  Total Time:     %.2f seconds\n", metrics->total_elapsed_time);
    printf("  GPU Memory:     %.2f MB\n", metrics->gpu_memory_used_bytes / (1024.0 * 1024.0));
    printf("  CPU Memory:     %.2f MB\n\n", metrics->cpu_memory_used_bytes / (1024.0 * 1024.0));

    printf("================================================================================\n\n");
}

void saveMetricsToFile(const PerformanceMetrics* metrics, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Could not open %s for writing metrics\n", filename);
        return;
    }

    fprintf(fp, "operation_counts:\n");
    fprintf(fp, "  inserts: %lu\n", metrics->total_inserts);
    fprintf(fp, "  deletes: %lu\n", metrics->total_deletes);
    fprintf(fp, "  queries: %lu\n\n", metrics->total_queries);

    fprintf(fp, "throughput:\n");
    fprintf(fp, "  insert_qps: %.2f\n", metrics->insert_qps);
    fprintf(fp, "  delete_qps: %.2f\n", metrics->delete_qps);
    fprintf(fp, "  query_qps: %.2f\n", metrics->query_qps);
    fprintf(fp, "  overall: %.2f\n\n", metrics->overall_throughput);

    fprintf(fp, "latency_ms:\n");
    fprintf(fp, "  insert_avg: %.3f\n", metrics->insert_latency_avg);
    fprintf(fp, "  delete_avg: %.3f\n", metrics->delete_latency_avg);
    fprintf(fp, "  query_avg: %.3f\n\n", metrics->query_latency_avg);

    fprintf(fp, "accuracy:\n");
    fprintf(fp, "  recall_at_1: %.2f\n", metrics->recall_at_1);
    fprintf(fp, "  recall_at_10: %.2f\n", metrics->recall_at_10);
    fprintf(fp, "  recall_at_100: %.2f\n\n", metrics->recall_at_100);

    fprintf(fp, "index:\n");
    fprintf(fp, "  static_size: %u\n", metrics->static_index_size);
    fprintf(fp, "  fresh_size: %u\n", metrics->fresh_index_size);
    fprintf(fp, "  deleted: %u\n\n", metrics->num_deleted);

    fprintf(fp, "consolidation:\n");
    fprintf(fp, "  count: %u\n", metrics->num_consolidations);
    fprintf(fp, "  total_time: %.2f\n", metrics->consolidation_time_total);
    if (metrics->num_consolidations > 0) {
        fprintf(fp, "  avg_time: %.2f\n",
                metrics->consolidation_time_total / metrics->num_consolidations);
    }
    fprintf(fp, "\n");

    fprintf(fp, "total_time: %.2f\n", metrics->total_elapsed_time);

    fclose(fp);
    printf("[Metrics] Saved to %s\n", filename);
}
