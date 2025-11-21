#!/bin/bash

# Benchmark script for testing different Search L values
# Usage: ./benchmark_search_L.sh

echo "=================================="
echo "   Search L Benchmark Suite"
echo "=================================="
echo ""

# Configuration
GRAPH="build/vamana_alpha1.2.out"
QUERIES="data/siftsmall_query.bin"
GROUNDTRUTH="data/siftsmall_groundtruth.bin"
WORKERS=8
BATCH=50
K=10

# Search L values to test
L_VALUES=(10 20 40 100 150)

# Workloads to test
declare -a WORKLOADS=(
    "test/workload_query_only_400.jsonl:Query-Only"
    "test/workload_query_heavy_400.jsonl:Query-Heavy"
    "test/workload_balanced_400.jsonl:Balanced"
    "test/workload_insert_heavy_400.jsonl:Insert-Heavy"
    "test/large_query_workload_50k.jsonl:Large-50K"
)

# Output files
RESULTS_FILE="results_searchL_$(date +%Y%m%d_%H%M%S).txt"
CSV_FILE="results_searchL_$(date +%Y%m%d_%H%M%S).csv"

# CSV Header
echo "Workload,SearchL,Inserts,Deletes,Queries,AvgInsert(ms),AvgDelete(ms),AvgQuery(ms),QPS,Recall@5(%),Recall@10(%),TotalTime(ms),Throughput(ops/s)" > "$CSV_FILE"

echo "Results will be saved to: $RESULTS_FILE"
echo "CSV data will be saved to: $CSV_FILE"
echo ""

# Function to extract value from output
extract_value() {
    local pattern="$1"
    local output="$2"
    echo "$output" | grep "$pattern" | awk '{print $NF}'
}

# Run benchmarks
for workload_entry in "${WORKLOADS[@]}"; do
    IFS=':' read -r workload_file workload_name <<< "$workload_entry"

    if [ ! -f "$workload_file" ]; then
        echo "WARNING: Workload file not found: $workload_file"
        continue
    fi

    echo "========================================" | tee -a "$RESULTS_FILE"
    echo "Workload: $workload_name" | tee -a "$RESULTS_FILE"
    echo "File: $workload_file" | tee -a "$RESULTS_FILE"
    echo "========================================" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    for L in "${L_VALUES[@]}"; do
        echo "Testing Search L = $L..." | tee -a "$RESULTS_FILE"

        # Run the executor
        output=$(./bin/concurrent_fresh_diskann "$GRAPH" "$workload_file" \
                 "$QUERIES" "$GROUNDTRUTH" \
                 --searchL "$L" --k "$K" --workers "$WORKERS" --batch "$BATCH" 2>&1)

        # Check if execution was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: Execution failed for L=$L" | tee -a "$RESULTS_FILE"
            echo "" | tee -a "$RESULTS_FILE"
            continue
        fi

        # Extract statistics
        inserts=$(extract_value "Inserts:" "$output")
        deletes=$(extract_value "Deletes:" "$output")
        queries=$(extract_value "Queries:" "$output")
        avg_insert=$(extract_value "Insert avg:" "$output" | sed 's/ms//')
        avg_delete=$(extract_value "Delete avg:" "$output" | sed 's/ms//')
        avg_query=$(extract_value "Query avg:" "$output" | sed 's/ms//')
        qps=$(extract_value "Query QPS:" "$output")
        recall5=$(extract_value "5-recall@5:" "$output" | sed 's/%//')
        recall10=$(extract_value "10-recall@10:" "$output" | sed 's/%//')
        total_time=$(extract_value "Total execution time:" "$output" | sed 's/ms//')
        throughput=$(extract_value "Throughput:" "$output" | sed 's/ops\/sec//')

        # Print results
        echo "  Inserts: $inserts, Deletes: $deletes, Queries: $queries" | tee -a "$RESULTS_FILE"
        echo "  Avg Insert: ${avg_insert}ms, Avg Delete: ${avg_delete}ms, Avg Query: ${avg_query}ms" | tee -a "$RESULTS_FILE"
        echo "  QPS: $qps, Recall@5: ${recall5}%, Recall@10: ${recall10}%" | tee -a "$RESULTS_FILE"
        echo "  Total Time: ${total_time}ms, Throughput: ${throughput} ops/sec" | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"

        # Write to CSV
        echo "$workload_name,$L,$inserts,$deletes,$queries,$avg_insert,$avg_delete,$avg_query,$qps,$recall5,$recall10,$total_time,$throughput" >> "$CSV_FILE"
    done

    echo "" | tee -a "$RESULTS_FILE"
done

echo "========================================" | tee -a "$RESULTS_FILE"
echo "Benchmark Complete!" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "CSV saved to: $CSV_FILE" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
