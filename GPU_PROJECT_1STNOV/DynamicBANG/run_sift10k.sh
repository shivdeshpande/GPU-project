#!/bin/bash

# Run script for DynamicBANG with SIFT10K dataset and e-commerce workload

echo "================================================================"
echo "         Running DynamicBANG with SIFT10K"
echo "================================================================"
echo ""

# Configuration
INDEX_FILE="../GPU-project-main/data/sift10k/sift10k_randomgraph.bin"
QUERY_FILE="../GPU-project-main/data/sift10k/siftsmall_query.fvecs"
GT_FILE="../GPU-project-main/data/sift10k/siftsmall_groundtruth.ivecs"
WORKLOAD_FILE="../../workload_200_events.jsonl"
RECALL_AT=100
NUM_THREADS=64

# Check if executable exists
if [ ! -f "./dynamicBANG" ]; then
    echo "Error: dynamicBANG executable not found"
    echo "Please run ./compile_sift10k.sh first"
    exit 1
fi

# Check if data files exist
echo "Checking data files..."
if [ ! -f "$INDEX_FILE" ]; then
    echo "Warning: Index file not found: $INDEX_FILE"
    echo "You may need to build the graph first using BANG-Variants-vamana-gpu"
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "Warning: Query file not found: $QUERY_FILE"
fi

if [ ! -f "$WORKLOAD_FILE" ]; then
    echo "Warning: Workload file not found: $WORKLOAD_FILE"
    echo "You may need to generate it using workload_generator.py"
fi

echo ""
echo "Running with configuration:"
echo "  Index:      $INDEX_FILE"
echo "  Queries:    $QUERY_FILE"
echo "  Truth:      $GT_FILE"
echo "  Workload:   $WORKLOAD_FILE"
echo "  Recall@:    $RECALL_AT"
echo "  Threads:    $NUM_THREADS"
echo ""
echo "================================================================"
echo ""

# Run
./dynamicBANG \
    "$INDEX_FILE" \
    "$QUERY_FILE" \
    "$GT_FILE" \
    "$WORKLOAD_FILE" \
    $RECALL_AT \
    $NUM_THREADS

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "         Run Completed Successfully!"
    echo "================================================================"
    echo ""
    echo "Results saved to: dynamicBANG_metrics.txt"
else
    echo "         Run Failed with Exit Code: $EXIT_CODE"
    echo "================================================================"
fi

exit $EXIT_CODE
