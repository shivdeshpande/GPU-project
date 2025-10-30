#!/bin/bash

# Compilation script for DynamicBANG with SIFT10K dataset

echo "================================================================"
echo "         Compiling DynamicBANG for SIFT10K Dataset"
echo "================================================================"
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (CUDA compiler) not found"
    echo "Please ensure CUDA toolkit is installed and nvcc is in your PATH"
    exit 1
fi

# Check CUDA version
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""

# Clean previous build
echo "Cleaning previous build..."
make clean
echo ""

# Compile
echo "Compiling DynamicBANG..."
make DATASET=SIFT10K_DATASET ARCH=-arch=sm_80

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "         Compilation Successful!"
    echo "================================================================"
    echo "Executable: ./dynamicBANG"
    echo ""
    echo "To run:"
    echo "./dynamicBANG <index_file> <query_file> <ground_truth> <workload_jsonl> <recall_at> <num_threads>"
    echo ""
    echo "Example:"
    echo "./dynamicBANG \\"
    echo "  ../GPU-project-main/data/sift10k/sift10k_graph.bin \\"
    echo "  ../GPU-project-main/data/sift10k/siftsmall_query.fvecs \\"
    echo "  ../GPU-project-main/data/sift10k/siftsmall_groundtruth.ivecs \\"
    echo "  ../GPU-project-main/workload_ecom_20k.jsonl \\"
    echo "  100 \\"
    echo "  64"
else
    echo ""
    echo "================================================================"
    echo "         Compilation Failed!"
    echo "================================================================"
    echo "Please check the error messages above."
    exit 1
fi
