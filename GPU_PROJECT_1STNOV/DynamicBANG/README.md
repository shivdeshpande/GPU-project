# DynamicBANG: GPU-Accelerated FreshDiskANN Implementation

## Overview

DynamicBANG implements FreshDiskANN-style dynamic approximate nearest neighbor search on GPU, extending BANG_Exactdistance with support for:
- **Dynamic inserts** to a fresh index
- **Lazy deletions** using GPU bitmaps
- **Dual-index search** (static + fresh indices)
- **Hybrid consolidation** (time + size threshold)
- **Workload replay** from streaming JSONL files

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Static Index   │     │   Fresh Index    │
│  (GPU Memory)   │     │  (GPU Memory)    │
│  - Pre-built    │     │  - Dynamic       │
│  - Read-only    │     │  - Insertions    │
└─────────────────┘     └──────────────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
              ┌───────────────┐
              │ Delete Buffer │
              │  (GPU Bitmap) │
              └───────────────┘
```

### Key Components

1. **Static Index** (`uint8_t* d_pIndex`):
   - Pre-built Vamana graph from base dataset
   - Format: `[vector(D*float)][degree(uint)][neighbors(R*uint)]` per node
   - Loaded from disk, stays in GPU memory
   - Read-only during normal operation

2. **Fresh Index** (`uint8_t* d_pIndex_fresh`):
   - Mutable index for recent insertions
   - Same format as static index
   - Capacity: 10% of static index size
   - Atomic counter tracks current size

3. **Delete Buffer** (`uint32_t* d_bitmap`):
   - Bitmap for lazy deletion (1 bit per node)
   - Set bit = deleted node
   - Checked during search, filtered from results

4. **Dual-Index Search**:
   - Launches greedy search on both indices in parallel
   - Merges results, filters deleted nodes
   - Returns top-K neighbors

5. **Consolidation**:
   - Triggered by hybrid threshold (time OR size)
   - Rebuilds static index with fresh vectors
   - Clears fresh index and delete buffer
   - Atomic swap for consistency

## Implementation Strategy (Following BANG_Exactdistance)

### Search Flow
```
1. neighbor_filtering_new: Filter neighbors using bloom filter
2. compute_neighborDist_par: Compute L2 distances (8 threads/neighbor)
3. compute_BestLSets_par_sort_msort_new: Sort & merge into Best-L set
4. Iterate until convergence (nextIter == false)
5. compute_NearestNeighbours: Final top-K selection
```

### Insert Flow
```
1. Batch insert vectors to h_pIndex_fresh
2. Copy to d_pIndex_fresh at current offset
3. Atomic increment fresh index counter
4. Build graph edges using greedy search against static + existing fresh
5. Update degree and neighbors in fresh index
```

### Delete Flow
```
1. Batch delete IDs
2. Set corresponding bits in d_bitmap using atomic OR
3. No graph modification (lazy)
```

### Consolidation Flow
```
1. Check: fresh_size >= threshold OR elapsed_time >= 60s
2. Allocate new combined index (static + fresh - deleted)
3. Copy active nodes, rebuild Vamana graph
4. Swap pointers atomically
5. Clear fresh index & delete buffer
```

## Files

```
DynamicBANG/
├── README.md                 # This file
├── dynamicBANG.h            # Header with data structures & declarations
├── dynamicBANG.cu           # Main search & orchestration kernels
├── insert.cu                # Insert operation kernels
├── delete.cu                # Delete buffer operations
├── consolidate.cu           # Consolidation logic
├── workload.cu              # Workload loading & processing
├── metrics.cu               # Performance metrics collection
├── main.cu                  # Entry point
├── Makefile                 # Build configuration
├── compile_sift10k.sh       # Compilation script for SIFT10K
├── run_sift10k.sh           # Execution script
└── utils/                   # Copied from BANG_Exactdistance
    ├── utils.h              # CUDA error checking
    └── timer.h              # CPU/GPU timers
```

## Dataset Configuration

Supports multiple datasets (compile-time selection):

| Dataset | Vectors | Dimensions | Type | INDEX_ENTRY_LEN |
|---------|---------|------------|------|-----------------|
| SIFT10K | 10,000 | 128 | float | 772 bytes |
| SIFT1M | 1,000,000 | 128 | float | 772 bytes |
| SIFT100M | 100,000,000 | 128 | uint8_t | 388 bytes |

## Configuration Parameters

```cpp
#define R 64                    // Max node degree
#define L 100                   // Search list size
#define K 100                   // Top-K results
#define BF_ENTRIES 399887U      // Bloom filter size (prime)
#define FRESH_INDEX_CAPACITY (N / 10)  // 10% of static size
#define CONSOLIDATE_TIME_THRESHOLD 60.0f  // 60 seconds
#define CONSOLIDATE_SIZE_THRESHOLD 0.08f  // 8% full
```

## Compilation

```bash
# For SIFT10K (default)
./compile_sift10k.sh

# For SIFT1M
nvcc -DSIFT1M_DATASET -O3 -arch=sm_80 *.cu -o dynamicBANG

# For SIFT100M
nvcc -DSIFT100M_DATASET -O3 -arch=sm_80 *.cu -o dynamicBANG
```

## Usage

```bash
./dynamicBANG \\
  <index_file> \\
  <query_file> \\
  <ground_truth_file> \\
  <workload_jsonl> \\
  <recall_at> \\
  <num_threads>

# Example
./dynamicBANG \\
  ../GPU-project-main/data/sift10k/sift10k_graph.bin \\
  ../GPU-project-main/data/sift10k/siftsmall_query.fvecs \\
  ../GPU-project-main/data/sift10k/siftsmall_groundtruth.ivecs \\
  ../GPU-project-main/workload_ecom_20k.jsonl \\
  100 \\
  64
```

## Workload Format

DynamicBANG processes workloads from JSONL files generated by the workload_generator.py:

```json
{"type":"metadata","scenario":"e_commerce","total_events":20000,...}
{"t":0,"type":"insert","id":10000,"vec":[0.1,0.2,...]}
{"t":1,"type":"query","vec":[0.3,0.4,...]}
{"t":2,"type":"delete","id":10000}
```

Supported scenarios:
- `e_commerce`: 30% insert, 5% delete, 65% query
- `social_burst`: 50% insert, 5% delete, 45% query
- `concurrent_mixed`: 33/33/34 split (stress test)
- All 8 scenarios from GPU-project-main

## Performance Metrics

DynamicBANG tracks:

**Throughput**:
- Insert QPS, Delete QPS, Query QPS
- Overall operations/second

**Latency** (p50, p99):
- Insert, Delete, Query latencies

**Accuracy**:
- Recall@1, Recall@10, Recall@100

**Index Statistics**:
- Static/Fresh index sizes
- Number of consolidations
- Consolidation time
- Deleted node count

**Memory Usage**:
- GPU memory utilization
- CPU memory usage

## Implementation Status

✅ **Completed**:
- Directory structure & build system
- Header files with data structures
- Utils (timer, error checking) from BANG
- Delete buffer implementation
- Index management structures

🚧 **In Progress**:
- Greedy search kernels (dual-index)
- Insert operation with graph building
- Consolidation logic
- Workload processing
- Metrics collection

📋 **Planned**:
- Full testing with SIFT10K
- Optimization for larger datasets
- Performance tuning

## References

1. **BANG_Exactdistance**: GPU ANN search with exact distances
2. **BANG-Variants-vamana-gpu**: GPU Vamana graph construction
3. **FreshDiskANN Paper** (2105.09613v1.pdf): Dynamic ANN indices
4. **Efficient ANN Search with Dynamic Queries.pdf**: Query optimization strategies

## Contact & Contributions

This is a research implementation. For questions or contributions, please refer to the project documentation.
