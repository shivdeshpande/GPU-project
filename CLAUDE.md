# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated implementation of FreshDiskANN - a dynamic approximate nearest neighbor search (ANNS) system supporting real-time INSERT, DELETE, QUERY, and CONSOLIDATE operations. Uses α-Relative Neighborhood Graph (α=1.2) to maintain graph connectivity under continuous updates.

## Build Commands

All commands run from `BANG-Variants-vamana-gpu/` directory:

```bash
# Static graph construction
make compile                    # Optimized build
make compile-debug              # Debug build

# Sequential workload executor
make compile-freshdiskann       # Or manually:
nvcc -O3 -rdc=true src/dynamic/fresh_diskann_main.cu src/dynamic/workload.cpp \
  src/dynamic/insert.cu src/dynamic/consolidate.cu src/dynamic/deleteList.cu \
  src/util.cu src/bloomFilter.cu src/greedySearch.cu src/outNeighbors.cu \
  src/reverseEdge.cu -o bin/fresh_diskann

# Concurrent workload executor (parallel INSERT/DELETE/QUERY)
make compile-concurrent         # Requires -std=c++17 -fopenmp -lpthread -lgomp

# Run tests
./bin/fresh_diskann build/vamana_alpha1.2.out test/mixed_50_50_100.jsonl \
  data/siftsmall_query.bin data/siftsmall_groundtruth.bin --searchL 100 --k 10
make run-concurrent             # Run concurrent version with test workload
```

**Compiler Requirements**: CUDA 11.0+, C++17, OpenMP

## Architecture

### Core Algorithm Pipeline

```
GreedySearch (beam search from MEDOID)
    ↓
RobustPrune with α-RNG (select R neighbors, α=1.2 critical for recall)
    ↓
Reverse Edge Update (track nodes pointing to each node)
```

### Key Source Files

**Core Search** (`src/`):
- `vamana.h` - All macro definitions (N, D, R, L, etc.)
- `greedySearch.cu` - Beam search traversal, returns candidate set
- `outNeighbors.cu` - RobustPrune algorithm with α-RNG pruning
- `reverseEdge.cu` - Reverse index for consolidation
- `util.cu` - L2 distance computation, bitonic sort

**Dynamic Operations** (`src/dynamic/`):
- `deleteList.cu` - GPU bitvector for lazy deletion
- `insert.cu` - Point insertion using GreedySearch + RobustPrune
- `consolidate.cu` - Rebuild edges after deletions reach threshold
- `fresh_diskann_main.cu` - Sequential workload executor
- `concurrent_executor.cu` - Parallel execution engine with CUDA streams

**Concurrency Support**:
- `lockfree_graph.cuh` - Version-based seqlock for lock-free graph access
- `lockfree_queue.h` - Lock-free MPMC queue for operation scheduling
- `background_consolidate.cu` - Background consolidation thread

### Configuration (vamana.h)

```c
#define N 10000           // Number of vertices
#define D 128             // Vector dimensionality
#define R 64              // Maximum degree per vertex
#define L 150             // Search list length
#define MEDOID 5000       // Graph entry point
```

Graph entry: 772 bytes per vertex = [vector: 512B][degree: 4B][neighbors: 256B]

### Data Formats

- **Binary graphs**: N vertices × 772 bytes each
- **Workloads**: JSONL format with INSERT/DELETE/QUERY events
- **Vectors**: D-dimensional float arrays

## Key Design Decisions

- **α=1.2 for RobustPrune**: Maintains dense graph connectivity (97-99% recall vs 75-80% with α=1.0)
- **Lazy deletion**: O(1) bitvector marking, batch consolidation when threshold reached
- **Seqlock concurrency**: Odd version = write in progress, even = stable (reader retries on version change)

## Testing

Test workloads in `test/`:
- `mixed_50_50_100.jsonl` - 50 inserts, 50 deletes, 100 queries
- `workload_gen_*.jsonl` - Various test scenarios

Unit tests:
```bash
nvcc -O3 -rdc=true test/test_insert.cu [sources] -o bin/test_insert
```

## Performance

- Query: ~2.6 ms @ L=100 (21.5K QPS, 100% recall)
- Insert: ~9-10 ms per point
- Delete: ~0.02 ms (bitvector marking)
