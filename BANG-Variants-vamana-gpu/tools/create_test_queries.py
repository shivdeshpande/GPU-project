#!/usr/bin/env python3
"""
Create test queries from learn dataset (not in base)
and compute brute-force groundtruth
"""

import struct
import numpy as np
import sys

def read_fvecs(filename, max_vectors=None):
    """Read .fvecs file"""
    print(f"Reading {filename}...")
    vectors = []

    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]
            vector = struct.unpack(f'{dim}f', f.read(4 * dim))
            vectors.append(vector)

            if max_vectors and len(vectors) >= max_vectors:
                break

    return np.array(vectors, dtype=np.float32)

def write_bin(filename, data):
    """Write .bin file"""
    print(f"Writing {filename}...")
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', data.shape[0]))
        f.write(struct.pack('I', data.shape[1]))
        f.write(data.tobytes())
    print(f"  {data.shape[0]} vectors, {data.shape[1]} dimensions")

def compute_groundtruth(base, queries, k):
    """Compute brute-force k-NN"""
    print(f"\nComputing brute-force groundtruth (k={k})...")
    num_queries = queries.shape[0]
    num_base = base.shape[0]

    groundtruth = np.zeros((num_queries, k), dtype=np.uint32)

    for i in range(num_queries):
        if (i + 1) % 10 == 0:
            print(f"  Query {i+1}/{num_queries}...")

        # Compute L2 distances to all base points
        diffs = base - queries[i]
        distances = np.sum(diffs * diffs, axis=1)

        # Get top-k nearest neighbors
        top_k_indices = np.argpartition(distances, k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        groundtruth[i] = top_k_indices

    print("Groundtruth computation complete!")
    return groundtruth

def write_groundtruth(filename, gt):
    """Write groundtruth in .bin format"""
    print(f"Writing {filename}...")
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', gt.shape[0]))
        f.write(struct.pack('I', gt.shape[1]))
        f.write(gt.tobytes())
    print(f"  {gt.shape[0]} queries, k={gt.shape[1]}")

def main():
    print("╔════════════════════════════════════════════╗")
    print("║   Test Query & Groundtruth Generator      ║")
    print("╚════════════════════════════════════════════╝\n")

    # Load base dataset
    base_fvecs = "/home/nishat/Documents/GPU_PROG/WORKLOAD_GENERATOR/data/sift10k/siftsmall_base.fvecs"
    base = read_fvecs(base_fvecs)
    print(f"  Loaded {base.shape[0]} base vectors\n")

    # Load learn dataset and take first 100 as queries
    learn_fvecs = "/home/nishat/Documents/GPU_PROG/WORKLOAD_GENERATOR/data/sift10k/siftsmall_learn.fvecs"
    queries = read_fvecs(learn_fvecs, max_vectors=100)
    print(f"  Loaded {queries.shape[0]} query vectors\n")

    # Write query file
    query_output = "data/test_query_external.bin"
    write_bin(query_output, queries)

    # Compute groundtruth
    k = 100
    groundtruth = compute_groundtruth(base, queries, k)

    # Write groundtruth
    gt_output = "data/test_groundtruth_external.bin"
    write_groundtruth(gt_output, groundtruth)

    print(f"\n✓ Success!")
    print(f"  Query file: {query_output}")
    print(f"  Groundtruth file: {gt_output}")
    print(f"\nThese queries are NOT in the base dataset, so recall should be ~97%")

if __name__ == "__main__":
    main()
