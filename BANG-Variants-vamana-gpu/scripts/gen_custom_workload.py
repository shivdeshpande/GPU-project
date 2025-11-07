#!/usr/bin/env python3
"""
Generate a custom mixed workload for FreshDiskANN
50 inserts, 50 deletes, 100 queries - randomly interleaved
"""
import json
import random
import numpy as np

# Configuration
NUM_INSERTS = 50
NUM_DELETES = 50
NUM_QUERIES = 100
DIM = 128

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_random_vector(dim=128):
    """Generate a random D-dimensional vector"""
    return [float(x) for x in (np.random.randn(dim) * 50 + 50)]

# Create events
events = []

# Add 100 queries (referencing query IDs 0-99 from the loaded query file)
for i in range(NUM_QUERIES):
    events.append(("query", i))

# Add 50 inserts (new point IDs starting from 10000)
for i in range(NUM_INSERTS):
    events.append(("insert", 10000 + i, generate_random_vector(DIM)))

# Add 50 deletes (delete existing points 10, 20, 30, ... 500)
for i in range(NUM_DELETES):
    events.append(("delete", 10 + i * 10))

# Shuffle to create realistic interleaving
random.shuffle(events)

# Write to JSONL file
output_file = "test/mixed_50_50_100.jsonl"
timestamp = 0

with open(output_file, 'w') as f:
    for event in events:
        event_type = event[0]

        if event_type == "query":
            query_id = event[1]
            f.write(json.dumps({
                "type": "query",
                "timestamp": timestamp,
                "query_id": query_id
            }) + '\n')

        elif event_type == "insert":
            point_id = event[1]
            vector = event[2]
            f.write(json.dumps({
                "type": "insert",
                "timestamp": timestamp,
                "point_id": point_id,
                "vector": vector
            }) + '\n')

        elif event_type == "delete":
            point_id = event[1]
            f.write(json.dumps({
                "type": "delete",
                "timestamp": timestamp,
                "point_id": point_id
            }) + '\n')

        timestamp += 10

print(f"✓ Generated workload: {output_file}")
print(f"  Total events: {len(events)}")
print(f"  Inserts:  {NUM_INSERTS}")
print(f"  Deletes:  {NUM_DELETES}")
print(f"  Queries:  {NUM_QUERIES}")
print(f"  Delete ratio: {NUM_DELETES / len(events) * 100:.1f}%")
