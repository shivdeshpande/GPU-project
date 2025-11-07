#!/usr/bin/env python3
"""
Generate a mixed workload for FreshDiskANN testing
"""
import json
import random
import numpy as np

def generate_random_vector(dim=128):
    """Generate a random vector"""
    return [float(x) for x in np.random.randn(dim) * 50 + 50]

def generate_workload(num_inserts, num_deletes, num_queries, output_file):
    """
    Generate a mixed workload with inserts, deletes, and queries
    """
    events = []
    timestamp = 0

    # Create event list
    for i in range(num_queries):
        events.append(("query", i))

    for i in range(num_inserts):
        events.append(("insert", 10000 + i))

    for i in range(num_deletes):
        events.append(("delete", i * 10))  # Delete points 0, 10, 20, ...

    # Shuffle to mix operations
    random.shuffle(events)

    # Write to file
    with open(output_file, 'w') as f:
        for event_type, event_id in events:
            if event_type == "query":
                f.write(json.dumps({
                    "type": "query",
                    "timestamp": timestamp,
                    "query_id": event_id
                }) + '\n')
            elif event_type == "insert":
                f.write(json.dumps({
                    "type": "insert",
                    "timestamp": timestamp,
                    "point_id": event_id,
                    "vector": generate_random_vector()
                }) + '\n')
            elif event_type == "delete":
                f.write(json.dumps({
                    "type": "delete",
                    "timestamp": timestamp,
                    "point_id": event_id
                }) + '\n')

            timestamp += 10

    print(f"Generated workload: {output_file}")
    print(f"  Total events: {len(events)}")
    print(f"  Inserts: {num_inserts}")
    print(f"  Deletes: {num_deletes}")
    print(f"  Queries: {num_queries}")

if __name__ == "__main__":
    generate_workload(
        num_inserts=50,
        num_deletes=50,
        num_queries=100,
        output_file="test/large_mixed_workload.jsonl"
    )
