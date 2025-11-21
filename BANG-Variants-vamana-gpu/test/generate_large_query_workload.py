#!/usr/bin/env python3
"""
Generate a large query-only workload for testing batch processing performance
"""
import json
import random

# Generate 50,000 query events
num_queries = 50000
workload = []

for i in range(num_queries):
    event = {
        "timestamp": i * 0.1,  # 10 queries per second
        "type": "QUERY",
        "query_id": i % 100  # Cycle through the 100 available queries
    }
    workload.append(event)

# Write to file
output_file = "test/large_query_workload_50k.jsonl"
with open(output_file, 'w') as f:
    for event in workload:
        f.write(json.dumps(event) + '\n')

print(f"Generated {num_queries} query events in {output_file}")
