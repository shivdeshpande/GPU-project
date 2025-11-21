#!/usr/bin/env python3
import json
import random

def create_workload(name, total_ops, query_pct, insert_pct, delete_pct):
    """Create a simple workload with specified mix"""
    num_queries = int(total_ops * query_pct / 100)
    num_inserts = int(total_ops * insert_pct / 100)
    num_deletes = total_ops - num_queries - num_inserts
    
    events = []
    next_point_id = 10001
    
    # Generate events
    for i in range(num_queries):
        events.append({"type": "query", "timestamp": i * 0.01, "query_id": i % 100})
    
    for i in range(num_inserts):
        vector = [random.uniform(-50, 200) for _ in range(128)]
        events.append({"type": "insert", "timestamp": 0, "point_id": next_point_id + i, "vector": vector})
    
    for i in range(num_deletes):
        events.append({"type": "delete", "timestamp": 0, "point_id": random.randint(0, 9999)})
    
    # Shuffle and re-timestamp
    random.shuffle(events)
    for i, event in enumerate(events):
        event["timestamp"] = i * 0.01
    
    # Write to file
    output_file = f"test/workload_{name}.jsonl"
    with open(output_file, 'w') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    print(f"✓ {output_file}: {total_ops} ops ({num_queries}Q/{num_inserts}I/{num_deletes}D) = {query_pct}%/{insert_pct}%/{delete_pct}%")

# Create workload scenarios
print("Creating custom workloads...\n")

create_workload("query_heavy_400", 400, 95, 3, 2)       # 95% queries - for high throughput
create_workload("balanced_400", 400, 60, 25, 15)        # Balanced mix
create_workload("insert_heavy_400", 400, 40, 50, 10)    # 50% inserts - show bottleneck
create_workload("query_only_400", 400, 100, 0, 0)       # Pure queries - max throughput

print("\n✓ All workloads created!")
