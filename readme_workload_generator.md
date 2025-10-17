# Scenario-Based Workload Generator for GPU FreshDiskANN

## Overview

This tool generates realistic, scenario-based dynamic workloads for evaluating GPU-accelerated Approximate Nearest Neighbor (ANN) search systems with streaming updates. It combines patterns from three seminal papers:

- **FreshDiskANN** – Real-world streaming scenarios (e-commerce, news feeds, social media)
- **CleANN** – Full dynamism with concurrent operations and distribution shifts
- **Quake** – Adaptive patterns with skewed access distributions

The generator produces robust, reproducible workloads in JSON Lines format compatible with GPU-based ANN evaluation frameworks.

---

## Table of Contents

1. [Core Approach](#core-approach)
2. [Scenario Specifications](#scenario-specifications)
3. [Installation & Usage](#installation--usage)
4. [Output Format](#output-format)
5. [Architecture & Components](#architecture--components)
6. [Configuration](#configuration)
7. [Examples](#examples)

---

## Core Approach

### Design Philosophy

The workload generator is built on **four orthogonal dimensions** that can be independently controlled to create diverse, realistic scenarios:

1. **Temporal Patterns** – How events are spaced in time
2. **Vector Drift Models** – How data evolves and changes
3. **Deletion Policies** – How old items are removed
4. **Query Characteristics** – How queries relate to the dataset

By combining these dimensions, we can simulate real-world dynamics while maintaining reproducibility and control.

### Key Principles

- **Modularity**: Each component (temporal sampler, drift engine, deletion policy, query sampler) is independent and composable
- **Realism**: Patterns inspired by real production systems (e-commerce, social media, news indexing)
- **Robustness**: Comprehensive validation, graceful fallbacks, and detailed statistics
- **Reproducibility**: Deterministic generation with seed control and metadata logging
- **Traceability**: Generated workloads include full configuration and execution metadata

### Workload Composition

Each workload consists of three operation types:

| Operation | Purpose | Ratio Range |
|-----------|---------|-------------|
| **Insert** | Add new vectors to the index | 5%–50% |
| **Delete** | Remove vectors from the index | 5%–33% |
| **Query** | Search for nearest neighbors | 34%–65% |

The ratios are configurable per scenario and validated against actual event generation.

---

## Scenario Specifications

### Scenario 1: E-Commerce (`e_commerce`)

**Real-world use case:** Product search in e-commerce platforms

**Characteristics:**
- Steady stream of product arrivals (new inventory)
- Very few product removals (only stale/discontinued items)
- High query rate (customers constantly searching)
- Minimal temporal locality (searches are independent)
- Low data drift (new products similar to existing ones)

**Configuration:**
```
Insert ratio:        30%  (steady new products)
Delete ratio:        5%   (rare removals)
Query ratio:         65%  (active search)
Drift sigma:         0.02 (low drift)
Temporal pattern:    Poisson (realistic arrival)
Drift model:         Gaussian (minor variations)
Deletion policy:     FIFO (oldest products first)
Query distribution:  Uniform (all products equally likely)
Query coupling:      Decoupled (independent searches)
Locality window:     100
```

**Typical workload:**
- 20,000–50,000 events
- Steady-state performance testing
- Index growth evaluation

---

### Scenario 2: Social Media (`social_burst`)

**Real-world use case:** Trending content on social media platforms

**Characteristics:**
- Viral content creates sudden bursts of insertions
- Queries spike during burst phase, decay afterward
- Few deletions (content persists indefinitely)
- Strong temporal hotspots (trendy content queries fade quickly)
- Moderate data drift (new trends different from old trends)

**Configuration:**
```
Insert ratio:        50%  (burst insertions)
Delete ratio:        5%   (minimal removal)
Query ratio:         45%  (concentrated queries)
Drift sigma:         0.06 (moderate drift)
Temporal pattern:    Burst-Decay (two-phase: spike → decay)
Drift model:         Directional (topics evolve along direction)
Deletion policy:     FIFO (chronological removal)
Query distribution:  Zipfian (20% content gets 80% queries)
Query coupling:      Semi-Coupled (30% queries target recent)
Locality window:     200
```

**Typical workload:**
- 20,000–50,000 events
- Spike handling and recovery testing
- Temporal locality evaluation

---

### Scenario 3: News Indexing (`news_indexing`)

**Real-world use case:** News search engines with continuously evolving content

**Characteristics:**
- Balanced insert/delete rate (old news removed as new arrives)
- Moderate query rate (users search fresh content)
- Fast-moving dataset (topics evolve quickly)
- High data drift (semantic meaning of queries changes)
- FIFO deletion ensures freshness

**Configuration:**
```
Insert ratio:        35%  (steady new articles)
Delete ratio:        25%  (aggressive old removal)
Query ratio:         40%  (news searches)
Drift sigma:         0.08 (high drift)
Temporal pattern:    Poisson (realistic arrivals)
Drift model:         Cluster (news clusters shift)
Deletion policy:     FIFO (oldest news first)
Query distribution:  Uniform (diverse search interests)
Query coupling:      Decoupled (independent queries)
Locality window:     50
```

**Typical workload:**
- 20,000–50,000 events
- Index stability under high churn
- Drift handling evaluation

---

### Scenario 4: Concurrent Mixed (`concurrent_mixed`)

**Real-world use case:** Stress test for full dynamism (inspired by CleANN)

**Characteristics:**
- Fully interleaved operations (no phase separation)
- Equal ratios of inserts, deletes, and queries
- Tests system's ability to handle concurrent updates
- Minimal constraints on operation ordering
- Random deletion policy (unpredictable removal pattern)

**Configuration:**
```
Insert ratio:        33%  (equal operations)
Delete ratio:        33%  (equal operations)
Query ratio:         34%  (equal operations)
Drift sigma:         0.03 (low drift)
Temporal pattern:    Uniform (constant rate)
Drift model:         Gaussian (mild variations)
Deletion policy:     Random (unpredictable)
Query distribution:  Uniform (unbiased)
Query coupling:      Decoupled (independent)
Locality window:     100
```

**Typical workload:**
- 20,000–50,000 events
- Concurrency and contention testing
- Update isolation and correctness verification

---

### Scenario 5: Distribution Shift (`distribution_shift`)

**Real-world use case:** System with gradual covariate drift (inspired by CleANN)

**Characteristics:**
- Data distribution shifts gradually over time
- Queries follow evolved distribution with lag
- Deletes remove "old" semantic region
- Inserts populate "new" semantic region
- Tests index quality degradation and adaptation

**Configuration:**
```
Insert ratio:        40%  (new region population)
Delete ratio:        10%  (selective old removal)
Query ratio:         50%  (evolved queries)
Drift sigma:         0.12 (high drift)
Temporal pattern:    Poisson (realistic)
Drift model:         Directional (systematic shift)
Deletion policy:     LRU (recently used retained)
Query distribution:  Gaussian (centered around new region)
Query coupling:      Semi-Coupled (30% locality)
Locality window:     500
Batch size:          10
```

**Typical workload:**
- 50,000–100,000 events
- Long-running drift evaluation
- Recall degradation analysis

---

### Scenario 6: Skewed Access (`skewed_access`)

**Real-world use case:** Power-law query distribution (inspired by Quake)

**Characteristics:**
- 80% of queries target only 20% of data (Zipfian)
- Insert/delete ratios less skewed
- Tests adaptive graph partitioning
- Hotspot clustering in semantic space
- Exercises cache coherency

**Configuration:**
```
Insert ratio:        25%  (steady additions)
Delete ratio:        10%  (normal removal)
Query ratio:         65%  (high query load)
Drift sigma:         0.02 (stable data)
Temporal pattern:    Poisson (realistic)
Drift model:         Gaussian (minor variations)
Deletion policy:     Random (uniform removal)
Query distribution:  Zipfian (skew parameter α=1.5)
Query coupling:      Decoupled (independent)
Locality window:     100
```

**Typical workload:**
- 50,000–100,000 events
- Hotspot handling evaluation
- Cache effectiveness measurement

---

### Scenario 7: Adaptive Growth (`adaptive_growth`)

**Real-world use case:** Growing dataset with evolving queries (inspired by Quake)

**Characteristics:**
- Dataset grows over time (net positive insertions)
- Queries adapt to larger dataset
- K-NN search expands as dataset grows
- Insert pool exhaustion gracefully handled
- Tests scalability under growth

**Configuration:**
```
Insert ratio:        45%  (aggressive growth)
Delete ratio:        5%   (minimal removal)
Query ratio:         50%  (adaptive queries)
Drift sigma:         0.03 (low drift)
Temporal pattern:    Poisson (realistic)
Drift model:         Gaussian (stable)
Deletion policy:     Random (unbiased)
Query distribution:  Uniform (all regions searched)
Query coupling:      Fully-Coupled (target recent)
Locality window:     300
Growth curve:        Linear (constant growth rate)
```

**Typical workload:**
- 100,000–200,000 events
- Index growth performance
- Memory scaling evaluation

---

### Scenario 8: Locality-Aware (`locality_aware`)

**Real-world use case:** Application with strong temporal/spatial locality

**Characteristics:**
- Recent insertions queried soon after
- Queries semantically close to inserted vectors
- Tests neighborhood quality maintenance
- Exercises cache locality
- LRU deletion retains recently accessed items

**Configuration:**
```
Insert ratio:        30%  (steady stream)
Delete ratio:        10%  (selective removal)
Query ratio:         60%  (high locality)
Drift sigma:         0.04 (moderate drift)
Temporal pattern:    Uniform (constant rate)
Drift model:         Gaussian (minor noise)
Deletion policy:     LRU (recently used retained)
Query distribution:  Uniform (within locality)
Query coupling:      Fully-Coupled (always local)
Locality window:     1000 (large window)
```

**Typical workload:**
- 50,000–100,000 events
- Locality-based performance
- Cache effectiveness testing

---

## Installation & Usage

### Prerequisites

```bash
Python 3.8+
numpy
argparse (built-in)
json (built-in)
```

### Basic Usage

```bash
python workload_generator.py \
    --scenario <scenario_name> \
    --dataset <path_to_base.fvecs> \
    --queries <path_to_query.fvecs> \
    --output <output.jsonl> \
    --max_events <num_events> \
    --seed <seed> \
    --validate
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--scenario` | str | ✓ | — | Scenario name (see list below) |
| `--dataset` | str | ✓ | — | Path to base dataset (.fvecs) |
| `--queries` | str | ✓ | — | Path to query vectors (.fvecs) |
| `--output` | str | ✓ | — | Output path for workload (.jsonl) |
| `--max_events` | int | ✗ | 20000 | Maximum number of events |
| `--seed` | int | ✗ | 42 | Random seed for reproducibility |
| `--validate` | flag | ✗ | False | Enable post-generation validation |

### Available Scenarios

```
e_commerce        - E-commerce product search
social_burst      - Social media viral content
news_indexing     - News search with fast churn
concurrent_mixed  - Fully concurrent operations
distribution_shift - Covariate drift over time
skewed_access     - Power-law (80/20) queries
adaptive_growth   - Growing dataset
locality_aware    - Temporal/spatial locality
```

---

## Output Format

### JSONL Structure

```
{"type":"metadata","scenario":"e_commerce","total_events":20000,...}
{"t":0,"type":"insert","scenario":"e_commerce","id":7000,"vec":[...]}
{"t":1,"type":"query","scenario":"e_commerce","vec":[...]}
{"t":2,"type":"delete","scenario":"e_commerce","id":7000}
...
```

### Metadata Header (Line 1)

```json
{
  "type": "metadata",
  "scenario": "e_commerce",
  "description": "Steady product stream with searches",
  "total_events": 20000,
  "target_ratios": {
    "insert": 0.30,
    "delete": 0.05,
    "query": 0.65
  },
  "actual_ratios": {
    "insert": 0.299,
    "delete": 0.051,
    "query": 0.650
  },
  "stats": {
    "inserts": 5980,
    "deletes": 1020,
    "queries": 13000,
    "failed_deletes": 0,
    "max_active_ids": 4960,
    "drift_sigma": 0.02,
    "temporal_pattern": "poisson",
    "drift_model": "gaussian",
    "deletion_policy": "fifo",
    "query_distribution": "uniform",
    "query_coupling": "decoupled"
  },
  "generated_at": "2025-10-17T14:30:00.123456",
  "seed": 42
}
```

### Event Fields

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `t` | int | Always | Event timestamp/sequence number |
| `type` | str | Always | Event type: `insert`, `delete`, `query` |
| `scenario` | str | Always | Scenario name |
| `id` | int | Insert, Delete | Vector ID (for active set tracking) |
| `vec` | float[] | Insert, Query | Vector data (float32 normalized) |

---

## Architecture & Components

### 1. Temporal Samplers (`TemporalSampler`)

Controls spacing between events:

- **Uniform**: Constant interval (1 time unit)
- **Poisson**: Realistic streaming with variable intervals
- **Burst-Decay**: Two-phase pattern (spike → exponential decay)

### 2. Drift Engines (`DriftEngine`)

Applies evolution to inserted vectors:

- **Gaussian**: Random noise addition (∼N(0, σ²))
- **Directional**: Systematic movement along direction vector
- **Cluster**: Cluster centers shift, members follow

### 3. Deletion Policies (`DeletionPolicy`)

Determines which ID to delete:

- **FIFO**: Oldest insertion timestamp first
- **LRU**: Least recently used (tracked via query times)
- **Random**: Uniformly random from active set
- **Clustered**: Delete boundary items (cluster edges)

### 4. Query Samplers (`QueryDistributionSampler`)

Selects queries from various distributions:

- **Uniform**: Random index in [0, n_query)
- **Zipfian**: Power-law with skew parameter α (80/20 rule)
- **Gaussian**: Centered around middle of distribution

### 5. Query Coupling (`QueryCouplingStrategy`)

Relates queries to insertion stream:

- **Decoupled**: Queries independent of insertions
- **Semi-Coupled**: 30% queries target recent inserts
- **Fully-Coupled**: All queries target recent inserts (locality window)

### 6. Scenario Generator (`ScenarioGenerator`)

Orchestrates all components:

1. Splits base dataset into initial and insert pool
2. Normalizes all vectors (L2)
3. Generates events according to scenario config
4. Tracks active IDs for valid delete operations
5. Validates event sequence
6. Collects comprehensive statistics

### 7. State Tracking

- **active_ids**: Set of currently inserted vector IDs
- **insertion_times**: Timestamp of each insertion (for FIFO)
- **last_query_times**: Last access time (for LRU)
- **recent_inserts**: Circular buffer of recent inserts (for locality coupling)
- **stats**: Counters for insert/delete/query/failed operations

---

## Configuration

### Adding Custom Scenarios

To add a new scenario, define a `ScenarioConfig` and register it:

```python
def get_scenario_config(scenario_name: str) -> ScenarioConfig:
    scenarios = {
        'my_scenario': ScenarioConfig(
            name='my_scenario',
            description='Custom scenario description',
            insert_ratio=0.30,
            delete_ratio=0.10,
            query_ratio=0.60,
            drift_sigma=0.05,
            temporal_pattern='poisson',
            drift_model='gaussian',
            deletion_policy='fifo',
            query_distribution='uniform',
            query_skew_alpha=0.0,
            query_coupling='semi_coupled',
            batch_size=1,
            locality_window=200,
            base_init_ratio=0.7
        ),
        # ... existing scenarios
    }
```

### Tuning Parameters

| Parameter | Effect | Range |
|-----------|--------|-------|
| `insert_ratio` | Insertion frequency | 0.05–0.50 |
| `delete_ratio` | Deletion frequency | 0.05–0.33 |
| `query_ratio` | Query frequency | 0.34–0.65 |
| `drift_sigma` | Vector evolution magnitude | 0.0–0.20 |
| `query_skew_alpha` | Zipfian skewness | 0.0–2.0 |
| `locality_window` | Recent insert tracking size | 50–5000 |
| `base_init_ratio` | Initial dataset fraction | 0.5–0.9 |

---

## Examples

### Example 1: E-Commerce Workload

```bash
python workload_generator.py \
    --scenario e_commerce \
    --dataset data/sift10k/siftsmall_base.fvecs \
    --queries data/sift10k/siftsmall_query.fvecs \
    --output workload_ecom_20k.jsonl \
    --max_events 20000 \
    --seed 42 \
    --validate
```

**Output:**
```
[Load] data/sift10k/siftsmall_base.fvecs: 10000 vectors, dim=128
[Load] data/sift10k/siftsmall_query.fvecs: 100 vectors, dim=128
[Split] 7000 initial, 3000 insert pool
[Generate] Creating 20000 events for scenario: e_commerce
  Progress: 1000/20000
  Progress: 2000/20000
  ...
[Save] workload_ecom_20k.jsonl: 5.2 MB, 20000 events

WORKLOAD GENERATION COMPLETE
Statistics:
  Inserts:      5980 (29.90%)
  Queries:      13000 (65.00%)
  Deletes:      1020 (5.10%)
  Failed deletes: 0
  Max active IDs: 4960
```

### Example 2: Distribution Shift Scenario

```bash
python workload_generator.py \
    --scenario distribution_shift \
    --dataset data/sift10k/siftsmall_base.fvecs \
    --queries data/sift10k/siftsmall_query.fvecs \
    --output workload_drift_100k.jsonl \
    --max_events 100000 \
    --seed 123 \
    --validate
```

### Example 3: Stress Test (Skewed + High Volume)

```bash
python workload_generator.py \
    --scenario skewed_access \
    --dataset data/sift10k/siftsmall_base.fvecs \
    --queries data/sift10k/siftsmall_query.fvecs \
    --output workload_stress_100k.jsonl \
    --max_events 100000 \
    --seed 999
```

---

## Validation & Quality Assurance

The generator includes comprehensive validation:

### Invariant Checks

- Timestamps strictly increasing
- Valid event types (insert/delete/query)
- All insert/delete events have IDs
- All insert/query events have vectors
- No delete of non-existent IDs

### Statistical Validation

- Actual vs. target ratios reconciliation
- Drift sigma distribution verification
- Temporal pattern distribution matching
- Zipfian skewness measurement
- Active ID count tracking

### Graceful Fallbacks

- Insert pool exhaustion → convert to queries
- Empty active set (no deletes available) → convert to queries
- Invalid configuration → descriptive error message

---

## Performance Considerations

### Memory Efficiency

- Streaming JSON output (no full buffer)
- Efficient active ID tracking (Python set)
- Lazy metadata computation
- Vector normalization in-place where possible

### Generation Speed

For SIFT10K (10,000 base, 100 query):

| Workload Size | Time | Memory |
|---------------|------|--------|
| 20K events | ~2–3s | ~100 MB |
| 50K events | ~5–7s | ~150 MB |
| 100K events | ~10–15s | ~200 MB |

---

## Extending the Generator

### Adding a New Temporal Pattern

```python
class TemporalSampler:
    def sample_custom_pattern(self) -> int:
        # Your logic here
        return interval
    
    def get_sample(self) -> int:
        if self.pattern == "custom":
            return self.sample_custom_pattern()
        # ... existing patterns
```

### Adding a New Drift Model

```python
class DriftEngine:
    def my_drift_model(self, vec: np.ndarray) -> np.ndarray:
        # Your drift logic
        drifted = vec + transformation
        return drifted / np.linalg.norm(drifted)
```

---



## References

- **FreshDiskANN**: A. Singh et al., "FreshDiskANN: A fast and accurate graph-based ANN index for streaming similarity search," arXiv:2105.09613
- **CleANN**: [Efficient Full Dynamism in Graph-based ANN Search]
- **Quake**: [Adaptive Indexing for Vector Search]
- **DiskANN**: S. Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node," NeurIPS 2019

---

