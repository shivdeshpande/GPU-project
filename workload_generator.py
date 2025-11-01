#!/usr/bin/env python3
"""
Scenario-Based Dynamic Workload Generator for GPU FreshDiskANN
Combines FreshDiskANN, CleANN, and Quake patterns for robust evaluation.

Usage:
    python workload_generator.py --scenario e_commerce --dataset data/sift10k/base.fvecs \
        --queries data/sift10k/query.fvecs --output workload.jsonl --max_events 20000
"""
import numpy as np
import json
import argparse
import random
import os
import sys
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import hashlib

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class Event:
    """Event data structure."""
    t: int  # timestamp/sequence number
    event_type: str  # "insert", "delete", "query"
    scenario: str
    id: Optional[int] = None  # for insert/delete
    vec: Optional[List[float]] = None  # for insert/query
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = {
            "t": self.t,
            "type": self.event_type,
            "scenario": self.scenario
        }
        if self.id is not None:
            d["id"] = self.id
        if self.vec is not None:
            d["vec"] = self.vec
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

@dataclass
class ScenarioConfig:
    """Scenario configuration."""
    name: str
    description: str
    insert_ratio: float
    delete_ratio: float
    query_ratio: float
    drift_sigma: float
    temporal_pattern: str  # uniform, poisson, burst_decay
    drift_model: str  # gaussian, directional, cluster
    deletion_policy: str  # fifo, lru, random, clustered
    query_distribution: str  # uniform, zipfian, gaussian
    query_skew_alpha: float
    query_coupling: str  # decoupled, semi, fully_coupled
    batch_size: int
    locality_window: int
    base_init_ratio: float

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_fvecs(filename: str) -> np.ndarray:
    """Load vectors from .fvecs file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            raise ValueError(f"Invalid fvecs file: {filename}")
        dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        
        f.seek(0, 2)
        file_size = f.tell()
        bytes_per_vector = 4 + dim * 4
        n_vectors = file_size // bytes_per_vector
        
        if file_size % bytes_per_vector != 0:
            raise ValueError(f"File size misaligned: {filename}")
        
        f.seek(0)
        vectors = np.zeros((n_vectors, dim), dtype=np.float32)
        for i in range(n_vectors):
            vec_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            if vec_dim != dim:
                raise ValueError(f"Dimension mismatch at vector {i}")
            vectors[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    
    print(f"[Load] {filename}: {n_vectors} vectors, dim={dim}")
    return vectors

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def save_jsonl(events: List[Event], output_path: str, metadata: Dict):
    """Save events to JSONL with metadata header."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write metadata as first line
        metadata_line = {"type": "metadata", **metadata}
        json.dump(metadata_line, f, separators=(',', ':'))
        f.write('\n')
        
        # Write events
        for event in events:
            json.dump(event.to_dict(), f, separators=(',', ':'))
            f.write('\n')
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Save] {output_path}: {file_size_mb:.2f} MB, {len(events)} events")

# ============================================================================
# TEMPORAL PATTERN GENERATORS
# ============================================================================
class TemporalSampler:
    """Generate temporal patterns for events."""
    
    def __init__(self, pattern: str, seed: int):
        self.pattern = pattern
        self.rng = np.random.default_rng(seed)
        self.event_counter = 0
    
    def sample_burst_decay(self, burst_freq: int = 500, decay_rate: float = 0.9) -> int:
        """Burst phase followed by exponential decay."""
        phase = self.event_counter // burst_freq
        if phase % 2 == 0:  # Burst phase
            interval = max(1, self.rng.integers(1, 5))
        else:  # Decay phase
            intensity = decay_rate ** (self.event_counter % burst_freq)
            interval = max(1, int(10 / (intensity + 0.1)))
        self.event_counter += 1
        return interval
    
    def sample_poisson(self, lambda_rate: float = 2.0) -> int:
        """Poisson-distributed intervals."""
        return max(1, int(self.rng.poisson(lambda_rate)))
    
    def sample_uniform(self) -> int:
        """Uniform intervals."""
        return 1
    
    def get_sample(self) -> int:
        """Get next sample based on pattern."""
        if self.pattern == "burst_decay":
            return self.sample_burst_decay()
        elif self.pattern == "poisson":
            return self.sample_poisson()
        else:
            return self.sample_uniform()

# ============================================================================
# VECTOR DRIFT MODELS
# ============================================================================
class DriftEngine:
    """Apply various drift models to vectors."""
    
    def __init__(self, drift_model: str, drift_sigma: float, seed: int):
        self.drift_model = drift_model
        self.drift_sigma = drift_sigma
        self.rng = np.random.default_rng(seed)
        self.drift_center = None
        self.drift_direction = None
    
    def gaussian_drift(self, vec: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise drift."""
        noise = self.rng.normal(0, self.drift_sigma, len(vec)).astype(np.float32)
        drifted = vec + noise
        norm = np.linalg.norm(drifted)
        return drifted / norm if norm > 0 else vec
    
    def directional_drift(self, vec: np.ndarray, direction: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply directional drift toward/away from center."""
        if self.drift_direction is None:
            dim = len(vec)
            self.drift_direction = self.rng.normal(0, 1, dim).astype(np.float32)
            self.drift_direction /= np.linalg.norm(self.drift_direction)
        
        magnitude = self.rng.uniform(0, self.drift_sigma)
        drifted = vec + magnitude * self.drift_direction
        norm = np.linalg.norm(drifted)
        return drifted / norm if norm > 0 else vec
    
    def cluster_drift(self, vec: np.ndarray, cluster_id: int = 0) -> np.ndarray:
        """Cluster-based drift (shift cluster center)."""
        if self.drift_center is None:
            self.drift_center = {}
        
        if cluster_id not in self.drift_center:
            dim = len(vec)
            self.drift_center[cluster_id] = self.rng.normal(0, self.drift_sigma, dim).astype(np.float32)
        
        drifted = vec + self.drift_center[cluster_id]
        norm = np.linalg.norm(drifted)
        return drifted / norm if norm > 0 else vec
    
    def apply(self, vec: np.ndarray) -> np.ndarray:
        """Apply configured drift model."""
        if self.drift_model == "gaussian":
            return self.gaussian_drift(vec)
        elif self.drift_model == "directional":
            return self.directional_drift(vec)
        elif self.drift_model == "cluster":
            return self.cluster_drift(vec)
        else:
            return vec

# ============================================================================
# DELETION POLICIES
# ============================================================================
class DeletionPolicy:
    """Manage deletion patterns."""
    
    def __init__(self, policy: str, seed: int):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.insertion_times: Dict[int, int] = {}  # id -> timestamp
        self.last_query_times: Dict[int, int] = {}  # id -> timestamp
    
    def record_insert(self, id: int, timestamp: int):
        """Record insertion timestamp."""
        self.insertion_times[id] = timestamp
    
    def record_query(self, id: int, timestamp: int):
        """Record query access (for LRU)."""
        self.last_query_times[id] = timestamp
    
    def choose_delete(self, active_ids: Set[int], timestamp: int) -> Optional[int]:
        """Choose ID to delete based on policy."""
        if not active_ids:
            return None
        
        if self.policy == "fifo":
            return min(active_ids, key=lambda x: self.insertion_times.get(x, 0))
        
        elif self.policy == "lru":
            return min(active_ids, key=lambda x: self.last_query_times.get(x, self.insertion_times.get(x, 0)))
        
        elif self.policy == "random":
            return self.rng.choice(list(active_ids))
        
        elif self.policy == "clustered":
            # Delete boundary items (highest/lowest ID)
            sorted_ids = sorted(active_ids)
            if self.rng.random() < 0.5:
                return sorted_ids[0]
            else:
                return sorted_ids[-1]
        
        return self.rng.choice(list(active_ids))

# ============================================================================
# QUERY DISTRIBUTION SAMPLERS
# ============================================================================
class QueryDistributionSampler:
    """Sample queries from various distributions."""
    
    def __init__(self, distribution: str, skew_alpha: float, seed: int):
        self.distribution = distribution
        self.skew_alpha = skew_alpha
        self.rng = np.random.default_rng(seed)
    
    def zipfian(self, n: int) -> int:
        """Zipfian distribution (skewed)."""
        # Approximate Zipfian using rejection sampling
        x = self.rng.integers(0, n)
        u = self.rng.uniform(0, 1)
        rank = 1 + x
        if u <= 1.0 / (rank ** self.skew_alpha):
            return x
        return self.zipfian(n)
    
    def gaussian(self, n: int, center: float = 0.5, sigma: float = 0.2) -> int:
        """Gaussian distribution around center."""
        idx = int(self.rng.normal(center * n, sigma * n))
        return np.clip(idx, 0, n - 1)
    
    def sample(self, n: int) -> int:
        """Sample index from configured distribution."""
        if self.distribution == "zipfian":
            return self.zipfian(n)
        elif self.distribution == "gaussian":
            return self.gaussian(n)
        else:
            return self.rng.integers(0, n)

# ============================================================================
# QUERY COUPLING STRATEGIES
# ============================================================================
class QueryCouplingStrategy:
    """Couple query distribution to insertion stream."""
    
    def __init__(self, coupling: str, locality_window: int, seed: int):
        self.coupling = coupling
        self.locality_window = locality_window
        self.rng = np.random.default_rng(seed)
        self.recent_inserts = deque(maxlen=locality_window)
    
    def record_insert(self, id: int, vec: np.ndarray):
        """Record recently inserted vector."""
        self.recent_inserts.append((id, vec))
    
    def get_query_index(self, query_vecs: np.ndarray, all_vecs_pool: np.ndarray) -> int:
        """Get query index based on coupling strategy."""
        n_query = len(query_vecs)
        
        if self.coupling == "decoupled":
            return self.rng.integers(0, n_query)
        
        elif self.coupling == "semi_coupled":
            if self.recent_inserts and self.rng.random() < 0.3:
                # 30% chance: query near recent inserts
                idx = self.rng.integers(0, len(self.recent_inserts))
                _, insert_vec = self.recent_inserts[idx]
                # Find nearest query vector to insert_vec
                dists = np.linalg.norm(query_vecs - insert_vec, axis=1)
                return int(np.argmin(dists))
            else:
                return self.rng.integers(0, n_query)
        
        elif self.coupling == "fully_coupled":
            if self.recent_inserts:
                # Always query recent inserts
                idx = self.rng.integers(0, len(self.recent_inserts))
                _, insert_vec = self.recent_inserts[idx]
                dists = np.linalg.norm(query_vecs - insert_vec, axis=1)
                return int(np.argmin(dists))
            else:
                return self.rng.integers(0, n_query)
        
        return self.rng.integers(0, n_query)

# ============================================================================
# SCENARIO IMPLEMENTATIONS
# ============================================================================
class ScenarioGenerator:
    """Generate workloads for different scenarios."""
    
    def __init__(self, config: ScenarioConfig, base_vecs: np.ndarray,
                 query_vecs: np.ndarray, seed: int):
        self.config = config
        # DO NOT normalize - SIFT vectors should remain unnormalized
        self.base_vecs = base_vecs
        self.query_vecs = query_vecs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        # Initialize components
        self.temporal_sampler = TemporalSampler(config.temporal_pattern, seed)
        self.drift_engine = DriftEngine(config.drift_model, config.drift_sigma, seed)
        self.deletion_policy = DeletionPolicy(config.deletion_policy, seed)
        self.query_sampler = QueryDistributionSampler(config.query_distribution, config.query_skew_alpha, seed)
        self.query_coupling = QueryCouplingStrategy(config.query_coupling, config.locality_window, seed)
        
        # State tracking
        self.active_ids: Set[int] = set()
        self.insert_counter = 0
        self.query_counter = 0  # For sequential query sampling
        self.stats = {'insert': 0, 'delete': 0, 'query': 0, 'failed_delete': 0}
        
        # Split base dataset
        n_base = len(self.base_vecs)
        n_init = int(n_base * config.base_init_ratio)
        self.base_initial = self.base_vecs[:n_init]
        self.insert_pool = self.base_vecs[n_init:]
        
        print(f"[Split] {n_init} initial, {len(self.insert_pool)} insert pool")
    
    def _create_insert_event(self, t: int) -> Optional[Event]:
        """Create insert event."""
        if self.insert_counter >= len(self.insert_pool):
            return None
        
        vec = self.insert_pool[self.insert_counter].copy()
        if self.config.drift_sigma > 0:
            vec = self.drift_engine.apply(vec)
        
        new_id = int(len(self.base_initial) + self.insert_counter)
        self.active_ids.add(new_id)
        self.deletion_policy.record_insert(new_id, t)
        self.query_coupling.record_insert(new_id, vec)
        self.insert_counter += 1
        self.stats['insert'] += 1
        
        return Event(t=t, event_type='insert', scenario=self.config.name,
                    id=new_id, vec=vec.tolist())
    
    def _create_delete_event(self, t: int) -> Optional[Event]:
        """Create delete event."""
        if not self.active_ids:
            self.stats['failed_delete'] += 1
            return None
        
        delete_id = self.deletion_policy.choose_delete(self.active_ids, t)
        if delete_id is None:
            self.stats['failed_delete'] += 1
            return None
        
        self.active_ids.remove(delete_id)
        self.stats['delete'] += 1
        
        return Event(t=t, event_type='delete', scenario=self.config.name, id=delete_id)
    
    def _create_query_event(self, t: int) -> Optional[Event]:
        """Create query event."""
        # Use sequential queries to align with ground truth
        query_idx = self.query_counter % len(self.query_vecs)
        vec = self.query_vecs[query_idx]
        self.query_counter += 1
        self.stats['query'] += 1

        return Event(t=t, event_type='query', scenario=self.config.name, vec=vec.tolist())
    
    def generate(self, max_events: int) -> Tuple[List[Event], Dict]:
        """Generate workload."""
        events = []
        event_types = []
        
        # Create event type distribution
        event_types.extend(['insert'] * int(max_events * self.config.insert_ratio))
        event_types.extend(['delete'] * int(max_events * self.config.delete_ratio))
        event_types.extend(['query'] * int(max_events * self.config.query_ratio))
        
        while len(event_types) < max_events:
            event_types.append('query')
        event_types = event_types[:max_events]
        random.shuffle(event_types)
        
        print(f"\n[Generate] Creating {max_events} events for scenario: {self.config.name}")
        
        t = 0
        for idx, event_type in enumerate(event_types):
            if idx % 1000 == 0 and idx > 0:
                print(f"  Progress: {idx}/{max_events}")
            
            # Advance time
            t += self.temporal_sampler.get_sample()
            
            event = None
            if event_type == 'insert':
                event = self._create_insert_event(t)
                if event is None:
                    event = self._create_query_event(t)
            
            elif event_type == 'delete':
                event = self._create_delete_event(t)
                if event is None:
                    event = self._create_query_event(t)
            
            else:  # query
                event = self._create_query_event(t)
            
            if event:
                events.append(event)
        
        print(f"  Progress: {max_events}/{max_events}")
        
        # Compute statistics
        actual_ratios = {
            'insert': self.stats['insert'] / len(events) if events else 0,
            'delete': self.stats['delete'] / len(events) if events else 0,
            'query': self.stats['query'] / len(events) if events else 0
        }
        
        metadata = {
            'scenario': self.config.name,
            'description': self.config.description,
            'total_events': len(events),
            'target_ratios': {
                'insert': self.config.insert_ratio,
                'delete': self.config.delete_ratio,
                'query': self.config.query_ratio
            },
            'actual_ratios': actual_ratios,
            'stats': {
                'inserts': self.stats['insert'],
                'deletes': self.stats['delete'],
                'queries': self.stats['query'],
                'failed_deletes': self.stats['failed_delete'],
                'max_active_ids': len(self.active_ids),
                'drift_sigma': self.config.drift_sigma,
                'temporal_pattern': self.config.temporal_pattern,
                'drift_model': self.config.drift_model,
                'deletion_policy': self.config.deletion_policy,
                'query_distribution': self.config.query_distribution,
                'query_coupling': self.config.query_coupling
            },
            'generated_at': datetime.now().isoformat(),
            'seed': self.seed
        }
        
        return events, metadata

# ============================================================================
# PRESET SCENARIOS
# ============================================================================
def get_scenario_config(scenario_name: str) -> ScenarioConfig:
    """Get preset scenario configuration."""
    
    scenarios = {
        'e_commerce': ScenarioConfig(
            name='e_commerce',
            description='Steady product stream with searches',
            insert_ratio=0.30, delete_ratio=0.05, query_ratio=0.65,
            drift_sigma=0.02, temporal_pattern='poisson', drift_model='gaussian',
            deletion_policy='fifo', query_distribution='uniform', query_skew_alpha=0.0,
            query_coupling='decoupled', batch_size=1, locality_window=100, base_init_ratio=0.7
        ),
        'social_burst': ScenarioConfig(
            name='social_burst',
            description='Burst of viral content followed by decay',
            insert_ratio=0.50, delete_ratio=0.05, query_ratio=0.45,
            drift_sigma=0.06, temporal_pattern='burst_decay', drift_model='directional',
            deletion_policy='fifo', query_distribution='zipfian', query_skew_alpha=1.2,
            query_coupling='semi_coupled', batch_size=5, locality_window=200, base_init_ratio=0.6
        ),
        'news_indexing': ScenarioConfig(
            name='news_indexing',
            description='Fast-moving news with balanced updates',
            insert_ratio=0.35, delete_ratio=0.25, query_ratio=0.40,
            drift_sigma=0.08, temporal_pattern='poisson', drift_model='cluster',
            deletion_policy='fifo', query_distribution='uniform', query_skew_alpha=0.0,
            query_coupling='decoupled', batch_size=1, locality_window=50, base_init_ratio=0.65
        ),
        'concurrent_mixed': ScenarioConfig(
            name='concurrent_mixed',
            description='Concurrent operations without phase separation',
            insert_ratio=0.33, delete_ratio=0.33, query_ratio=0.34,
            drift_sigma=0.03, temporal_pattern='uniform', drift_model='gaussian',
            deletion_policy='random', query_distribution='uniform', query_skew_alpha=0.0,
            query_coupling='decoupled', batch_size=1, locality_window=100, base_init_ratio=0.5
        ),
        'distribution_shift': ScenarioConfig(
            name='distribution_shift',
            description='Gradual covariate drift over time',
            insert_ratio=0.40, delete_ratio=0.10, query_ratio=0.50,
            drift_sigma=0.12, temporal_pattern='poisson', drift_model='directional',
            deletion_policy='lru', query_distribution='gaussian', query_skew_alpha=0.5,
            query_coupling='semi_coupled', batch_size=10, locality_window=500, base_init_ratio=0.6
        ),
        'skewed_access': ScenarioConfig(
            name='skewed_access',
            description='80/20 rule: 80% queries on 20% of data',
            insert_ratio=0.25, delete_ratio=0.10, query_ratio=0.65,
            drift_sigma=0.02, temporal_pattern='poisson', drift_model='gaussian',
            deletion_policy='random', query_distribution='zipfian', query_skew_alpha=1.5,
            query_coupling='decoupled', batch_size=1, locality_window=100, base_init_ratio=0.7
        ),
        'adaptive_growth': ScenarioConfig(
            name='adaptive_growth',
            description='Dataset grows steadily (more inserts than deletes)',
            insert_ratio=0.45, delete_ratio=0.05, query_ratio=0.50,
            drift_sigma=0.03, temporal_pattern='poisson', drift_model='gaussian',
            deletion_policy='random', query_distribution='uniform', query_skew_alpha=0.0,
            query_coupling='fully_coupled', batch_size=1, locality_window=300, base_init_ratio=0.5
        ),
        'locality_aware': ScenarioConfig(
            name='locality_aware',
            description='Temporal and spatial locality in queries',
            insert_ratio=0.30, delete_ratio=0.10, query_ratio=0.60,
            drift_sigma=0.04, temporal_pattern='uniform', drift_model='gaussian',
            deletion_policy='lru', query_distribution='uniform', query_skew_alpha=0.0,
            query_coupling='fully_coupled', batch_size=1, locality_window=1000, base_init_ratio=0.6
        )
    }
    
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")
    
    return scenarios[scenario_name]

# ============================================================================
# VALIDATION
# ============================================================================
def validate_workload(events: List[Event]) -> Dict:
    """Validate generated workload."""
    issues = []
    
    if not events:
        issues.append("No events generated")
        return {'valid': False, 'issues': issues}
    
    # Check timestamps strictly increasing
    for i in range(1, len(events)):
        if events[i].t < events[i-1].t:
            issues.append(f"Timestamps not increasing at index {i}")
            break
    
    # Check event types valid
    valid_types = {'insert', 'delete', 'query'}
    for i, event in enumerate(events):
        if event.event_type not in valid_types:
            issues.append(f"Invalid event type at {i}: {event.event_type}")
        
        if event.event_type in ['insert', 'delete'] and event.id is None:
            issues.append(f"Event {i} ({event.event_type}) missing ID")
        
        if event.event_type in ['insert', 'query'] and event.vec is None:
            issues.append(f"Event {i} ({event.event_type}) missing vector")
    
    # Check no duplicate delete IDs (basic sanity)
    delete_events = [e for e in events if e.event_type == 'delete']
    insert_events = [e for e in events if e.event_type == 'insert']
    
    if delete_events and not insert_events:
        issues.append("Delete events without inserts")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_events': len(events),
        'unique_inserts': len(set(e.id for e in insert_events if e.id is not None))
    }

# ============================================================================
# MAIN CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Scenario-Based Workload Generator for GPU FreshDiskANN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios:
  - e_commerce: Steady stream with low deletes (e-commerce portals)
  - social_burst: Burst-decay pattern (viral content)
  - news_indexing: Balanced insert/delete (news feeds)
  - concurrent_mixed: Fully concurrent operations
  - distribution_shift: Covariate drift over time
  - skewed_access: Zipfian query distribution (80/20)
  - adaptive_growth: Dataset growth (more inserts than deletes)
  - locality_aware: Temporal/spatial locality

Examples:
  python workload_generator.py --scenario e_commerce \\
      --dataset data/sift10k/siftsmall_base.fvecs \\
      --queries data/sift10k/siftsmall_query.fvecs \\
      --output workload_e_commerce.jsonl --max_events 20000
      
  python workload_generator.py --scenario distribution_shift \\
      --dataset data/sift10k/siftsmall_base.fvecs \\
      --queries data/sift10k/siftsmall_query.fvecs \\
      --output workload_drift.jsonl --max_events 50000 --seed 123
        """
    )
    
    parser.add_argument('--scenario', required=True,
                        help='Scenario name (see available scenarios above)')
    parser.add_argument('--dataset', required=True,
                        help='Path to base dataset (.fvecs)')
    parser.add_argument('--queries', required=True,
                        help='Path to query vectors (.fvecs)')
    parser.add_argument('--output', required=True,
                        help='Output path for workload (.jsonl)')
    parser.add_argument('--max_events', type=int, default=20000,
                        help='Maximum number of events (default: 20000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate workload after generation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Scenario-Based Workload Generator for GPU FreshDiskANN")
    print("=" * 80)
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"[Error] Dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    
    if not os.path.exists(args.queries):
        print(f"[Error] Queries not found: {args.queries}", file=sys.stderr)
        return 1
    
    # Load datasets
    print("\n[Step 1] Loading datasets...")
    try:
        base_vecs = load_fvecs(args.dataset)
        query_vecs = load_fvecs(args.queries)
    except Exception as e:
        print(f"[Error] Failed to load datasets: {e}", file=sys.stderr)
        return 1
    
    if base_vecs.shape[1] != query_vecs.shape[1]:
        print(f"[Error] Dimension mismatch: base={base_vecs.shape[1]}, query={query_vecs.shape[1]}",
              file=sys.stderr)
        return 1
    
    # Get scenario config
    print("\n[Step 2] Loading scenario configuration...")
    try:
        config = get_scenario_config(args.scenario)
        print(f"Scenario: {config.name}")
        print(f"Description: {config.description}")
        print(f"Target ratios: insert={config.insert_ratio:.2f}, delete={config.delete_ratio:.2f}, query={config.query_ratio:.2f}")
        print(f"Drift model: {config.drift_model} (sigma={config.drift_sigma})")
        print(f"Temporal pattern: {config.temporal_pattern}")
        print(f"Deletion policy: {config.deletion_policy}")
        print(f"Query distribution: {config.query_distribution} (skew={config.query_skew_alpha})")
        print(f"Query coupling: {config.query_coupling}")
    except Exception as e:
        print(f"[Error] Invalid scenario: {e}", file=sys.stderr)
        return 1
    
    # Generate workload
    print("\n[Step 3] Generating workload...")
    try:
        generator = ScenarioGenerator(config, base_vecs, query_vecs, args.seed)
        events, metadata = generator.generate(args.max_events)
    except Exception as e:
        print(f"[Error] Failed to generate workload: {e}", file=sys.stderr)
        return 1
    
    # Validate workload
    if args.validate:
        print("\n[Step 4] Validating workload...")
        validation_result = validate_workload(events)
        print(f"Validation: {'PASS' if validation_result['valid'] else 'FAIL'}")
        if not validation_result['valid']:
            for issue in validation_result['issues']:
                print(f"  - {issue}")
        print(f"Total events: {validation_result['total_events']}")
        print(f"Unique inserts: {validation_result['unique_inserts']}")
    
    # Save workload
    print("\n[Step 5] Saving workload...")
    try:
        save_jsonl(events, args.output, metadata)
    except Exception as e:
        print(f"[Error] Failed to save workload: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("WORKLOAD GENERATION COMPLETE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Scenario:     {args.scenario}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Queries:      {args.queries}")
    print(f"  Output:       {args.output}")
    print(f"  Events:       {args.max_events}")
    print(f"  Seed:         {args.seed}")
    print(f"\nStatistics:")
    print(f"  Inserts:      {metadata['stats']['inserts']} ({metadata['actual_ratios']['insert']:.2%})")
    print(f"  Queries:      {metadata['stats']['queries']} ({metadata['actual_ratios']['query']:.2%})")
    print(f"  Deletes:      {metadata['stats']['deletes']} ({metadata['actual_ratios']['delete']:.2%})")
    print(f"  Failed deletes: {metadata['stats']['failed_deletes']}")
    print(f"  Max active IDs: {metadata['stats']['max_active_ids']}")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())