import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

from irkit import (
    IndexEngine,
    ArXivSource,
    HybridRanker,
    BM25Ranker,
    SemanticRanker,
    SentenceTransformerEmbedder,
    InMemoryStorage,
    CrossEncoderRanker,
)
from irkit.core.cache import SemanticCache

QUERIES = [
    "transformer models for natural language processing",
    "quantum computing algorithms and error correction",
    "graph neural network representations in molecular chemistry",
    "observational constraints on dark matter cosmology",
    "gravitational wave detection with laser interferometry",
    "contrastive learning for self-supervised visual representation",
    "stochastic gradient descent optimization with momentum",
    "cosmic microwave background anisotropy analysis",
    "non-euclidean geometry and gauge theories in physics",
    "reinforcement learning policy optimization in robotics",
    "federated learning across heterogeneous edge devices",
    "diffusion probabilistic models for generative image synthesis",
    "attention mechanisms in deep neural networks",
    "black hole thermodynamics and event horizon physics",
    "sparse matrix decomposition for large scale information retrieval",
]

def run_benchmark():
    print("=" * 70)
    print("IRKit Latency Benchmark: Uncached vs. Semantic Vector Cache (P50/P90/P95)")
    print("=" * 70)

    # 1. Initialize Components
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    hybrid = HybridRanker(rankers=[bm25, semantic])
    reranker = CrossEncoderRanker()
    cache = SemanticCache(threshold=0.90)

    engine = IndexEngine(ranker=hybrid, storage=InMemoryStorage(), reranker=reranker, cache=cache)

    print("\n[1/3] Indexing 200 documents from ArXiv...")
    engine.index(ArXivSource(), max_docs=200)

    # Warmup pass
    engine.search("warmup query", rerank=False)

    results_data: Dict[str, Dict] = {}

    # -------------------------------------------------------------
    # Experiment A: Hybrid Retrieval (BM25 + Semantic Dense)
    # -------------------------------------------------------------
    print("\n[2/3] Benchmarking Hybrid Retrieval (BM25 + Semantic)...")
    engine.cache.clear()

    # Uncached runs (cold/unique query execution)
    uncached_hybrid = []
    for q in QUERIES:
        t0 = time.perf_counter()
        engine.search(q, rerank=False)
        uncached_hybrid.append((time.perf_counter() - t0) * 1000)

    # Cached runs (warm hits from semantic vector cache)
    cached_hybrid = []
    for _ in range(5):
        for q in QUERIES:
            t0 = time.perf_counter()
            engine.search(q, rerank=False)
            cached_hybrid.append((time.perf_counter() - t0) * 1000)

    h_p50_uncached = float(np.percentile(uncached_hybrid, 50))
    h_p90_uncached = float(np.percentile(uncached_hybrid, 90))
    h_p95_uncached = float(np.percentile(uncached_hybrid, 95))

    h_p50_cached = float(np.percentile(cached_hybrid, 50))
    h_p90_cached = float(np.percentile(cached_hybrid, 90))
    h_p95_cached = float(np.percentile(cached_hybrid, 95))

    h_speedup_p95 = h_p95_uncached / h_p95_cached

    results_data["hybrid_retrieval"] = {
        "uncached": {"p50_ms": h_p50_uncached, "p90_ms": h_p90_uncached, "p95_ms": h_p95_uncached},
        "cached": {"p50_ms": h_p50_cached, "p90_ms": h_p90_cached, "p95_ms": h_p95_cached},
        "speedup_p95": round(h_speedup_p95, 2),
    }

    # -------------------------------------------------------------
    # Experiment B: Hybrid + Cross-Encoder Reranking Pipeline
    # -------------------------------------------------------------
    print("\n[3/3] Benchmarking Two-Stage Pipeline (Hybrid + Cross-Encoder Rerank)...")
    engine.cache.clear()

    # Warmup reranker
    engine.search("warmup rerank", rerank=True)
    engine.cache.clear()

    uncached_rerank = []
    for q in QUERIES:
        t0 = time.perf_counter()
        engine.search(q, rerank=True)
        uncached_rerank.append((time.perf_counter() - t0) * 1000)

    cached_rerank = []
    for _ in range(5):
        for q in QUERIES:
            t0 = time.perf_counter()
            engine.search(q, rerank=True)
            cached_rerank.append((time.perf_counter() - t0) * 1000)

    r_p50_uncached = float(np.percentile(uncached_rerank, 50))
    r_p90_uncached = float(np.percentile(uncached_rerank, 90))
    r_p95_uncached = float(np.percentile(uncached_rerank, 95))

    r_p50_cached = float(np.percentile(cached_rerank, 50))
    r_p90_cached = float(np.percentile(cached_rerank, 90))
    r_p95_cached = float(np.percentile(cached_rerank, 95))

    r_speedup_p95 = r_p95_uncached / r_p95_cached

    results_data["hybrid_plus_rerank"] = {
        "uncached": {"p50_ms": r_p50_uncached, "p90_ms": r_p90_uncached, "p95_ms": r_p95_uncached},
        "cached": {"p50_ms": r_p50_cached, "p90_ms": r_p90_cached, "p95_ms": r_p95_cached},
        "speedup_p95": round(r_speedup_p95, 2),
    }

    # Save to JSON
    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "cache_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Pipeline Mode':<28} | {'Uncached P95':<14} | {'Cached P95':<12} | {'P95 Speedup':<12}")
    print("-" * 70)
    print(
        f"{'Hybrid (BM25 + Semantic)':<28} | {h_p95_uncached:>10.2f} ms | {h_p95_cached:>8.2f} ms | {h_speedup_p95:>10.1f}x"
    )
    print(
        f"{'Hybrid + Cross-Encoder':<28} | {r_p95_uncached:>10.2f} ms | {r_p95_cached:>8.2f} ms | {r_speedup_p95:>10.1f}x"
    )
    print("=" * 70)
    print(f"Raw telemetry saved to: {json_path}")

if __name__ == "__main__":
    run_benchmark()

