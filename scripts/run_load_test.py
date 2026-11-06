import asyncio
import json
import time
from pathlib import Path
import numpy as np
import httpx

from irkit import IndexEngine, ArXivSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage
from irkit.core.cache import SemanticCache
from irkit.serve.api import app, set_engine

QUERIES = [
    "transformer architecture",
    "deep learning optimization",
    "quantum computing algorithms",
    "information retrieval benchmarks",
    "consistent hashing Redis",
    "dense vector embeddings",
    "BM25 keyword search",
    "reciprocal rank fusion",
    "fastapi async endpoints",
    "cross encoder reranking",
]

async def benchmark_api(concurrency: int = 20, total_requests: int = 400):
    print("=" * 70)
    print("IRKit HTTP Serving Load Test (Concurrent Client Simulation)")
    print("=" * 70)

    print("\n[1/3] Initializing and indexing in-memory IRKit engine...")
    embedder = SentenceTransformerEmbedder()
    ranker = HybridRanker(rankers=[BM25Ranker(), SemanticRanker(embedder)])
    cache = SemanticCache(threshold=0.90)
    engine = IndexEngine(ranker=ranker, storage=InMemoryStorage(), cache=cache)
    engine.index(ArXivSource(), max_docs=200)
    set_engine(engine)

    # Warmup
    engine.search("warmup")

    print(f"\n[2/3] Simulating {total_requests} requests with concurrency={concurrency}...")

    latencies = []
    transport = httpx.ASGITransport(app=app)

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async def fetch(idx: int):
            q = QUERIES[idx % len(QUERIES)]
            async with sem:
                t0 = time.perf_counter()
                resp = await client.get(f"/search?q={q}&top_k=10&rerank=false")
                lat = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    latencies.append(lat)
                else:
                    print(f"Error {resp.status_code}: {resp.text}")

        t_start = time.perf_counter()
        tasks = [fetch(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    qps = len(latencies) / total_time
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    report = {
        "total_requests": len(latencies),
        "concurrency": concurrency,
        "total_time_seconds": round(total_time, 2),
        "throughput_qps": round(qps, 1),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
    }

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "load_test_results.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("LOAD TEST TELEMETRY REPORT")
    print("=" * 70)
    print(f"Total Requests Completed: {len(latencies)} / {total_requests}")
    print(f"Throughput (QPS):         {qps:.1f} req/s")
    print(f"Latency P50:              {p50:.2f} ms")
    print(f"Latency P90:              {p90:.2f} ms")
    print(f"Latency P95:              {p95:.2f} ms")
    print(f"Latency P99:              {p99:.2f} ms")
    print("=" * 70)
    print(f"Saved results to: {report_file}")

if __name__ == "__main__":
    asyncio.run(benchmark_api(concurrency=20, total_requests=400))

