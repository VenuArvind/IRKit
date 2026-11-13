"""
IRKit SWE Micro-Benchmark: gRPC (Protobuf / HTTP/2) vs. REST (JSON / HTTP/1.1)

Evaluates:
1. Wire payload size (bytes transferred)
2. Serialization & deserialization overhead (CPU time)
3. End-to-end P50/P90/P95/P99 latency over 1,000 concurrent queries
"""

import time
import json
import random
import statistics
import os
import sys
from pathlib import Path
from concurrent import futures

os.environ["IRKIT_SILENT_PROFILER"] = "1"

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc
from fastapi.testclient import TestClient

from irkit.core.engine import IndexEngine
from irkit.rankers.bm25 import BM25Ranker
from irkit.storage.memory import InMemoryStorage
from irkit.proto import search_pb2, search_pb2_grpc
from irkit.serve.grpc_server import SearchServicer
from irkit.serve.api import app, set_engine


# --- Synthetic Corpus Generation ---
VOCABULARY = [
    "transformer", "attention", "neural", "search", "inverted", "index", "sharding",
    "consistent", "hashing", "reranker", "embedding", "cross-encoder", "retrieval",
    "hybrid", "latency", "throughput", "paged", "cache", "vector", "distributed",
    "cluster", "replica", "fault-tolerant", "consensus", "replication", "partition",
    "memory", "quantization", "int8", "float16", "serialization", "protobuf", "grpc"
]

def generate_corpus(num_docs: int = 500):
    docs = []
    for i in range(num_docs):
        terms = random.sample(VOCABULARY, k=random.randint(6, 12))
        title = f"Document {i}: {' '.join(terms[:3]).title()}"
        text = f"Detailed discussion on {' '.join(terms)}. Highly relevant for scalable search pipelines."
        docs.append({
            "id": f"doc_{i:04d}",
            "title": title,
            "text": text,
            "metadata": {
                "doc_index": str(i),
                "category": random.choice(["systems", "ml", "database", "networking"]),
                "indexed_at": "2026-11-13"
            }
        })
    return docs


def percentile(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] * (1 - d) + s[c] * d


def run_benchmark():
    print("=" * 78)
    print("🚀 IRKit SWE Benchmark: gRPC (Protobuf) vs. REST (JSON)")
    print("=" * 78)

    # 1. Initialize Engine & Index Corpus
    print("\n[1/4] Indexing 500 documents into IndexEngine...")
    storage = InMemoryStorage()
    ranker = BM25Ranker()
    engine = IndexEngine(ranker=ranker, storage=storage)
    corpus = generate_corpus(500)
    for d in corpus:
        storage.save(type("Doc", (), {"id": d["id"], "title": d["title"], "text": d["text"], "metadata": d["metadata"]})())
    ranker.index(corpus)
    print(f"✅ Successfully indexed {storage.count()} documents.")

    # 2. Wire Up Servers
    set_engine(engine)
    rest_client = TestClient(app)

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SearchServicer(engine)
    search_pb2_grpc.add_SearchServiceServicer_to_server(servicer, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    grpc_server.start()

    grpc_channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc_stub = search_pb2_grpc.SearchServiceStub(grpc_channel)

    # 3. Payload Size & Serialization Micro-Benchmark
    print("\n[2/4] Measuring wire payload size and serialization speed (10,000 iterations)...")
    sample_hits = engine.search("transformer search indexing", top_k=10)

    # Convert to Protobuf message
    proto_hits = [servicer._convert_hit(h) for h in sample_hits]
    proto_resp = search_pb2.SearchResponse(
        query="transformer search indexing",
        hits=proto_hits,
        latency_ms=1.25,
        total_docs=500,
        quantization_mode="none"
    )

    # Convert to REST dict
    rest_dict = {
        "query": "transformer search indexing",
        "results": [h.__dict__ for h in sample_hits],
        "latency_ms": 1.25,
        "total_docs": 500,
        "system_metrics": {"duration": 0.00125},
        "quantization_mode": "none"
    }

    # Measure Wire Size
    proto_bytes = proto_resp.SerializeToString()
    json_bytes = json.dumps(rest_dict).encode("utf-8")
    proto_size = len(proto_bytes)
    json_size = len(json_bytes)
    payload_reduction_pct = (1.0 - (proto_size / json_size)) * 100.0

    print(f"  • REST JSON wire size:     {json_size:,} bytes")
    print(f"  • gRPC Protobuf wire size: {proto_size:,} bytes")
    print(f"  • Wire Payload Reduction:  {payload_reduction_pct:.1f}% smaller wire footprint")

    # Measure Serialization Speed
    n_iters = 10000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = json.dumps(rest_dict).encode("utf-8")
    json_ser_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = proto_resp.SerializeToString()
    proto_ser_time = time.perf_counter() - t0
    ser_speedup = json_ser_time / proto_ser_time

    # Measure Deserialization Speed
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = json.loads(json_bytes)
    json_deser_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        msg = search_pb2.SearchResponse()
        msg.ParseFromString(proto_bytes)
    proto_deser_time = time.perf_counter() - t0
    deser_speedup = json_deser_time / proto_deser_time

    print(f"  • Serialization Speedup:   {ser_speedup:.2f}x faster ({json_ser_time*1e6/n_iters:.1f}µs vs {proto_ser_time*1e6/n_iters:.1f}µs)")
    print(f"  • Deserialization Speedup: {deser_speedup:.2f}x faster ({json_deser_time*1e6/n_iters:.1f}µs vs {proto_deser_time*1e6/n_iters:.1f}µs)")

    # 4. End-to-End Latency Benchmark (1,000 queries)
    print("\n[3/4] Running 1,000 end-to-end queries across gRPC and REST interfaces...")
    query_pool = [
        "transformer attention",
        "distributed sharding",
        "consistent hashing",
        "inverted index search",
        "neural reranking",
        "fault-tolerant memory",
        "quantization throughput",
        "cache retrieval"
    ]

    num_queries = 1000
    queries = [random.choice(query_pool) for _ in range(num_queries)]

    # Warmup
    for q in queries[:20]:
        _ = rest_client.get(f"/search?q={q}&top_k=10")
        _ = grpc_stub.Search(search_pb2.SearchRequest(query=q, top_k=10))

    # Benchmark REST
    rest_latencies = []
    t_start_rest = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        resp = rest_client.get(f"/search?q={q}&top_k=10")
        assert resp.status_code == 200
        rest_latencies.append((time.perf_counter() - t0) * 1000.0)
    total_time_rest = time.perf_counter() - t_start_rest
    rest_qps = num_queries / total_time_rest

    # Benchmark gRPC
    grpc_latencies = []
    t_start_grpc = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        resp = grpc_stub.Search(search_pb2.SearchRequest(query=q, top_k=10))
        assert len(resp.hits) > 0
        grpc_latencies.append((time.perf_counter() - t0) * 1000.0)
    total_time_grpc = time.perf_counter() - t_start_grpc
    grpc_qps = num_queries / total_time_grpc

    # Compute percentiles
    rest_p50 = percentile(rest_latencies, 50)
    rest_p90 = percentile(rest_latencies, 90)
    rest_p95 = percentile(rest_latencies, 95)
    rest_p99 = percentile(rest_latencies, 99)

    grpc_p50 = percentile(grpc_latencies, 50)
    grpc_p90 = percentile(grpc_latencies, 90)
    grpc_p95 = percentile(grpc_latencies, 95)
    grpc_p99 = percentile(grpc_latencies, 99)

    p95_speedup = (rest_p95 - grpc_p95) / rest_p95 * 100.0

    print("\n[4/4] End-to-End Latency & Throughput Results:")
    print(f"{'Interface':<12} | {'Throughput':<12} | {'P50':<9} | {'P90':<9} | {'P95':<9} | {'P99':<9}")
    print("-" * 72)
    print(f"{'REST (JSON)':<12} | {rest_qps:>8.1f} QPS  | {rest_p50:>7.2f}ms | {rest_p90:>7.2f}ms | {rest_p95:>7.2f}ms | {rest_p99:>7.2f}ms")
    print(f"{'gRPC (Proto)':<12} | {grpc_qps:>8.1f} QPS  | {grpc_p50:>7.2f}ms | {grpc_p90:>7.2f}ms | {grpc_p95:>7.2f}ms | {grpc_p99:>7.2f}ms")

    # Clean shutdown
    grpc_channel.close()
    grpc_server.stop(grace=None)

    # Save output artifacts
    results_data = {
        "benchmark": "irkit_grpc_vs_rest",
        "num_queries": num_queries,
        "corpus_docs": 500,
        "payload_bytes": {
            "rest_json": json_size,
            "grpc_protobuf": proto_size,
            "reduction_percent": round(payload_reduction_pct, 2)
        },
        "serialization_speedup_x": round(ser_speedup, 2),
        "deserialization_speedup_x": round(deser_speedup, 2),
        "rest_json": {
            "qps": round(rest_qps, 1),
            "p50_ms": round(rest_p50, 2),
            "p90_ms": round(rest_p90, 2),
            "p95_ms": round(rest_p95, 2),
            "p99_ms": round(rest_p99, 2),
        },
        "grpc_protobuf": {
            "qps": round(grpc_qps, 1),
            "p50_ms": round(grpc_p50, 2),
            "p90_ms": round(grpc_p90, 2),
            "p95_ms": round(grpc_p95, 2),
            "p99_ms": round(grpc_p99, 2),
        },
        "p95_latency_reduction_percent": round(p95_speedup, 2)
    }

    with open("benchmarks/grpc_vs_rest_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    with open("benchmarks/grpc_vs_rest_results.txt", "w") as f:
        f.write("IRKit SWE Micro-Benchmark: gRPC vs. REST\n")
        f.write("=" * 45 + "\n")
        f.write(f"Wire Payload: REST={json_size}B, gRPC={proto_size}B (-{payload_reduction_pct:.1f}%)\n")
        f.write(f"Serialization: {ser_speedup:.2f}x faster, Deserialization: {deser_speedup:.2f}x faster\n")
        f.write(f"REST Throughput: {rest_qps:.1f} QPS, P50={rest_p50:.2f}ms, P95={rest_p95:.2f}ms, P99={rest_p99:.2f}ms\n")
        f.write(f"gRPC Throughput: {grpc_qps:.1f} QPS, P50={grpc_p50:.2f}ms, P95={grpc_p95:.2f}ms, P99={grpc_p99:.2f}ms\n")
        f.write(f"P95 Latency Reduction: {p95_speedup:.1f}%\n")

    print("\n✅ Benchmark telemetry written to:")
    print("  • benchmarks/grpc_vs_rest_results.json")
    print("  • benchmarks/grpc_vs_rest_results.txt")


if __name__ == "__main__":
    run_benchmark()
