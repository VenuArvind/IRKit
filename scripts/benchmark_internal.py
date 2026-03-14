import time
import numpy as np
from irkit import IndexEngine, ArXivSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage, CrossEncoderRanker

def run_benchmark():
    print("🚀 Starting Internal Benchmark...")
    
    # 1. Setup Engine
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    hybrid = HybridRanker(rankers=[bm25, semantic])
    reranker = CrossEncoderRanker()
    
    engine = IndexEngine(ranker=hybrid, storage=InMemoryStorage(), reranker=reranker)
    
    # 2. Index Data
    print("📥 Indexing 500 ArXiv papers (this may take a minute)...")
    engine.index(ArXivSource(), max_docs=500)
    
    queries = [
        "transformer models", "quantum computing", "climate change machine learning",
        "deep reinforcement learning", "natural language processing", "computer vision",
        "graph neural networks", "federated learning", "diffusion models", "llm efficiency"
    ]
    
    results_map = {}
    
    # 3. Benchmark Configurations
    configs = [
        ("BM25", {"ranker": bm25, "rerank": False}),
        ("Semantic", {"ranker": semantic, "rerank": False}),
        ("Hybrid", {"ranker": hybrid, "rerank": False}),
        ("Reranked", {"ranker": hybrid, "rerank": True}),
    ]
    
    for name, params in configs:
        print(f"⏱️ Benchmarking {name}...")
        engine.ranker = params["ranker"]
        engine.metrics.reset()
        
        # Warmup
        engine.search(queries[0], rerank=params["rerank"])
        
        for q in queries:
            for _ in range(5): # 50 samples total (10 queries * 5)
                engine.search(q, rerank=params["rerank"])
        
        stats = engine.stats()["latency"]
        results_map[name] = stats
        print(f"   Done: P50={stats['p50']}ms, P95={stats['p95']}ms")

    print("\n📊 FINAL BENCHMARK RESULTS:")
    print("| Ranker | P50 (ms) | P95 (ms) | P99 (ms) |")
    print("|--------|----------|----------|----------|")
    for name, stats in results_map.items():
        print(f"| {name:8} | {stats['p50']:8.2f} | {stats['p95']:8.2f} | {stats['p99']:8.2f} |")

if __name__ == "__main__":
    run_benchmark()
