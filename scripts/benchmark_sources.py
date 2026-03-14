import time
import numpy as np
from irkit import IndexEngine, ArXivSource, WikipediaSource, NewsSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage, CrossEncoderRanker

def run_source_benchmark(source_name, source_obj):
    print(f"\n🚀 Starting Benchmark for: {source_name}")
    
    # Setup Engine
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    hybrid = HybridRanker(rankers=[bm25, semantic])
    reranker = CrossEncoderRanker()
    
    engine = IndexEngine(ranker=hybrid, storage=InMemoryStorage(), reranker=reranker)
    
    # 2. Index Data
    print(f"📥 Indexing 100 docs from {source_name}...")
    engine.index(source_obj, max_docs=100)
    
    queries = [
        "artificial intelligence", "space exploration", "renewable energy",
        "global economy", "health and wellness", "technology trends",
        "scientific discovery", "world history", "modern art", "cybersecurity"
    ]
    
    results_map = {}
    
    # 3. Benchmark Configurations
    configs = [
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
            for _ in range(3): # 30 samples total
                engine.search(q, rerank=params["rerank"])
        
        stats = engine.stats()["latency"]
        results_map[name] = stats
        print(f"   Done: P50={stats['p50']}ms, P95={stats['p95']}ms")

    return results_map

def main():
    sources = [
        ("Wikipedia", WikipediaSource(category="Artificial_intelligence")),
        ("News", NewsSource())
    ]
    
    all_results = {}
    for name, obj in sources:
        all_results[name] = run_source_benchmark(name, obj)

    print("\n📊 CROSS-DATASET BENCHMARK RESULTS (100 docs):")
    print("| Source | Type | P50 (ms) | P95 (ms) |")
    print("|--------|------|----------|----------|")
    for src_name, metrics in all_results.items():
        for type_name, stats in metrics.items():
            print(f"| {src_name:10} | {type_name:8} | {stats['p50']:8.2f} | {stats['p95']:8.2f} |")

if __name__ == "__main__":
    main()
