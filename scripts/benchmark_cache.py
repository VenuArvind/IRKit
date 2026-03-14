import time
import numpy as np
from irkit import IndexEngine, ArXivSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage
from irkit.core.cache import SemanticCache

def run_cache_benchmark():
    print("🚀 Starting Semantic Cache Benchmark...")
    
    # 1. Setup Engine with Cache
    embedder = SentenceTransformerEmbedder()
    ranker = HybridRanker(rankers=[BM25Ranker(), SemanticRanker(embedder)])
    cache = SemanticCache(threshold=0.9) # More generous for semantic test
    
    engine = IndexEngine(ranker=ranker, storage=InMemoryStorage(), cache=cache)
    
    # 2. Index Data
    print("📥 Indexing 100 ArXiv papers...")
    engine.index(ArXivSource(), max_docs=100)
    
    query = "transformer models for NLP"
    
    print(f"\n🔍 Testing Query: '{query}'")
    
    # Round 1: Cold Cache
    t0 = time.perf_counter()
    engine.search(query)
    cold_latency = (time.perf_counter() - t0) * 1000
    print(f"❄️ Cold Search Latency: {cold_latency:.2f}ms")
    
    # Round 2: Hot Cache (Exact match)
    t0 = time.perf_counter()
    engine.search(query)
    hot_latency = (time.perf_counter() - t0) * 1000
    print(f"🔥 Hot Search Latency (Exact): {hot_latency:.2f}ms")
    
    # Round 3: Semantic Match (Slightly different wording)
    semantic_query = "transformers in natural language processing"
    t0 = time.perf_counter()
    engine.search(semantic_query)
    semantic_latency = (time.perf_counter() - t0) * 1000
    print(f"🧠 Semantic Cache Latency (Similar): {semantic_latency:.2f}ms")

    print(f"\n📈 Cache Speedup: {cold_latency / hot_latency:.1f}x faster")

if __name__ == "__main__":
    run_cache_benchmark()
