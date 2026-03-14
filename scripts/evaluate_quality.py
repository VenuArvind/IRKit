import time
from typing import List, Dict, Set
from irkit import IndexEngine, ArXivSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage, CrossEncoderRanker
from irkit.core.evaluation import calculate_mrr, calculate_ndcg

def run_evaluation_suite():
    print("🔬 Starting Search Quality Evaluation...")
    
    # 1. Setup Engine
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    hybrid = HybridRanker(rankers=[bm25, semantic])
    reranker = CrossEncoderRanker()
    
    engine = IndexEngine(ranker=hybrid, storage=InMemoryStorage(), reranker=reranker)
    
    # 2. Index Data
    print("📥 Indexing 200 ArXiv papers...")
    engine.index(ArXivSource(), max_docs=200)
    
    # 3. Define Ground Truth (Gold Standard) based on actual ArXiv IDs
    benchmarks = {
        "diphoton": {"0704.0001"},
        "graph": {"0704.0002"},
        "earth moon": {"0704.0003"},
        "shock": {"0704.0008"},
    }
    
    modes = [
        ("BM25 Only", {"ranker": bm25, "rerank": False}),
        ("Semantic Only", {"ranker": semantic, "rerank": False}),
        ("Hybrid (RRF)", {"ranker": hybrid, "rerank": False}),
        ("Hybrid + Reranking", {"ranker": hybrid, "rerank": True}),
    ]
    
    print("\n📊 SEARCH QUALITY REPORT:")
    print("| Mode | Mean MRR | Mean nDCG@10 |")
    print("|------|----------|--------------|")
    
    for mode_name, params in modes:
        mrr_scores = []
        ndcg_scores = []
        
        engine.ranker = params["ranker"]
        
        for query, relevant_ids in benchmarks.items():
            results = engine.search(query, top_k=10, rerank=params["rerank"])
            
            mrr_scores.append(calculate_mrr(relevant_ids, results))
            ndcg_scores.append(calculate_ndcg(relevant_ids, results, k=10))
            
        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        
        print(f"| {mode_name:18} | {avg_mrr:8.4f} | {avg_ndcg:12.4f} |")

if __name__ == "__main__":
    run_evaluation_suite()
