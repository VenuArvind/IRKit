"""
IRKit Season Finale: Semantic ArXiv Search Demo
"""
from irkit.sources.arxiv import ArxivSource
from irkit.embedders.huggingface import HuggingFaceEmbedder
from irkit.rankers.hybrid import HybridRanker
from irkit.rankers.bm25 import BM25Ranker
from irkit.rankers.semantic import SemanticRanker
from irkit.storage.memory import MemoryStorage
from irkit.core.index import IndexEngine

def main():
    print("--- 🚀 Welcome to the IRKit ArXiv Search Demo 🚀 ---")
    
    # 1. Initialize our components
    source = ArxivSource(max_docs=20)
    embedder = HuggingFaceEmbedder()
    storage = MemoryStorage()
    
    # 2. Setup Hybrid Ranking
    bm25 = BM25Ranker()
    semantic = SemanticRanker(embedder)
    ranker = HybridRanker(rankers=[bm25, semantic])
    
    # 3. Build the Engine
    engine = IndexEngine(ranker=ranker, storage=storage)
    
    # 4. Ingest real data from ArXiv
    engine.index(source)
    
    # 5. Interactive Search Loop
    while True:
        query = input("\n🔍 Enter a search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        results = engine.search(query, top_k=3)
        
        if not results:
            print("❌ No matches found try different keywords or concepts.")
            continue
            
        print(f"\n✅ Found {len(results)} results:")
        for i, res in enumerate(results):
            print(f"\n[{i+1}] {res.title}")
            print(f"    Link: {res.metadata.get('url', 'N/A')}")
            print(f"    Summary: {res.snippet}")
            print(f"    Hybrid Score: {res.score:.4f}")

if __name__ == "__main__":
    main()
