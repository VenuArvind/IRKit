import numpy as np
import sys
from typing import List, Iterator
from irkit.core.engine import IndexEngine
from irkit.rankers.semantic import SemanticRanker
from irkit.embedders.base import BaseEmbedder
from irkit.storage.memory import InMemoryStorage
from irkit.sources.base import BaseSource, Document

# 1. Mock Infrastructure for pure systems testing
class MockEmbedder(BaseEmbedder):
    def __init__(self, dimension=384):
        self._dim = dimension
    def embed(self, texts: List[str]) -> np.ndarray:
        return np.random.randn(len(texts), self._dim).astype(np.float32)
    @property
    def dimension(self) -> int:
        return self._dim

class MockSource(BaseSource):
    def load(self, max_docs: int = 100) -> Iterator[Document]:
        for i in range(max_docs):
            yield Document(id=f"doc_{i}", title=f"Title {i}", text=f"Text content for document {i} with some length.")

def run_benchmark(mode: str = None):
    print(f"\n" + "="*60)
    print(f"RUNNING BENCHMARK: Mode={mode if mode else 'BASELINE (float32)'}")
    print("="*60)
    
    embedder = MockEmbedder()
    storage = InMemoryStorage()
    ranker = SemanticRanker(embedder=embedder, quantization=mode)
    engine = IndexEngine(ranker=ranker, storage=storage)
    
    source = MockSource()
    
    # 1. Indexing (Profiling will print report automatically)
    engine.index(source, max_docs=20000)
    
    # 2. Search
    query = "test search query"
    results = engine.search(query, top_k=5)
    
    print(f"Index size: {storage.count()} documents")
    print(f"Search results returned: {len(results)}")

if __name__ == "__main__":
    # Run all three modes
    try:
        run_benchmark(mode=None)   # Baseline
        run_benchmark(mode="sq8")  # Scalar Quantization
        run_benchmark(mode="pq")   # Product Quantization
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
