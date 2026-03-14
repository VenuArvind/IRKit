import numpy as np 
import faiss 
from typing import List
from irkit.rankers.base import BaseRanker, SearchResult
from irkit.embedders.base import BaseEmbedder

class SemanticRanker(BaseRanker):
    """
    Semantic Ranker using FAISS for vector similarity search.
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self. index_dim = embedder.dimension
        self.faiss_index = faiss.IndexFlatL2(self.index_dim)
        self.doc_store = []

    def index(self, documents: List[dict]) -> None:
        self.doc_store = documents
        texts = [doc['text'] for doc in documents]
        vectors = self.embedder.embed(texts)
        self.faiss_index.add(vectors.astype('float32'))
        print(f" [SemanticRanker] Indexed {len(documents)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.faiss_index.ntotal == 0:
            return []
        # 1. Embed the query
        query_vector = self.embedder.embed([query]).astype("float32")
        
        # 2. Search FAISS
        # distances is the "L2 distance" (smaller is better)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue # FAISS returns -1 if no match found
            
            doc = self.doc_store[idx]
            # Convert L2 distance to a "score" where higher is better
            # (Adding 1 to avoid division by zero)
            score = 1.0 / (1.0 + float(dist))
            
            results.append(SearchResult(
                doc_id=doc["id"],
                score=score,
                title=doc["title"],
                snippet=doc["text"][:200] + "...",
                metadata=doc.get("metadata", {})
            ))
        return results