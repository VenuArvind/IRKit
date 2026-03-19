import numpy as np 
import faiss 
from typing import List
from irkit.rankers.base import BaseRanker, SearchResult
from irkit.embedders.base import BaseEmbedder

class SemanticRanker(BaseRanker):
    """
    Semantic Ranker using FAISS for vector similarity search.
    """

    def __init__(self, embedder: BaseEmbedder, quantization: str = None):
        self.embedder = embedder
        self.index_dim = embedder.dimension
        self.quantization = quantization
        self.faiss_index = faiss.IndexFlatL2(self.index_dim)
        self.doc_store = []
        
        # Quantization state
        self.quantizer = None
        self.quantized_vectors = None

    def index(self, documents: List[dict]) -> None:
        from irkit.core.quantization import ScalarQuantizer, ProductQuantizer
        
        self.doc_store = documents
        texts = [doc['text'] for doc in documents]
        vectors = self.embedder.embed(texts).astype('float32')
        
        if self.quantization == "sq8":
            self.quantizer = ScalarQuantizer()
            self.quantized_vectors = self.quantizer.quantize(vectors)
        elif self.quantization == "pq":
            self.quantizer = ProductQuantizer(num_subspaces=8)
            self.quantizer.train(vectors)
            self.quantized_vectors = self.quantizer.encode(vectors)
        else:
            self.faiss_index.add(vectors)
            
        print(f" [SemanticRanker] Indexed {len(documents)} documents (Quantization: {self.quantization}).")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # 1. Embed the query
        query_vector = self.embedder.embed([query]).astype("float32")
        
        if self.quantization:
            from irkit.core.quantization import asymmetric_distance
            # Use our custom quantized search
            scores = asymmetric_distance(query_vector, self.quantized_vectors, self.quantizer, self.quantization)
            
            # Sort by score descending
            indices = np.argsort(scores)[::-1][:top_k]
            distances = scores[indices] # These are actually similarities
            
            results = []
            for score, idx in zip(distances, indices):
                doc = self.doc_store[idx]
                results.append(SearchResult(
                    doc_id=doc["id"],
                    score=float(score),
                    title=doc["title"],
                    snippet=doc["text"][:200] + "...",
                    metadata=doc.get("metadata", {})
                ))
            return results
        
        # 2. Search FAISS (Standard)
        if self.faiss_index.ntotal == 0:
            return []
            
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue 
            
            doc = self.doc_store[idx]
            score = 1.0 / (1.0 + float(dist))
            
            results.append(SearchResult(
                doc_id=doc["id"],
                score=score,
                title=doc["title"],
                snippet=doc["text"][:200] + "...",
                metadata=doc.get("metadata", {})
            ))
        return results