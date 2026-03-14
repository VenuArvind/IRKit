import numpy as np
from typing import List, Optional, Tuple
from irkit.rankers.base import SearchResult

class SemanticCache:
    """
    A semantic cache that stores and retrieves search results based on vector similarity.
    Uses a simple in-memory list of (embedding, results) for this demonstration.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.entries: List[Tuple[np.ndarray, List[SearchResult]]] = []

    def get(self, query_embedding: np.ndarray) -> Optional[List[SearchResult]]:
        """
        Check if a similar query exists in the cache.
        """
        if not self.entries:
            return None

        # Calculate cosine similarities
        # query_embedding: (D,)
        # all_embeddings: (N, D)
        all_embeddings = np.array([e[0] for e in self.entries])
        
        # Norms for cosine similarity
        norm_q = np.linalg.norm(query_embedding)
        norm_all = np.linalg.norm(all_embeddings, axis=1)
        
        similarities = np.dot(all_embeddings, query_embedding) / (norm_all * norm_q)
        
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= self.threshold:
            return self.entries[max_idx][1]
        
        return None

    def set(self, query_embedding: np.ndarray, results: List[SearchResult]):
        """
        Store results in the cache.
        """
        # Simple management: keep cache size reasonable
        if len(self.entries) > 1000:
            self.entries.pop(0)
            
        self.entries.append((query_embedding, results))

    def clear(self):
        self.entries.clear()
