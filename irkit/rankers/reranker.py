from typing import List
from irkit.rankers.base import SearchResult
from sentence_transformers import CrossEncoder

class CrossEncoderRanker:
    """ 
    A High-Precision Reranker using Cross-Encoders.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        1. Prepare query-doc pairs
        2. Get scores from model
        3. Re-sort candidates
        """
        pairs = [[query, f"{res.title} {res.snippet}"] for res in candidates]
        scores = self.model.predict(pairs)
        for res, score in zip(candidates, scores):
            res.score = float(score)
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates
