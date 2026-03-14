from typing import List, Dict
from irkit.rankers.base import BaseRanker, SearchResult

class HybridRanker(BaseRanker):
    """ Hybrid Ranker combining BM25 and Semantic Search using Reciprocal Rank Fusion (RRF) """

    def __init__(self, rankers: List[BaseRanker], k: int = 60):
        self.rankers = rankers
        self.k = k
        # Expose embedder for caching logic
        self.embedder = next((r.embedder for r in rankers if hasattr(r, 'embedder')), None)
    
    def index(self, documents: List[dict]) -> None:
        for ranker in self.rankers:
            ranker.index(documents)
    
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        
        all_results = [ranker.search(query, top_k=50) for ranker in self.rankers]

        # Apply RRF: score(d) = sum( 1 / (k+rank) )
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}

        for ranker_results in all_results:
            for rank, res in enumerate(ranker_results):
                doc_id = res.doc_id
                doc_map[doc_id] = res

                score = 1.0 / (self.k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        
        # Sort by RRF score
        combined = []
        for doc_id, score in rrf_scores.items():
            res = doc_map[doc_id]
            res.score = score
            combined.append(res)
        
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]


        
    