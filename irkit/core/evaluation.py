import numpy as np
from typing import List, Set
from irkit.rankers.base import SearchResult

def calculate_mrr(relevant_ids: Set[str], results: List[SearchResult]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    MRR = 1/rank of the first relevant document.
    """
    for i, res in enumerate(results):
        if res.doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_dcg(relevances: List[int]) -> float:
    """
    Calculates Discounted Cumulative Gain (DCG).
    """
    relevances = np.asarray(relevances, dtype=float)
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0

def calculate_ndcg(relevant_ids: Set[str], results: List[SearchResult], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG@k).
    Assumes binary relevance (1 if in relevant_ids, else 0).
    """
    # 1. Get binary relevance scores for the results
    relevances = [1 if res.doc_id in relevant_ids else 0 for res in results[:k]]
    
    # 2. Calculate actual DCG
    actual_dcg = calculate_dcg(relevances)
    
    # 3. Calculate Ideal DCG (IDCG)
    # The best possible ranking would have all relevant docs at the top
    ideal_relevances = sorted([1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids)), reverse=True)
    ideal_dcg = calculate_dcg(ideal_relevances)
    
    if ideal_dcg > 0:
        return actual_dcg / ideal_dcg
    return 0.0
