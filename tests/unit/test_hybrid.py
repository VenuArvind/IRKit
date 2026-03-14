import pytest
from irkit.rankers.hybrid import HybridRanker
from irkit.rankers.bm25 import BM25Ranker
from irkit.rankers.base import SearchResult

class MockRanker:
    """ A simple mock ranker that returns fixed results for testing fusion """
    def __init__(self, results):
        self.results = results
    def index(self, docs): pass
    def search(self, query, top_k=10): return self.results

def test_hybrid_rrf_fusion():
    """ Verify that HybridRanker correctly fuses results using Reciprocal Rank Fusion (RRF) """
    
    # doc1 is #1 in ranker A, doc2 is #2
    # doc2 is #1 in ranker B, doc1 is #2
    res_a = [
        SearchResult(doc_id="doc1", score=1.0, title="Doc 1", snippet=""),
        SearchResult(doc_id="doc2", score=0.8, title="Doc 2", snippet=""),
    ]
    res_b = [
        SearchResult(doc_id="doc2", score=1.0, title="Doc 2", snippet=""),
        SearchResult(doc_id="doc1", score=0.8, title="Doc 1", snippet=""),
    ]
    
    ranker_a = MockRanker(res_a)
    ranker_b = MockRanker(res_b)
    
    hybrid = HybridRanker(rankers=[ranker_a, ranker_b], k=60)
    
    # Fusion should result in doc1 and doc2 having identical scores in this symmetric case
    results = hybrid.search("any")
    
    assert len(results) == 2
    ids = [r.doc_id for r in results]
    assert "doc1" in ids
    assert "doc2" in ids
    
    # RRF Score for rank 1: 1 / (60 + 1) = 0.01639
    # RRF Score for rank 2: 1 / (60 + 2) = 0.01612
    # Total for both: 0.03251
    assert pytest.approx(results[0].score, rel=1e-3) == 0.03251
