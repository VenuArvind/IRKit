import pytest
from irkit.rankers.bm25 import BM25Ranker

DOCS = [
    {"id": "1", "title": "Attention Is All You Need", "text": "transformer self-attention", "metadata": {}},
    {"id": "2", "title": "BERT: Pre-training", "text": "bidirectional encoder representations", "metadata": {}},
    {"id": "3", "title": "GPT-4 Technical Report", "text": "large language model openai", "metadata": {}},
]

def test_bm25_ranking_accuracy():
    """ Verify that BM25 correctly ranks the most relevant document first """
    ranker = BM25Ranker()
    ranker.index(DOCS)
    
    results = ranker.search("transformer", top_k=1)
    
    assert len(results) == 1
    assert results[0].doc_id == "1"
    assert "Attention" in results[0].title

def test_bm25_empty_query():
    """ Ensure the ranker doesn't crash on empty queries """
    ranker = BM25Ranker()
    ranker.index(DOCS)
    results = ranker.search("", top_k=3)
    assert len(results) == 3

def test_bm25_error_handling():
    """ Ensure the ranker raises an error if search is called before index """
    ranker = BM25Ranker()
    with pytest.raises(RuntimeError):
        ranker.search("any query")
