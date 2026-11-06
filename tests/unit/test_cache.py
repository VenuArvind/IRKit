import numpy as np
from irkit.core.cache import SemanticCache
from irkit.rankers.base import SearchResult

def test_semantic_cache_exact_and_threshold():
    cache = SemanticCache(threshold=0.90)

    # 1. Empty cache check
    q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cache.get(q_vec) is None

    # 2. Store result
    mock_results = [SearchResult(doc_id="1", score=0.95, title="Test", snippet="Sample snippet")]
    cache.set(q_vec, mock_results)

    # 3. Exact query hit
    hit = cache.get(q_vec)
    assert hit is not None
    assert len(hit) == 1
    assert hit[0].doc_id == "1"

    # 4. Similar query hit (cos sim ~0.948 >= 0.90)
    similar_vec = np.array([0.95, 0.31, 0.0], dtype=np.float32)
    similar_hit = cache.get(similar_vec)
    assert similar_hit is not None
    assert similar_hit[0].doc_id == "1"

    # 5. Dissimilar query miss (orthogonal)
    orthogonal_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cache.get(orthogonal_vec) is None

    # 6. Clear cache
    cache.clear()
    assert cache.get(q_vec) is None

