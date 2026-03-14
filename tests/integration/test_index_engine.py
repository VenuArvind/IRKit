import pytest
from irkit.core.engine import IndexEngine
from irkit.sources.file import FileSource
from irkit.rankers.bm25 import BM25Ranker
from irkit.storage.memory import InMemoryStorage

def test_engine_end_to_end():
    """ Verify the full flow: File Source -> Memory Storage -> BM25 Ranker """
    
    # 1. Setup components
    # Assuming tests/dummy_data.csv exists
    source = FileSource(file_path="tests/dummy_data.csv")
    storage = InMemoryStorage()
    ranker = BM25Ranker()
    engine = IndexEngine(ranker=ranker, storage=storage)
    
    # 2. Run Indexing
    engine.index(source)
    
    # 3. Verify Storage
    assert storage.count() == 3
    doc1 = storage.get("1")
    assert doc1.title == "Intro to AI"
    
    # 4. Run Search
    results = engine.search("mathematical vectors", top_k=1)
    
    # 5. Verify Results
    assert len(results) == 1
    assert results[0].doc_id == "2" # Vector Search doc
    assert "Jane Smith" in results[0].metadata.get("author", "")
