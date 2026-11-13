import pytest
import grpc
from concurrent import futures

from irkit.core.engine import IndexEngine
from irkit.rankers.bm25 import BM25Ranker
from irkit.storage.memory import InMemoryStorage
from irkit.proto import search_pb2, search_pb2_grpc
from irkit.serve.grpc_server import SearchServicer, create_grpc_server

DOCS = [
    {"id": "doc1", "title": "Attention Is All You Need", "text": "transformer self attention multi-head", "metadata": {"year": "2017", "author": "Vaswani"}},
    {"id": "doc2", "title": "BERT Language Model", "text": "deep bidirectional transformer representations", "metadata": {"year": "2018", "author": "Devlin"}},
    {"id": "doc3", "title": "Distributed Search Engines", "text": "sharding consistent hashing indexing inverted index", "metadata": {"year": "2024", "category": "systems"}},
]

@pytest.fixture(scope="module")
def grpc_service():
    # Setup test engine
    storage = InMemoryStorage()
    ranker = BM25Ranker()
    engine = IndexEngine(ranker=ranker, storage=storage)
    
    for d in DOCS:
        storage.save(type("Doc", (), {"id": d["id"], "title": d["title"], "text": d["text"], "metadata": d["metadata"]})())
    ranker.index(DOCS)
    
    # Setup test server on ephemeral port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = SearchServicer(engine)
    search_pb2_grpc.add_SearchServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = search_pb2_grpc.SearchServiceStub(channel)

    yield stub, engine

    channel.close()
    server.stop(grace=None)


def test_grpc_health(grpc_service):
    stub, engine = grpc_service
    response = stub.Health(search_pb2.HealthRequest())
    assert response.status == "SERVING"
    assert response.engine_ready is True
    assert response.total_docs == len(DOCS)


def test_grpc_unary_search(grpc_service):
    stub, engine = grpc_service
    request = search_pb2.SearchRequest(query="transformer", top_k=2, rerank=False)
    response = stub.Search(request)

    assert response.query == "transformer"
    assert len(response.hits) == 2
    assert response.hits[0].doc_id in ["doc1", "doc2"]
    assert response.latency_ms > 0
    assert "author" in response.hits[0].metadata


def test_grpc_streaming_search(grpc_service):
    stub, engine = grpc_service
    request = search_pb2.SearchRequest(query="sharding", top_k=1, rerank=False)
    stream = stub.StreamSearch(request)

    hits = list(stream)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc3"
    assert "Distributed" in hits[0].title
    assert hits[0].metadata.get("category") == "systems"
