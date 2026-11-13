import time
from concurrent import futures
from typing import Optional
import grpc

from irkit.proto import search_pb2, search_pb2_grpc


class SearchServicer(search_pb2_grpc.SearchServiceServicer):
    """
    gRPC Servicer implementing binary search RPCs on top of IndexEngine.
    """

    def __init__(self, engine):
        self.engine = engine

    def _get_quantization_mode(self) -> str:
        if not self.engine:
            return "none"
        if hasattr(self.engine.ranker, 'quantization'):
            return self.engine.ranker.quantization or "none"
        elif hasattr(self.engine.ranker, 'rankers'):
            for r in self.engine.ranker.rankers:
                if hasattr(r, 'quantization'):
                    return r.quantization or "none"
        return "none"

    def _convert_hit(self, result) -> search_pb2.SearchHit:
        # Convert metadata values to strings for proto map<string, string>
        meta = {}
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            for k, v in result.metadata.items():
                meta[str(k)] = str(v)

        return search_pb2.SearchHit(
            doc_id=str(getattr(result, "doc_id", "")),
            score=float(getattr(result, "score", 0.0)),
            title=str(getattr(result, "title", "")),
            snippet=str(getattr(result, "snippet", "")),
            metadata=meta
        )

    def Search(self, request: search_pb2.SearchRequest, context) -> search_pb2.SearchResponse:
        """
        Unary RPC search: fetches top_k results and serializes into Protobuf.
        """
        if self.engine is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Search engine not initialized")

        t0 = time.perf_counter()
        results = self.engine.search(
            request.query,
            top_k=request.top_k if request.top_k > 0 else 10,
            rerank=request.rerank
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        hits = [self._convert_hit(r) for r in results]
        total_docs = self.engine.storage.count() if hasattr(self.engine, "storage") else 0

        return search_pb2.SearchResponse(
            query=request.query,
            hits=hits,
            latency_ms=latency_ms,
            total_docs=total_docs,
            quantization_mode=self._get_quantization_mode()
        )

    def StreamSearch(self, request: search_pb2.SearchRequest, context):
        """
        Server-streaming RPC: streams hits back over HTTP/2 frame-by-frame.
        """
        if self.engine is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Search engine not initialized")

        results = self.engine.search(
            request.query,
            top_k=request.top_k if request.top_k > 0 else 10,
            rerank=request.rerank
        )

        for result in results:
            yield self._convert_hit(result)

    def Health(self, request: search_pb2.HealthRequest, context) -> search_pb2.HealthResponse:
        """
        Health check endpoint.
        """
        is_ready = self.engine is not None
        total_docs = self.engine.storage.count() if is_ready and hasattr(self.engine, "storage") else 0
        status = "SERVING" if is_ready else "NOT_SERVING"

        return search_pb2.HealthResponse(
            status=status,
            engine_ready=is_ready,
            total_docs=total_docs
        )


def create_grpc_server(engine, max_workers: int = 10, port: int = 50051) -> grpc.Server:
    """
    Factory function to configure and instantiate a production gRPC server.
    """
    options = [
        ('grpc.max_send_message_length', 16 * 1024 * 1024),
        ('grpc.max_receive_message_length', 16 * 1024 * 1024),
        ('grpc.so_reuseport', 1),
        ('grpc.http2.min_ping_interval_without_data_ms', 5000),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    servicer = SearchServicer(engine)
    search_pb2_grpc.add_SearchServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    return server

