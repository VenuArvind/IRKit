from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("query", "top_k", "rerank")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    RERANK_FIELD_NUMBER: _ClassVar[int]
    query: str
    top_k: int
    rerank: bool
    def __init__(self, query: _Optional[str] = ..., top_k: _Optional[int] = ..., rerank: _Optional[bool] = ...) -> None: ...

class SearchHit(_message.Message):
    __slots__ = ("doc_id", "score", "title", "snippet", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    score: float
    title: str
    snippet: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, doc_id: _Optional[str] = ..., score: _Optional[float] = ..., title: _Optional[str] = ..., snippet: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("query", "hits", "latency_ms", "total_docs", "quantization_mode")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_MODE_FIELD_NUMBER: _ClassVar[int]
    query: str
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    latency_ms: float
    total_docs: int
    quantization_mode: str
    def __init__(self, query: _Optional[str] = ..., hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ..., latency_ms: _Optional[float] = ..., total_docs: _Optional[int] = ..., quantization_mode: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "engine_ready", "total_docs")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_READY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCS_FIELD_NUMBER: _ClassVar[int]
    status: str
    engine_ready: bool
    total_docs: int
    def __init__(self, status: _Optional[str] = ..., engine_ready: _Optional[bool] = ..., total_docs: _Optional[int] = ...) -> None: ...
