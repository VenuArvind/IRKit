from fastapi import FastAPI, HTTPException, Query 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from typing import List, Optional 
import time 

app = FastAPI(title="IRKit Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None

def set_engine(engine):
    global _engine
    _engine = engine 


class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    latency_ms: float
    total_docs: int
    system_metrics: dict
    quantization_mode: Optional[str] = None

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
    rerank: bool = Query(False, description="Enable advanced reranking (Cross-Encoder)")
):
    """
    Performs a search using the IRKit engine.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    from irkit.core.profiler import last_metrics
    
    # The search call is already decorated with @profile_engine
    results = _engine.search(q, top_k=top_k, rerank=rerank)
    
    # Detect quantization mode
    q_mode = "none"
    if hasattr(_engine.ranker, 'quantization'):
        q_mode = _engine.ranker.quantization or "none"
    elif hasattr(_engine.ranker, 'rankers'):
         for r in _engine.ranker.rankers:
             if hasattr(r, 'quantization'):
                 q_mode = r.quantization or "none"

    return {
        "query": q,
        "results": [r.__dict__ for r in results],
        "latency_ms": round(last_metrics["duration"] * 1000, 2),
        "total_docs": _engine.storage.count(),
        "system_metrics": last_metrics,
        "quantization_mode": q_mode
    }

@app.get("/health")
def health():
    """Returns basic health information about the search engine."""

    return {"status": "ok", "engine_ready": _engine is not None}