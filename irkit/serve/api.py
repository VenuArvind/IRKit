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

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100)
):
    """
    Performs a search using the IRKit engine.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    t0 = time.perf_counter()
    results = _engine.search(q, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": q,
        "results": [r.__dict__ for r in results],
        "latency_ms": round(latency_ms, 2),
        "total_docs": _engine.storage.count()
    }

@app.get("/health")
def health():
    """Returns basic health information about the search engine."""

    return {"status": "ok", "engine_ready": _engine is not None}