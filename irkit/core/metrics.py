import numpy as np 
from typing import List 

class LatencyTracker:
    """
    Tracks latency metrics for search queries
    """
    
    def __init__(self):
        self._latencies: List[float] = []

    def record(self, latency_ms: float) -> None:
        """ Record a latency measurement """
        self._latencies.append(latency_ms)

    def percentiles(self) -> dict:
        """ Returns stats including p50, p95 and p99 latencies """
        if not self._latencies:
            return {
                "p50": None,
                "p95": None,
                "p99": None,
                "count": 0
            }

        arr = np.array(self._latencies)

        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "mean": round(float(np.mean(arr)), 2), 
            "count": len(self._latencies)
        }        

    def reset(self) ->None:
        """ Resets the latency tracker """
        self._latencies.clear()