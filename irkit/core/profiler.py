import time
import resource
import tracemalloc
from functools import wraps
from typing import Any, Callable

# Global storage for the last run metrics to be picked up by the API
last_metrics = {
    "duration": 0,
    "user_cpu": 0,
    "sys_cpu": 0,
    "peak_heap_mb": 0,
    "peak_rss_mb": 0
}

class EngineProfiler:
    """
    A low-overhead profiling harness for monitoring system resources.
    Can be used as a decorator or context manager.
    """
    def __init__(self, name: str = "operation"):
        self.name = name

    def __enter__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.start_time = time.perf_counter()
        self.start_resources = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.end_resources = resource.getrusage(resource.RUSAGE_SELF)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self._log_metrics(peak)

    def _log_metrics(self, peak_memory: int):
        duration = self.end_time - self.start_time
        user_cpu = self.end_resources.ru_utime - self.start_resources.ru_utime
        sys_cpu = self.end_resources.ru_stime - self.start_resources.ru_stime
        
        # ru_maxrss is in bytes on macOS, but can be KB on Linux
        # We assume macOS for self-consistent reporting
        peak_rss = self.end_resources.ru_maxrss / 1024 # KB to MB or similar
        
        # Store globally for API access
        global last_metrics
        last_metrics.update({
            "duration": duration,
            "user_cpu": user_cpu,
            "sys_cpu": sys_cpu,
            "peak_heap_mb": peak_memory / 1024 / 1024,
            "peak_rss_mb": peak_rss / 1024
        })

        print(f"\n🚀 --- Profiler Report: {self.name.upper()} ---")
        print(f"⏱️  Duration:         {duration:.6f} s")
        print(f"💻 CPU Time (User):   {user_cpu:.6f} s")
        print(f"🖥️  CPU Time (System): {sys_cpu:.6f} s")
        print(f"🧠 Peak Heap Memory:  {peak_memory / 1024 / 1024:.2f} MB")
        print(f"🏠 Peak RSS Memory:   {peak_rss / 1024:.2f} MB")
        print("-" * (26 + len(self.name)))

def profile_engine(name: str):
    """ Decorator wrapper for EngineProfiler """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with EngineProfiler(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
