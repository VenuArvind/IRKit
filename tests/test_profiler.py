import unittest
import time
from irkit.core.profiler import EngineProfiler, profile_engine

class TestProfiler(unittest.TestCase):
    def test_context_manager(self):
        with EngineProfiler("test_ctx"):
            time.sleep(0.1)
            # Allocation to test memory tracking
            x = [i for i in range(1000000)]
            del x

    def test_decorator(self):
        @profile_engine("test_decorator")
        def dummy_heavy_work():
            time.sleep(0.1)
        
        dummy_heavy_work()

if __name__ == "__main__":
    unittest.main()
