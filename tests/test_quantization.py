import unittest
import numpy as np
from irkit.core.quantization import ScalarQuantizer, ProductQuantizer, asymmetric_distance

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.num_vectors = 100
        self.data = np.random.randn(self.num_vectors, self.dim).astype(np.float32)

    def test_sq8_scaffolding(self):
        sq = ScalarQuantizer()
        quantized = sq.quantize(self.data)
        # TODO: Once implemented, assert quantized.dtype == np.uint8
        # dequantized = sq.dequantize(quantized)
        pass

    def test_pq_scaffolding(self):
        pq = ProductQuantizer(num_subspaces=8)
        # pq.train(self.data)
        # codes = pq.encode(self.data)
        # decoded = pq.decode(codes)
        pass

if __name__ == "__main__":
    unittest.main()
