import numpy as np
from typing import List, Tuple, Optional
from scipy.cluster.vq import kmeans2, vq

class ScalarQuantizer:
    """
    SQ8 Quantization.
    Reduces float32 (4 bytes) to uint8 (1 byte).
    """
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculates min/max per dimension, scales vectors to 0-255, 
        and casts to uint8.
        """
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)
        
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0

        normalized = (vectors - self.min_vals) / range_vals
        quantized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

        return quantized

    def dequantize(self, quantized_vectors: np.ndarray) -> np.ndarray:
        """
        Converts uint8 back to float32 using stored boundaries.
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Quantizer must be primed with 'quantize' first.")
        
        range_vals = self.max_vals - self.min_vals
        dequantized = (quantized_vectors.astype(np.float32) / 255.0) * range_vals + self.min_vals
        
        return dequantized

class ProductQuantizer:
    """
    Product Quantization (PQ).
    Divide vectors into sub-spaces and quantize each sub-space.
    """
    def __init__(self, num_subspaces: int = 8, cluster_bits: int = 8):
        self.num_subspaces = num_subspaces
        self.cluster_bits = cluster_bits
        self.k = 2 ** cluster_bits
        self.codebooks = [] 

    def train(self, vectors: np.ndarray):
        """
        Divide vectors into sub-spaces and run K-Means on each.
        """
        num_docs, dim = vectors.shape
        sub_dim = dim // self.num_subspaces

        self.codebooks = []

        for i in range(self.num_subspaces):
            start_col = i * sub_dim
            end_col = (i+1) * sub_dim
            sub_vectors = vectors[:, start_col:end_col]

            centroids, _ = kmeans2(sub_vectors, self.k, minit='points', iter=20)
            self.codebooks.append(centroids)

        self.codebooks = np.array(self.codebooks)
        print(f" [PQ] Training completed. Generated {self.num_subspaces} codebooks, each with {self.k} centroids.")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Maps every sub-vector to the index of its nearest centroid.
        """
        num_docs, dim = vectors.shape
        sub_dim = dim // self.num_subspaces

        codes = np.zeros((num_docs, self.num_subspaces), dtype=np.uint8)

        for i in range(self.num_subspaces):
            start_col = i * sub_dim
            end_col = (i+1) * sub_dim
            sub_vectors = vectors[:, start_col:end_col]

            indices, _ = vq(sub_vectors, self.codebooks[i])
            codes[:, i] = indices

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstructs vectors from centroid indices.
        """
        num_docs, num_subspaces = codes.shape
        sub_dim = self.codebooks.shape[2]
        dim = num_subspaces * sub_dim

        reconstructed = np.zeros((num_docs, dim), dtype=np.float32)

        for i in range(num_subspaces):
            start_col = i * sub_dim
            end_col = (i+1) * sub_dim
            
            centroid_indices = codes[:, i]
            reconstructed[:, start_col:end_col] = self.codebooks[i][centroid_indices]

        return reconstructed

def asymmetric_distance(query: np.ndarray, quantized_db: np.ndarray, quantizer: any, method: str = "sq8") -> np.ndarray:
    """
    Computes distances between a float32 query and quantized database.
    """
    if method == "sq8":
        db_floats = quantizer.dequantize(quantized_db)
        return np.dot(db_floats, query.flatten())

    elif method == "pq":
        num_docs, num_subspaces = quantized_db.shape
        sub_dim = quantizer.codebooks.shape[2]
        query_vec = query.flatten()

        distance_table = np.zeros((num_subspaces, quantizer.k))

        for i in range(num_subspaces):
            sub_query = query_vec[i*sub_dim : (i+1)*sub_dim]
            diffs = np.dot(quantizer.codebooks[i], sub_query)
            distance_table[i, :] = diffs

        scores = np.zeros(num_docs)
        for i in range(num_subspaces):
            scores += distance_table[i, quantized_db[:, i]]
        
        return scores

    return np.array([])