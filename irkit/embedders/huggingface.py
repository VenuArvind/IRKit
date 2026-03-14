from typing import List
import numpy as np 
from sentence_transformers import SentenceTransformer
from irkit.embedders.base import BaseEmbedder

class HuggingFaceEmbedder(BaseEmbedder):
    """ Embeddings using HuggingFace Sentence Transformers """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """ Generate embeddings for a list of texts """
        
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        return self._dimension