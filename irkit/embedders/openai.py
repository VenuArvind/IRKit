import os
import numpy as np 
from typing import List
from openai import OpenAI
from irkit.embedders.base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """ Embeddings using OpenAI API """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        test_vec = self.embed(["test"])
        self._dimension = test_vec.shape[1]

    def embed(self, texts: List[str]) -> np.ndarray:
        """ Embed a list of texts """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return np.array([e.embedding for e in response.data])

    @property
    def dimension(self) ->int:
        return self._dimension
        