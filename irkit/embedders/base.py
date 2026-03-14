from abc import ABC, abstractmethod
import numpy as np 
from typing import List

class BaseEmbedder(ABC):
    """ Abstract Base class for all embedders """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """ Embed a list of texts into a 2D numpy array of embeddings """
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        """ Embed a single query string into a 1D numpy array """
        return self.embed([query])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """ Return the dimension of the embeddings """
        raise NotImplementedError
    