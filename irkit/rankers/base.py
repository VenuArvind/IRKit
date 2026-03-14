from abc import ABC, abstractmethod 
from dataclasses import dataclass 
from typing import List

@dataclass
class SearchResult:
    """ Result of a single document search """
    
    doc_id: str
    score: float
    title: str
    snippet: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseRanker(ABC):
    """ Base class for all rankers """
    
    @abstractmethod
    def index(self, documents: List[dict]) -> None:
        """ Index a list of documents """
        
        raise NotImplementedError
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """ Search for documents similar to the query """
        
        raise NotImplementedError
