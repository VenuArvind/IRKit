from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

@dataclass
class Document:
    """ A single document in the index"""


    id: str
    title: str
    text: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    

class BaseSource(ABC):
    """Abstract Base class for all data sources"""
    
    @abstractmethod
    def load(self, max_docs: int = 1000) ->Iterator[Document]:
        """ Yield documents from the source one at a time """

        raise NotImplementedError
        