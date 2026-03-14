from abc import ABC, abstractmethod
from typing import List, Optional
from irkit.sources.base import Document

class BaseStorage(ABC):
    """ Abstract Base class for all storage backends """
    
    @abstractmethod
    def save(self, document: Document) -> None:
        """ Save a document to the storage """
        raise NotImplementedError

    @abstractmethod
    def get(self, doc_id: str) -> Optional[Document]:
        """ Get a document from the storage """
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> List[Document]:
        """ Get all documents from the storage """
        raise NotImplementedError
    
    @abstractmethod
    def count(self) -> int:
        """ Count the number of documents in the storage """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """ Clear the storage """
        raise NotImplementedError