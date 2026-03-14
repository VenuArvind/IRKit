from typing import List, Optional, Dict 
from irkit.sources.base import Document
from irkit.storage.base import BaseStorage

class InMemoryStorage(BaseStorage):
    """
    In-memory document storage
    Fast, but not persistent
    """

    def __init__(self):
        self._docs: Dict[str, Document] = {}
        
    def save(self, document: Document) -> None:
        self._docs[document.id] = document

    def get(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    def get_all(self) -> List[Document]:
        return list(self._docs.values())
    
    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()