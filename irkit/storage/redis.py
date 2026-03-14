import json
import redis
from typing import List, Dict, Optional 
from irkit.sources.base import Document 
from irkit.storage.base import BaseStorage

class RedisStorage(BaseStorage):
    """
    Redis-based persistent storage for documents.
    Distributed scaling and persistence
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.hash_name = "irkit:docs"

    def save(self, document: Document) ->None:
        doc_dict = {
            "id": document.id,
            "title": document.title,
            "text": document.text,
            "metadata": json.dumps(document.metadata)
        }
        self.r.hset(self.hash_name, document.id, json.dumps(doc_dict))
    
    def get(self, doc_id: str) -> Optional[Document]:
        data = self.r.hget(self.hash_name, doc_id)
        if not data:
            return None
        
        doc_dict = json.loads(data)
        
        return Document(
            id=doc_dict["id"],
            title=doc_dict["title"],
            text=doc_dict["text"],
            metadata=json.loads(doc_dict["metadata"])
        )

    def get_all(self) -> List[Document]:
        all_data = self.r.hgetall(self.hash_name)
        docs = []
        for doc_id, data in all_data.items():
            doc_dict = json.loads(data)
            docs.append(
                Document(
                    id=doc_dict["id"],
                    title=doc_dict["title"],
                    text=doc_dict["text"],
                    metadata=json.loads(doc_dict["metadata"])
                )
            )
        return docs
    
    def count(self) -> int:
        return self.r.hlen(self.hash_name)
    
    def clear(self) -> None:
        self.r.delete(self.hash_name)
        