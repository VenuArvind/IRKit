import json
import redis
from typing import List, Dict, Optional 
from irkit.sources.base import Document 
from irkit.storage.base import BaseStorage
from irkit.core.sharding import ConsistentHashRing

class RedisStorage(BaseStorage):
    """
    Distributed document store backed by multiple Redis instances.
    Uses consistent hashing to shard documents across nodes.
    """

    def __init__(self, hosts: List[str] = None, port: int = 6379, db: int = 0):
        if hosts is None:
            hosts = ["localhost"]
        
        self.clients = [redis.Redis(host=h, port=port, db=db, decode_responses=True) for h in hosts]
        self.ring = ConsistentHashRing(nodes=[f"{h}:{port}" for h in hosts])
        print(f"[RedisStorage] Connected to {len(self.clients)} Redis shard(s).")

    def _get_client(self, doc_id: str):
        node = self.ring.get_node(doc_id)
        # Find index of node in the original hosts list
        for i, client in enumerate(self.clients):
            if f"{client.connection_pool.connection_kwargs['host']}:{client.connection_pool.connection_kwargs['port']}" == node:
                return client
        return self.clients[0]

    def save(self, document: Document) -> None:
        client = self._get_client(document.id)
        doc_dict = {
            "id": document.id,
            "title": document.title,
            "text": document.text,
            "metadata": json.dumps(document.metadata)
        }
        client.hset("irkit:docs", document.id, json.dumps(doc_dict))
    
    def get(self, doc_id: str) -> Optional[Document]:
        client = self._get_client(doc_id)
        data = client.hget("irkit:docs", doc_id)
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
        docs = []
        for client in self.clients:
            all_data = client.hgetall("irkit:docs")
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
        return sum(client.hlen("irkit:docs") for client in self.clients)
    
    def clear(self) -> None:
        for client in self.clients:
            client.delete("irkit:docs")
        