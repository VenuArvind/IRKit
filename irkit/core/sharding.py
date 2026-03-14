import hashlib
from typing import List, Optional 

class ConsistentHashRing:
    """
    Implements a consistent hashing ring for distributing keys across multiple shards.
    """

    def __init__(self, nodes: List[str], replicas: int = 100):
        """
        Args:
            nodes: List of node identifiers (e.g. ['redis-1', 'redis-2'])
            replicas: 'Virtual nodes' to ensure data is spread evenly
        """

        self.nodes = nodes
        self.replicas = replicas 
        self._ring = {}
        self._sorted_keys = []

        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) ->None:
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            h = self._hash(virtual_key)
            self._ring[h] = node
            self._sorted_keys.append(h)
        self._sorted_keys.sort()

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        
        h = self._hash(key)
        for position in self._sorted_keys:
            if h <= position:
                return self._ring[position]
        
        return self._ring[self._sorted_keys[0]]
            