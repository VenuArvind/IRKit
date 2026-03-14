import pytest
from irkit.core.sharding import ConsistentHashRing

def test_consistent_hashing():
    """ Verify the same key always goes to the same node """
    ring = ConsistentHashRing(nodes=["node-a:6379", "node-b:6380"])
    
    node1 = ring.get_node("doc-abc")
    node2 = ring.get_node("doc-abc")
    assert node1 == node2

def test_distribution():
    """ Verify all nodes are used for a variety of keys """
    ring = ConsistentHashRing(nodes=["a", "b", "c"])
    used_nodes = set()
    for i in range(300):
        used_nodes.add(ring.get_node(f"doc-{i}"))
    assert len(used_nodes) == 3   # all 3 shards should get some docs

def test_single_node():
    """ Verify a single node ring always returns that node """
    ring = ConsistentHashRing(nodes=["only-node"])
    assert ring.get_node("any-key") == "only-node"
