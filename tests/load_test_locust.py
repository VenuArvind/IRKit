"""
Locust load test script for IRKit search API.

Usage:
  locust -f tests/load_test_locust.py --headless -u 20 -r 5 -t 30s --host http://localhost:8000
"""

import random
from locust import HttpUser, task, between

QUERIES = [
    "transformer architecture",
    "deep learning optimization",
    "quantum computing algorithms",
    "information retrieval benchmarks",
    "consistent hashing Redis",
    "dense vector embeddings",
    "BM25 keyword search",
    "reciprocal rank fusion",
    "fastapi async endpoints",
    "cross encoder reranking",
]

class SearchUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(4)
    def test_hybrid_search(self):
        query = random.choice(QUERIES)
        self.client.get(f"/search?q={query}&top_k=10&rerank=false", name="/search [hybrid]")

    @task(1)
    def test_reranked_search(self):
        query = random.choice(QUERIES)
        self.client.get(f"/search?q={query}&top_k=5&rerank=true", name="/search [rerank]")

