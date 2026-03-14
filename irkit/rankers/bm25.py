from typing import List
from rank_bm25 import BM25Okapi
from irkit.rankers.base import BaseRanker, SearchResult

class BM25Ranker(BaseRanker):
    """ BM25 Ranker """
    
    def __init__(self):
        self.bm25 = None
        self.doc_store = []

    def index(self, documents: List[dict]):
        """ Index documents for BM25 """
        self.doc_store = documents
        tokenized_docs = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 Indexing Completed for {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self.bm25:
            print("BM25 not indexed. Please index documents first.")
            return []
        
        tokenized_query = query.lower().split()
        
        # Get Scores for docs
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if score > 0:
                doc = self.doc_store[i]
                results.append(SearchResult(
                    doc_id = doc['id'],
                    score = float(score),
                    title = doc['title'],
                    snippet = doc['text'][0:200] + "...",
                    metadata = doc.get('metadata',{})
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]