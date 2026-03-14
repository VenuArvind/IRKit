from typing import List, Optional
from irkit.sources.base import BaseSource, Document
from irkit.rankers.base import BaseRanker, SearchResult
from irkit.storage.base import BaseStorage

class IndexEngine:
    """
    The orchestrator that ties everything together.
    Source -> Storage -> Ranker pipeline
    """

    def __init__(self, ranker: BaseRanker, storage: BaseStorage):
        self.ranker = ranker
        self.storage = storage

    def index(self, source: BaseSource, max_docs: int = 1000) -> None:
        """ Fetch documents from source, save to storage, and then index to ranker """
        print(f"[IndexEngine] Starting indexing from {source.__class__.__name__}...")

        docs_to_index = []

        for doc in source.load(max_docs=max_docs):
            self.storage.save(doc)
            docs_to_index.append(doc.__dict__)

        self.ranker.index(docs_to_index)

        print(f"[IndexEngine] Indexed {len(docs_to_index)} documents")
    
    def search(self, query: str, top_k : int = 10) -> List[SearchResult]:
        """ Search the index and return top k results """
        return self.ranker.search(query, top_k)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """ Retrieve a document from storage by its ID """
        return self.storage.get(doc_id)
        