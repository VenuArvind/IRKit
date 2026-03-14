from datasets import load_dataset
from typing import Iterator
from irkit.sources.base import BaseSource, Document

class ArxivSource(BaseSource):
    """ Loads papers from ArXiv (using gfissore/arxiv-abstracts-2021) """
    
    def __init__(self, category_filter: str = None, max_docs: int = 1000):
        self.category_filter = category_filter
        self.default_max_docs = max_docs
        
    def load(self, max_docs: int = None) -> Iterator[Document]:
        limit = max_docs or self.default_max_docs
        # Using a reliable Parquet-based dataset that includes full metadata
        dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

        count = 0
        for item in dataset:
            if count >= limit:
                break
            
            # Re-enable category filtering since this dataset supports it
            if self.category_filter:
                categories = item.get("categories", "")
                if self.category_filter not in categories:
                    continue
            
            yield Document(
                id=item["id"],
                title=item["title"].strip(),
                text=item["abstract"].strip(),
                metadata={
                    "authors": item.get("authors", "").strip(),
                    "categories": item.get("categories", ""),
                    "url": f"https://arxiv.org/abs/{item['id']}"
                }
            )
            count += 1