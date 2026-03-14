import json
import csv
from pathlib import Path
from typing import Iterator, Dict, Any
from irkit.sources.base import BaseSource, Document

class CustomSource(BaseSource):
    """
    Loads documents from local JSON or CSV files.
    """

    def __init__(self, file_path: str, id_field: str = "id", title_field: str = "title", text_field: str = "text"):
        self.file_path = Path(file_path)
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self, max_docs: int = 1000) -> Iterator[Document]:
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".json":
            yield from self._load_json(max_docs)
        elif suffix == ".csv":
            yield from self._load_csv(max_docs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")

    def _load_json(self, max_docs: int) -> Iterator[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            
            for i, item in enumerate(data[:max_docs]):
                yield self._map_to_document(item, i)

    def _load_csv(self, max_docs: int) -> Iterator[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_docs:
                    break
                yield self._map_to_document(row, i)

    def _map_to_document(self, item: Dict[str, Any], index: int) -> Document:
        return Document(
            id=str(item.get(self.id_field, index)),
            title=item.get(self.title_field, f"Document {index}"),
            text=item.get(self.text_field, ""),
            metadata={k: v for k, v in item.items() if k not in [self.id_field, self.title_field, self.text_field]}
        )
