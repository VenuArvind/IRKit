import json
import csv
from pathlib import Path
from typing import Iterator, Dict, Any, Union
from irkit.sources.base import BaseSource, Document

class FileSource(BaseSource):
    """
    Universal file loader supporting JSON, CSV, TXT, and MD.
    """

    def __init__(self, file_path: Union[str, Path], id_field: str = "id", title_field: str = "title", text_field: str = "text"):
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
        elif suffix in [".txt", ".md"]:
            yield from self._load_raw_text()
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json, .csv, .txt, or .md")

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

    def _load_raw_text(self) -> Iterator[Document]:
        """ Load a single .txt or .md file as a single document """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            yield Document(
                id=self.file_path.name,
                title=self.file_path.stem.replace("_", " ").title(),
                text=content,
                metadata={"file_size": self.file_path.stat().st_size, "extension": self.file_path.suffix}
            )

    def _map_to_document(self, item: Dict[str, Any], index: int) -> Document:
        return Document(
            id=str(item.get(self.id_field, index)),
            title=item.get(self.title_field, f"doc_{index}"),
            text=item.get(self.text_field, ""),
            metadata={k: v for k, v in item.items() if k not in [self.id_field, self.title_field, self.text_field]}
        )
