import os
from pathlib import Path
from typing import Iterator, List, Optional
from irkit.sources.base import BaseSource, Document
from irkit.sources.file import FileSource

class DirectorySource(BaseSource):
    """
    Recursively indexes files in a directory.
    """

    def __init__(self, directory_path: str, extensions: Optional[List[str]] = None):
        self.directory_path = Path(directory_path)
        self.extensions = extensions or [".txt", ".md", ".json", ".csv"]

        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory_path}")

    def load(self, max_docs: int = 1000) -> Iterator[Document]:
        count = 0
        for root, _, files in os.walk(self.directory_path):
            if count >= max_docs:
                break
                
            for file in files:
                if count >= max_docs:
                    break
                    
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.extensions:
                    try:
                        file_source = FileSource(file_path)
                        for doc in file_source.load():
                            yield doc
                            count += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
