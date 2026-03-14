import os
from pathlib import Path
from irkit.sources.arxiv import ArXivSource
from irkit.sources.wikipedia import WikipediaSource
from irkit.sources.news import NewsSource
from irkit.sources.file import FileSource
from irkit.sources.directory import DirectorySource
from irkit.sources.web import WebSource

def get_source(source_input: str, **kwargs):
    """
    Smart factory to resolve data sources from string input.
    """
    
    # 1. Check if it's a URL
    if source_input.startswith(("http://", "https://")):
        # If it ends with a comma-separated list or just one URL
        urls = [u.strip() for u in source_input.split(",")]
        return WebSource(urls=urls)
    
    # 2. Check if it's a local path
    path = Path(source_input)
    if path.exists():
        if path.is_dir():
            return DirectorySource(directory_path=str(path))
        else:
            return FileSource(file_path=str(path))
            
    # 3. Built-in sources
    source_lower = source_input.lower()
    if source_lower == "arxiv":
        return ArXivSource(**kwargs)
    elif source_lower == "wikipedia":
        return WikipediaSource(**kwargs)
    elif source_lower == "news":
        return NewsSource()
        
    raise ValueError(f"Unknown source or invalid path: {source_input}")
