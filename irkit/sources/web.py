import httpx
import re
from typing import Iterator, List, Optional
from irkit.sources.base import BaseSource, Document

class WebSource(BaseSource):
    """
    Indexes content from a list of URLs.
    Simple HTML-to-text conversion (removes tags).
    """

    def __init__(self, urls: List[str]):
        self.urls = urls

    def load(self, max_docs: int = 100) -> Iterator[Document]:
        count = 0
        with httpx.Client(timeout=10.0) as client:
            for url in self.urls:
                if count >= max_docs:
                    break
                
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    
                    html = response.text
                    # Simple regex to strip tags and scripts (fast, no heavy dependencies)
                    clean_text = self._strip_html(html)
                    
                    # Extract title
                    title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
                    title = title_match.group(1) if title_match else url
                    
                    yield Document(
                        id=url,
                        title=title,
                        text=clean_text,
                        metadata={"url": url, "status": response.status_code}
                    )
                    count += 1
                except Exception as e:
                    print(f"Error fetching {url}: {e}")

    def _strip_html(self, html: str) -> str:
        # Remove script and style elements
        html = re.sub(r'<(script|style).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all tags
        text = re.sub(r'<.*?>', ' ', html)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
