import feedparser
from typing import Iterator, List
from irkit.sources.base import BaseSource, Document

class NewsSource(BaseSource):
    """
    Fetches live news from RSS feeds.
    Default feeds: BBC News, CNN, Reuters.
    """

    DEFAULT_FEEDS = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/cnn_topstories.rss",
        "https://www.reutersagency.com/feed/?best-topics=technology&post_type=best",
    ]

    def __init__(self, feeds: List[str] = None):
        self.feeds = feeds or self.DEFAULT_FEEDS

    def load(self, max_docs: int = 1000) -> Iterator[Document]:
        count = 0
        for url in self.feeds:
            if count >= max_docs:
                break
            
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if count >= max_docs:
                    break
                
                # Use a combination of title and description for the text
                text = f"{entry.get('title', '')}. {entry.get('summary', '')}"
                
                yield Document(
                    id=entry.get('link', entry.get('id', f"news_{count}")),
                    title=entry.get('title', 'No Title'),
                    text=text,
                    metadata={
                        "url": entry.get('link'),
                        "published": entry.get('published'),
                        "source_feed": url
                    }
                )
                count += 1
