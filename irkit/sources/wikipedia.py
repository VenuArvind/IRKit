import wikipediaapi
from typing import Iterator, List
from irkit.sources.base import BaseSource, Document

class WikipediaSource(BaseSource):
    """
    Fetches Wikipedia articles by title or category.
    """

    def __init__(self, titles: List[str] = None, category: str = None, language: str = "en"):
        self.titles = titles or []
        self.category = category
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="irkit/1.0 (https://github.com/VenuArvind/irkit)",
            language=language
        )

    def load(self, max_docs: int = 1000) -> Iterator[Document]:
        titles_to_fetch = list(self.titles)

        # If a category is given, crawl it for article titles
        if self.category:
            cat = self.wiki.page(f"Category:{self.category}")
            if cat.exists():
                for title in list(cat.categorymembers.keys())[:max_docs]:
                    titles_to_fetch.append(title)

        for title in titles_to_fetch[:max_docs]:
            page = self.wiki.page(title)
            if not page.exists():
                continue
            yield Document(
                id=title.replace(" ", "_").lower(),
                title=page.title,
                text=page.summary,
                metadata={"url": page.fullurl}
            )