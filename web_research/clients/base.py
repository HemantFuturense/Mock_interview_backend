from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..models import SourceItem

class AbstractSearchClient(ABC):
    """Abstract interface for web search providers (e.g. Tavily)."""
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        """Perform search and return candidate sources."""
        pass

    def is_configured(self) -> bool:
        """Check whether the client credentials and settings are configured."""
        return True

class AbstractScraperClient(ABC):
    """Abstract interface for webpage scraping providers (e.g. Firecrawl)."""
    @abstractmethod
    async def scrape_url(self, url: str) -> Optional[str]:
        """Scrape a single URL and return clean Markdown content."""
        pass

    @abstractmethod
    async def scrape_batch(self, urls: List[str], max_pages: int = 3) -> Dict[str, Optional[str]]:
        """Scrape multiple URLs with concurrency and rate limits."""
        pass

    def is_configured(self) -> bool:
        """Check whether the scraper credentials and settings are configured."""
        return True

class AbstractLLMClient(ABC):
    """Abstract interface for synthesis and reasoning (e.g. Google Gemini)."""
    @abstractmethod
    async def synthesize(self, question: str, sources: List[SourceItem]) -> str:
        """Analyze gathered sources and produce a grounded, cited research answer."""
        pass

    def is_configured(self) -> bool:
        """Check whether the LLM credentials and settings are configured."""
        return True
