import asyncio
import httpx
import logging
from typing import List, Dict, Optional
from .base import AbstractScraperClient
from ..config import config, is_valid_key

logger = logging.getLogger(__name__)

class FirecrawlScrapeClient(AbstractScraperClient):
    """Client for Firecrawl v1 scrape API with markdown extraction and fault tolerance."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = None):
        self.api_key = api_key or config.FIRECRAWL_API_KEY
        self.timeout = timeout or config.RESEARCH_TIMEOUT_SECONDS
        self.api_url = config.FIRECRAWL_SCRAPE_URL
        # Cache scraped markdown by URL
        self._cache: Dict[str, str] = {}

    def is_configured(self) -> bool:
        return is_valid_key(self.api_key)

    async def scrape_url(self, url: str) -> Optional[str]:
        """Scrape a single webpage and return cleaned Markdown text."""
        if not is_valid_key(self.api_key):
            logger.warning("[FIRECRAWL_CLIENT] Valid FIRECRAWL_API_KEY is not configured.")
            return None

        clean_url = url.strip()
        if not clean_url or not clean_url.startswith("http"):
            return None

        if clean_url in self._cache:
            return self._cache[clean_url]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "url": clean_url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)

                if response.status_code == 429:
                    logger.error(f"[FIRECRAWL_RATE_LIMIT] Rate limit hit while scraping: {clean_url}")
                    return None
                elif response.status_code in (401, 403):
                    logger.error(f"[FIRECRAWL_AUTH_FAIL] Firecrawl auth failed (HTTP {response.status_code})")
                    return None

                response.raise_for_status()
                data = response.json()

                if not data.get("success", False):
                    err = data.get("error", "Unknown Firecrawl error")
                    logger.warning(f"[FIRECRAWL_PAGE_FAIL] Could not scrape {clean_url}: {err}")
                    return None

                markdown = data.get("data", {}).get("markdown", "")
                if markdown:
                    # Truncate overly massive pages to keep token usage within budget
                    if len(markdown) > config.MAX_CONTENT_CHARS_PER_PAGE:
                        markdown = markdown[:config.MAX_CONTENT_CHARS_PER_PAGE] + "\n\n...[Content truncated for length]..."
                    self._cache[clean_url] = markdown
                    return markdown

                return None

        except httpx.TimeoutException:
            logger.warning(f"[FIRECRAWL_TIMEOUT] Timed out scraping URL: {clean_url}")
            return None
        except Exception as e:
            logger.warning(f"[FIRECRAWL_ERROR] Failed to scrape {clean_url}: {e}")
            return None

    async def scrape_batch(self, urls: List[str], max_pages: int = None) -> Dict[str, Optional[str]]:
        """Concurrently scrape unique URLs up to configured limit with graceful per-page fallback."""
        limit = max_pages or config.FIRECRAWL_MAX_PAGES
        unique_urls = list(dict.fromkeys([u.strip() for u in urls if u and u.strip().startswith("http")]))[:limit]

        if not unique_urls or not is_valid_key(self.api_key):
            return {u: None for u in unique_urls}

        tasks = [self.scrape_url(u) for u in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_map: Dict[str, Optional[str]] = {}
        for url, res in zip(unique_urls, results):
            if isinstance(res, Exception):
                logger.warning(f"[FIRECRAWL_BATCH_EXCEPTION] Error on {url}: {res}")
                scraped_map[url] = None
            else:
                scraped_map[url] = res

        return scraped_map
