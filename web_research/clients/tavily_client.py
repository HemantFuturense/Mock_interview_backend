import time
import httpx
import logging
from typing import List, Dict, Optional, Tuple
from .base import AbstractSearchClient
from ..models import SourceItem
from ..config import config, is_valid_key

logger = logging.getLogger(__name__)

class TavilySearchClient(AbstractSearchClient):
    """Client for Tavily Search API with caching and error resiliency."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = None):
        self.api_key = api_key or config.TAVILY_API_KEY
        self.timeout = timeout or config.RESEARCH_TIMEOUT_SECONDS
        self.api_url = config.TAVILY_SEARCH_URL
        # In-memory TTL cache: {normalized_query_with_count: (timestamp, List[SourceItem])}
        self._cache: Dict[str, Tuple[float, List[SourceItem]]] = {}

    def _normalize_key(self, query: str, max_results: int) -> str:
        return f"{query.strip().lower()}__max_{max_results}"

    def is_configured(self) -> bool:
        return is_valid_key(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        """Search the web via Tavily API and return structured sources."""
        if not is_valid_key(self.api_key):
            logger.warning("[TAVILY_CLIENT] Valid TAVILY_API_KEY is not configured.")
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        # Check in-memory cache
        cache_key = self._normalize_key(clean_query, max_results)
        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_items = self._cache[cache_key]
            if (now - cached_time) < config.CACHE_TTL_SECONDS:
                logger.info(f"[TAVILY_CACHE_HIT] Reusing cached search for: '{clean_query[:50]}'")
                return cached_items

        payload = {
            "api_key": self.api_key,
            "query": clean_query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 429:
                    logger.error("[TAVILY_ERROR] Tavily API rate limit exceeded (HTTP 429).")
                    return []
                elif response.status_code in (401, 403):
                    logger.error(f"[TAVILY_ERROR] Tavily authentication failed (HTTP {response.status_code}).")
                    return []

                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException:
            logger.error(f"[TAVILY_TIMEOUT] Tavily search timed out after {self.timeout}s.")
            return []
        except Exception as e:
            logger.error(f"[TAVILY_EXCEPTION] Failed to execute Tavily search: {e}")
            return []

        raw_results = data.get("results", [])
        sources: List[SourceItem] = []

        for item in raw_results:
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "Web Source").strip()
            content = (item.get("content") or "").strip()
            score = item.get("score")
            try:
                relevance = float(score) if score is not None else None
            except (ValueError, TypeError):
                relevance = None

            if url and url.startswith("http"):
                sources.append(SourceItem(
                    title=title,
                    url=url,
                    snippet=content,
                    relevance_score=relevance,
                    source_type="tavily_search"
                ))

        # Store in cache
        self._cache[cache_key] = (now, sources)
        return sources
