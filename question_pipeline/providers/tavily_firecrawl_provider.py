"""
Real research provider: Tavily (search/discovery) -> Firecrawl (extraction).

Reuses the existing web_research HTTP clients (TavilySearchClient,
FirecrawlScrapeClient) instead of reimplementing the Tavily/Firecrawl API
calls a second time. Those clients already handle their own HTTP errors,
timeouts, rate limits, and auth failures internally (returning [] / None on
failure rather than raising) -- this provider's job is to turn that into an
explicit, honest outcome: either a genuinely-sourced ResearchOutput, or a
raised exception. It never fabricates research content to paper over a
failed search or a failed scrape.

Flow:
    Tavily search (discovery)
        -> rank by Tavily's own relevance_score, keep the top N
        -> Firecrawl scrape (extraction) of those URLs
        -> ResearchOutput built ONLY from what was actually retrieved
           (falls back to Tavily's own snippet text for a source only if
           Firecrawl could not extract it; raises if NEITHER is available
           for any source)
"""
import hashlib
from typing import Optional, List
from .base import ResearchProvider
from ..models import ResearchOutput
from ..config import config, is_valid_key
from ..governor import ProviderBlockedException

DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_MAX_SCRAPE_PAGES = 3

# Bounded, generic fallback query templates used ONLY for COMPANY research,
# and ONLY when the primary query yields no usable Tavily results. Built
# purely from the company name via fixed suffixes -- no company-specific
# hardcoding, so the same list applies identically to any company. Kept
# short and fixed-length so a company that never returns anything still
# fails after a small, predictable number of attempts (never an unbounded
# retry loop).
COMPANY_FALLBACK_QUERY_SUFFIXES = [
    "company overview products and technology",
    "engineering careers technical blog culture",
]


class TavilyFirecrawlResearchProvider(ResearchProvider):
    """ResearchProvider implementation backed by Tavily + Firecrawl."""

    def __init__(self, tavily_client=None, firecrawl_client=None):
        # Lazy import so importing this module never requires the
        # web_research package to already be on sys.path unless this
        # provider is actually instantiated.
        if tavily_client is None or firecrawl_client is None:
            from web_research.clients.tavily_client import TavilySearchClient
            from web_research.clients.firecrawl_client import FirecrawlScrapeClient
        self._tavily = tavily_client if tavily_client is not None else TavilySearchClient(
            api_key=config.TAVILY_API_KEY
        )
        self._firecrawl = firecrawl_client if firecrawl_client is not None else FirecrawlScrapeClient(
            api_key=config.FIRECRAWL_API_KEY
        )

    @property
    def provider_name(self) -> str:
        return "tavily_firecrawl"

    def _ensure_configured(self) -> None:
        if not is_valid_key(config.TAVILY_API_KEY):
            raise ProviderBlockedException(
                provider="tavily_firecrawl", model="tavily",
                message="TAVILY_API_KEY is not configured or is a placeholder in .env.",
            )
        if not is_valid_key(config.FIRECRAWL_API_KEY):
            raise ProviderBlockedException(
                provider="tavily_firecrawl", model="firecrawl",
                message="FIRECRAWL_API_KEY is not configured or is a placeholder in .env.",
            )

    @staticmethod
    def _build_raw_summary(subject: str, sources: List, extraction_status: str) -> str:
        parts = [
            f"Web research dossier for '{subject}' "
            f"({extraction_status}, {len(sources)} source(s))."
        ]
        for s in sources:
            body = s.scraped_content or s.snippet or ""
            parts.append(f"\n--- Source: {s.title} ({s.url}) ---\n{body}")
        return "\n".join(parts)

    async def _research_single(self, subject: str, query: str) -> ResearchOutput:
        """One full discovery -> extraction -> ResearchOutput attempt for a
        single query. This is the exact original _research() body, unchanged
        -- role research still calls this directly, with no fallback, so its
        behavior (including exact error messages) is byte-for-byte preserved."""
        self._ensure_configured()

        # Stage 1: Tavily discovery
        sources = await self._tavily.search(query, max_results=DEFAULT_MAX_SEARCH_RESULTS)
        if not sources:
            raise RuntimeError(
                f"Tavily search returned no usable results for '{subject}' "
                f"(query: '{query}'). Cannot proceed without discovered sources."
            )

        # Stage 2: select the highest-quality sources (Tavily's own relevance ranking)
        ranked = sorted(sources, key=lambda s: (s.relevance_score or 0.0), reverse=True)
        selected = ranked[:DEFAULT_MAX_SCRAPE_PAGES]
        urls = [s.url for s in selected]

        # Stage 3: Firecrawl extraction of the selected sources
        scraped_map = await self._firecrawl.scrape_batch(urls, max_pages=DEFAULT_MAX_SCRAPE_PAGES)
        for s in selected:
            s.scraped_content = scraped_map.get(s.url)

        fully_extracted = [s for s in selected if s.scraped_content]
        if fully_extracted:
            enriched = fully_extracted
            # Sources Firecrawl couldn't reach but that still have a Tavily
            # snippet are kept too (explicit degraded-but-real fallback, not
            # fabrication -- the text is genuinely Tavily's own snippet).
            enriched += [s for s in selected if not s.scraped_content and s.snippet]
            extraction_status = "FULL_EXTRACTION" if len(fully_extracted) == len(selected) else "PARTIAL_EXTRACTION"
        else:
            # Firecrawl failed on every selected source. Only proceed on
            # Tavily's own snippets if at least one exists; otherwise there
            # is nothing real to build a ResearchOutput from.
            snippet_only = [s for s in selected if s.snippet]
            if not snippet_only:
                raise RuntimeError(
                    f"Firecrawl extraction failed for all {len(selected)} selected source(s) "
                    f"for '{subject}', and no Tavily snippet text is available as a fallback. "
                    f"Refusing to fabricate research content."
                )
            enriched = snippet_only
            extraction_status = "SNIPPET_FALLBACK"

        raw_summary = self._build_raw_summary(subject, enriched, extraction_status)
        source_references = [f"{s.title} ({s.url})" for s in enriched]
        source_hash = hashlib.sha256(raw_summary.encode("utf-8")).hexdigest()

        return ResearchOutput(
            role=subject,
            technologies=[],
            engineering_practices=[],
            responsibilities=[],
            interview_topics=[s.title for s in enriched][:5],
            trends=[],
            raw_summary=raw_summary,
            source_references=source_references,
            source_hash=source_hash,
            actual_provider="tavily_firecrawl",
        )

    @staticmethod
    def _build_company_fallback_queries(company: str) -> List[str]:
        """Bounded (currently 2), generic alternative discovery queries for
        COMPANY research only -- derived purely from the company name via
        fixed templates. No per-company special-casing: this produces the
        same shape of fallback list for any company."""
        return [f"{company} {suffix}" for suffix in COMPANY_FALLBACK_QUERY_SUFFIXES]

    async def _research_with_fallback(self, subject: str, primary_query: str, fallback_queries: List[str]) -> ResearchOutput:
        """Tries primary_query first; if it raises the 'no usable results'
        RuntimeError, tries each fallback query in order, stopping at the
        first one that produces a genuine ResearchOutput. Every attempt goes
        through the exact same _research_single() pipeline as the primary
        query -- same Tavily discovery, same Firecrawl extraction, same
        snippet-fallback, same fail-closed refusal to fabricate. Sources are
        never merged across queries (only the first successful query's
        sources are used), so no cross-query duplicate-source risk is
        introduced. Bounded to len(fallback_queries) extra attempts -- never
        an unbounded or open-ended retry loop. If every attempt fails, raises
        one RuntimeError summarizing all attempts so the caller's FAILED job
        reason stays fully diagnostic (never silently swallowed)."""
        attempts = [primary_query] + list(fallback_queries)
        errors = []
        for idx, q in enumerate(attempts):
            try:
                result = await self._research_single(subject, q)
                if idx > 0:
                    print(f"[RESEARCH] '{subject}': primary query returned no usable results; "
                          f"fallback query {idx}/{len(fallback_queries)} succeeded: '{q}'")
                return result
            except RuntimeError as e:
                errors.append(f"[attempt {idx + 1}/{len(attempts)}] '{q}' -> {e}")
                continue
        raise RuntimeError(
            f"All {len(attempts)} bounded research queries for '{subject}' produced no usable results "
            f"(1 primary + {len(fallback_queries)} fallback, per the bounded generic company-fallback "
            f"strategy). Attempts: " + " || ".join(errors)
        )

    async def research_role(self, role: str) -> ResearchOutput:
        query = f"{role} technical interview topics skills responsibilities current best practices"
        return await self._research_single(role, query)

    async def research_company(self, company: str) -> ResearchOutput:
        primary_query = f"{company} engineering technology stack architecture technical interview process"
        fallback_queries = self._build_company_fallback_queries(company)
        result = await self._research_with_fallback(company, primary_query, fallback_queries)
        return result.model_copy(update={"company": company})
