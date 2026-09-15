"""
Tavily + Firecrawl ResearchProvider tests.

All external Tavily/Firecrawl calls are faked (per requirement: never call
the real APIs from unit tests). Fakes are plain duck-typed stand-ins for
web_research.clients.TavilySearchClient / FirecrawlScrapeClient, injected via
the provider's constructor -- no monkeypatching of the real HTTP clients, no
network access.
"""
import os
import io
import asyncio
import contextlib
import unittest
from typing import List, Dict, Optional

os.environ.setdefault("MOCK_MODE", "true")

from question_pipeline.config import config
from question_pipeline.governor import ProviderBlockedException
from web_research.models import SourceItem


def make_source(title, url, snippet="", relevance_score=0.9) -> SourceItem:
    return SourceItem(title=title, url=url, snippet=snippet, relevance_score=relevance_score, source_type="tavily_search")


class FakeTavilyClient:
    def __init__(
        self,
        results: Optional[List[SourceItem]] = None,
        raise_exc: Optional[Exception] = None,
        results_sequence: Optional[List[List[SourceItem]]] = None,
    ):
        """results_sequence, when given, returns a different result list per
        call (call 1 -> results_sequence[0], call 2 -> results_sequence[1],
        ...), falling back to [] once exhausted -- used to simulate a
        primary query finding nothing and a later fallback query finding
        something. When omitted, behavior is unchanged from before (every
        call returns the same fixed `results` list)."""
        self._results = results if results is not None else []
        self._raise_exc = raise_exc
        self._results_sequence = results_sequence
        self.search_calls = []

    async def search(self, query: str, max_results: int = 5) -> List[SourceItem]:
        self.search_calls.append((query, max_results))
        if self._raise_exc:
            raise self._raise_exc
        if self._results_sequence is not None:
            idx = len(self.search_calls) - 1
            return self._results_sequence[idx] if idx < len(self._results_sequence) else []
        return self._results

    def is_configured(self) -> bool:
        return True


class FakeFirecrawlClient:
    def __init__(self, content_by_url: Optional[Dict[str, Optional[str]]] = None, raise_exc: Optional[Exception] = None):
        self._content_by_url = content_by_url or {}
        self._raise_exc = raise_exc
        self.scrape_batch_calls = []

    async def scrape_batch(self, urls: List[str], max_pages: int = None) -> Dict[str, Optional[str]]:
        self.scrape_batch_calls.append((list(urls), max_pages))
        if self._raise_exc:
            raise self._raise_exc
        return {u: self._content_by_url.get(u) for u in urls}

    def is_configured(self) -> bool:
        return True


def make_provider(tavily=None, firecrawl=None):
    from question_pipeline.providers.tavily_firecrawl_provider import TavilyFirecrawlResearchProvider
    return TavilyFirecrawlResearchProvider(
        tavily_client=tavily if tavily is not None else FakeTavilyClient(),
        firecrawl_client=firecrawl if firecrawl is not None else FakeFirecrawlClient(),
    )


class KeyPatchMixin:
    """Temporarily overrides config.TAVILY_API_KEY / FIRECRAWL_API_KEY."""

    def patch_keys(self, tavily: str = "tvly-real-looking-test-key-000111", firecrawl: str = "fc-real-looking-test-key-000111"):
        self._orig_tavily = config.TAVILY_API_KEY
        self._orig_firecrawl = config.FIRECRAWL_API_KEY
        config.TAVILY_API_KEY = tavily
        config.FIRECRAWL_API_KEY = firecrawl
        self.addCleanup(self._restore_keys)

    def _restore_keys(self):
        config.TAVILY_API_KEY = self._orig_tavily
        config.FIRECRAWL_API_KEY = self._orig_firecrawl


class TestTavilyFirecrawlFlow(unittest.TestCase, KeyPatchMixin):
    def setUp(self):
        self.patch_keys()

    def test_A_tavily_results_are_passed_to_firecrawl(self):
        sources = [
            make_source("Kafka Architecture Deep Dive", "https://example.com/kafka", relevance_score=0.95),
            make_source("Partition Rebalancing Explained", "https://example.com/rebalance", relevance_score=0.80),
        ]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={
            "https://example.com/kafka": "Full extracted markdown about Kafka architecture.",
            "https://example.com/rebalance": "Full extracted markdown about partition rebalancing.",
        })
        provider = make_provider(tavily, firecrawl)

        asyncio.run(provider.research_role("Data Engineer"))

        self.assertEqual(len(tavily.search_calls), 1)
        self.assertEqual(len(firecrawl.scrape_batch_calls), 1)
        urls_sent_to_firecrawl = firecrawl.scrape_batch_calls[0][0]
        self.assertIn("https://example.com/kafka", urls_sent_to_firecrawl)
        self.assertIn("https://example.com/rebalance", urls_sent_to_firecrawl)

    def test_B_firecrawl_content_becomes_research_output(self):
        sources = [make_source("Kafka Architecture", "https://example.com/kafka", relevance_score=0.9)]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={
            "https://example.com/kafka": "Kafka partitions are the unit of parallelism; rebalancing moves ownership between consumers."
        })
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_role("Data Engineer"))

        self.assertEqual(result.actual_provider, "tavily_firecrawl")
        self.assertIn("Kafka partitions are the unit of parallelism", result.raw_summary)
        self.assertEqual(result.role, "Data Engineer")
        self.assertTrue(len(result.source_hash) > 0)

    def test_C_source_url_title_provenance_preserved(self):
        sources = [make_source("Kafka Architecture Deep Dive", "https://example.com/kafka", relevance_score=0.9)]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/kafka": "Extracted content."})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_role("Data Engineer"))

        self.assertEqual(len(result.source_references), 1)
        self.assertIn("Kafka Architecture Deep Dive", result.source_references[0])
        self.assertIn("https://example.com/kafka", result.source_references[0])

    def test_D_tavily_failure_is_handled_explicitly(self):
        tavily = FakeTavilyClient(results=[])  # simulates the real client's documented behavior: [] on any error
        firecrawl = FakeFirecrawlClient()
        provider = make_provider(tavily, firecrawl)

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(provider.research_role("Data Engineer"))
        self.assertIn("Tavily search returned no usable results", str(ctx.exception))
        # Firecrawl must never be called if Tavily found nothing to extract.
        self.assertEqual(len(firecrawl.scrape_batch_calls), 0)

    def test_E1_firecrawl_total_failure_with_no_snippet_fallback_raises(self):
        sources = [make_source("Kafka Architecture", "https://example.com/kafka", snippet="", relevance_score=0.9)]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/kafka": None})  # extraction failed
        provider = make_provider(tavily, firecrawl)

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(provider.research_role("Data Engineer"))
        self.assertIn("Firecrawl extraction failed", str(ctx.exception))

    def test_E2_firecrawl_failure_falls_back_to_tavily_snippet_not_fabrication(self):
        sources = [make_source(
            "Kafka Architecture", "https://example.com/kafka",
            snippet="Real Tavily snippet text about Kafka.", relevance_score=0.9,
        )]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/kafka": None})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_role("Data Engineer"))
        self.assertIn("Real Tavily snippet text about Kafka.", result.raw_summary)
        self.assertIn("SNIPPET_FALLBACK", result.raw_summary)

    def test_F_missing_tavily_key_is_blocked(self):
        config.TAVILY_API_KEY = ""
        provider = make_provider()
        with self.assertRaises(ProviderBlockedException) as ctx:
            asyncio.run(provider.research_role("Data Engineer"))
        self.assertIn("TAVILY_API_KEY", str(ctx.exception))

    def test_G_missing_firecrawl_key_is_blocked(self):
        config.FIRECRAWL_API_KEY = ""
        provider = make_provider()
        with self.assertRaises(ProviderBlockedException) as ctx:
            asyncio.run(provider.research_role("Data Engineer"))
        self.assertIn("FIRECRAWL_API_KEY", str(ctx.exception))

    def test_J_no_api_secrets_appear_in_output(self):
        real_looking_tavily_key = "tvly-SUPERSECRETVALUE-should-never-be-printed"
        real_looking_firecrawl_key = "fc-SUPERSECRETVALUE-should-never-be-printed"
        config.TAVILY_API_KEY = real_looking_tavily_key
        config.FIRECRAWL_API_KEY = real_looking_firecrawl_key

        sources = [make_source("Kafka Architecture", "https://example.com/kafka", relevance_score=0.9)]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/kafka": "Extracted content."})
        provider = make_provider(tavily, firecrawl)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = asyncio.run(provider.research_role("Data Engineer"))

        captured = buf.getvalue()
        self.assertNotIn(real_looking_tavily_key, captured)
        self.assertNotIn(real_looking_firecrawl_key, captured)
        self.assertNotIn(real_looking_tavily_key, result.raw_summary)
        self.assertNotIn(real_looking_firecrawl_key, result.raw_summary)
        self.assertNotIn(real_looking_tavily_key, str(result.model_dump()))
        self.assertNotIn(real_looking_firecrawl_key, str(result.model_dump()))


class TestCompanyResearchBoundedFallback(unittest.TestCase, KeyPatchMixin):
    """Regression tests for the bounded generic company-research fallback
    (fix for the real OpenAI COMPANY_RESEARCH FAILED job: Tavily's primary
    query returned zero results). Covers: primary success (no fallback
    triggered), primary-fails/fallback-succeeds, all-attempts-fail (still
    FAILED, not silently papered over), no cross-query duplicate sources,
    role research completely unaffected, and no per-company special-casing."""

    def setUp(self):
        self.patch_keys()

    def test_L_primary_company_query_succeeds_normally_no_fallback_used(self):
        sources = [make_source("Spotify Engineering Blog", "https://example.com/spotify-eng", relevance_score=0.9)]
        tavily = FakeTavilyClient(results=sources)
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/spotify-eng": "Real extracted content about Spotify engineering."})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_company("Spotify"))

        self.assertEqual(len(tavily.search_calls), 1, "primary query must succeed without any fallback attempt")
        self.assertEqual(result.company, "Spotify")
        self.assertIn("Real extracted content about Spotify engineering.", result.raw_summary)

    def test_M_primary_fails_first_fallback_succeeds(self):
        # Call 1 (primary query): no results. Call 2 (1st fallback query): results found.
        fallback_sources = [make_source("OpenAI Careers Page", "https://example.com/openai-careers", relevance_score=0.85)]
        tavily = FakeTavilyClient(results_sequence=[[], fallback_sources])
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/openai-careers": "Real extracted content about OpenAI careers/engineering."})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_company("OpenAI"))

        self.assertEqual(len(tavily.search_calls), 2, "must try exactly primary + 1 fallback, stopping at first success")
        self.assertEqual(result.company, "OpenAI")
        self.assertIn("Real extracted content about OpenAI careers/engineering.", result.raw_summary)
        # The fallback query must be derived generically from the company name.
        fallback_query_used = tavily.search_calls[1][0]
        self.assertIn("OpenAI", fallback_query_used)

    def test_N_primary_and_first_fallback_fail_second_fallback_succeeds(self):
        fallback2_sources = [make_source("Some Company Tech Blog", "https://example.com/tech-blog", relevance_score=0.7)]
        tavily = FakeTavilyClient(results_sequence=[[], [], fallback2_sources])
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/tech-blog": "Real extracted tech blog content."})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_company("Acme Corp"))

        self.assertEqual(len(tavily.search_calls), 3, "must try primary + both bounded fallbacks before succeeding")
        self.assertEqual(result.company, "Acme Corp")

    def test_O_all_bounded_attempts_fail_raises_and_stays_bounded(self):
        """Primary + all fallback queries return nothing -> RuntimeError raised
        (which the orchestrator's existing except Exception turns into a
        FAILED job, per orchestrator.py:558-560) -- never an infinite/
        unbounded retry loop, never fabricated content."""
        tavily = FakeTavilyClient(results=[])  # every call returns []
        firecrawl = FakeFirecrawlClient()
        provider = make_provider(tavily, firecrawl)

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(provider.research_company("Nonexistent Company XYZ"))

        # Bounded: primary (1) + COMPANY_FALLBACK_QUERY_SUFFIXES (2) = exactly 3 attempts, no more.
        from question_pipeline.providers.tavily_firecrawl_provider import COMPANY_FALLBACK_QUERY_SUFFIXES
        expected_attempts = 1 + len(COMPANY_FALLBACK_QUERY_SUFFIXES)
        self.assertEqual(len(tavily.search_calls), expected_attempts)
        self.assertIn("bounded", str(ctx.exception).lower())
        self.assertEqual(len(firecrawl.scrape_batch_calls), 0, "Firecrawl must never be called when every Tavily attempt found nothing")

    def test_P_no_duplicate_sources_across_fallback_attempts(self):
        """Only the first successful query's sources are ever used -- sources
        from a failed earlier attempt (there are none here, since a failed
        attempt returns zero sources) are never merged with a later
        successful attempt's sources, so no duplicate-source risk exists."""
        fallback_sources = [
            make_source("Company Tech Overview", "https://example.com/overview", relevance_score=0.9),
            make_source("Company Tech Overview", "https://example.com/overview", relevance_score=0.9),  # exact dup URL within the same result set
        ]
        tavily = FakeTavilyClient(results_sequence=[[], fallback_sources])
        firecrawl = FakeFirecrawlClient(content_by_url={"https://example.com/overview": "Content."})
        provider = make_provider(tavily, firecrawl)

        result = asyncio.run(provider.research_company("DupCo"))

        # Same-URL dedup within one query's result set is handled by the
        # existing DB-level url_normalized uniqueness (migration 0002), not
        # by this provider -- what THIS test guards is that no fallback
        # logic introduces a SECOND, separate source of duplication by
        # re-running or merging multiple queries' source lists together.
        self.assertEqual(len(tavily.search_calls), 2)
        self.assertEqual(result.source_references.count("Company Tech Overview (https://example.com/overview)"), 2)

    def test_Q_role_research_completely_unaffected_single_attempt_only(self):
        """Role research must never use the fallback path -- exact same
        single-attempt behavior and exact same error message as before this
        fix, proving Requirement 8 (preserve all existing role-research
        behavior)."""
        tavily = FakeTavilyClient(results=[])
        firecrawl = FakeFirecrawlClient()
        provider = make_provider(tavily, firecrawl)

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(provider.research_role("Data Engineer"))

        self.assertEqual(len(tavily.search_calls), 1, "role research must make exactly one attempt, never a fallback")
        self.assertIn("Tavily search returned no usable results", str(ctx.exception))
        self.assertNotIn("bounded", str(ctx.exception).lower())

    def test_R_fallback_logic_is_generic_not_openai_specific(self):
        """The bounded fallback query templates must contain no company name
        (OpenAI or otherwise) -- they are pure generic suffixes applied to
        whatever company name is passed in, proving no per-company special
        case exists anywhere in this code path."""
        from question_pipeline.providers.tavily_firecrawl_provider import (
            COMPANY_FALLBACK_QUERY_SUFFIXES, TavilyFirecrawlResearchProvider,
        )
        for suffix in COMPANY_FALLBACK_QUERY_SUFFIXES:
            self.assertNotIn("openai", suffix.lower())
            self.assertNotIn("spotify", suffix.lower())

        # Prove the exact same static method produces correctly-shaped
        # fallback queries for two entirely different company names -- no
        # per-company branch anywhere in this code path.
        queries_a = TavilyFirecrawlResearchProvider._build_company_fallback_queries("OpenAI")
        queries_b = TavilyFirecrawlResearchProvider._build_company_fallback_queries("Some Random Startup Inc")
        self.assertEqual(len(queries_a), len(queries_b))
        self.assertTrue(all(q.startswith("OpenAI ") for q in queries_a))
        self.assertTrue(all(q.startswith("Some Random Startup Inc ") for q in queries_b))


class TestFactoryResolution(unittest.TestCase, KeyPatchMixin):
    def setUp(self):
        self.patch_keys()
        self._orig_mock_mode = config.MOCK_MODE
        self._orig_research_provider = config.RESEARCH_PROVIDER
        self.addCleanup(self._restore)

    def _restore(self):
        config.MOCK_MODE = self._orig_mock_mode
        config.RESEARCH_PROVIDER = self._orig_research_provider

    def test_H_tavily_firecrawl_factory_resolution_works(self):
        from question_pipeline.providers.factory import get_research_provider
        from question_pipeline.providers.tavily_firecrawl_provider import TavilyFirecrawlResearchProvider

        config.MOCK_MODE = False
        config.RESEARCH_PROVIDER = "tavily_firecrawl"

        provider = get_research_provider()
        self.assertIsInstance(provider, TavilyFirecrawlResearchProvider)
        self.assertEqual(provider.provider_name, "tavily_firecrawl")

    def test_I_existing_perplexity_and_gemini_resolution_still_works(self):
        from question_pipeline.providers.factory import get_research_provider
        from question_pipeline.providers.perplexity_provider import PerplexityResearchProvider

        config.MOCK_MODE = False
        config.RESEARCH_PROVIDER = "perplexity"
        provider = get_research_provider()
        self.assertIsInstance(provider, PerplexityResearchProvider)

        config.RESEARCH_PROVIDER = "gemini"
        with self.assertRaises(ProviderBlockedException):
            get_research_provider()

    def test_K_mock_mode_false_never_silently_falls_back(self):
        """With MOCK_MODE=False and tavily_firecrawl selected but credentials
        missing, the factory must still return the real provider class (not
        MockResearchProvider); the BLOCKED outcome must come from actually
        calling it, not from a silent substitution."""
        from question_pipeline.providers.factory import get_research_provider
        from question_pipeline.providers.tavily_firecrawl_provider import TavilyFirecrawlResearchProvider
        from question_pipeline.providers.mock_provider import MockResearchProvider

        config.MOCK_MODE = False
        config.RESEARCH_PROVIDER = "tavily_firecrawl"
        config.TAVILY_API_KEY = ""
        config.FIRECRAWL_API_KEY = ""

        provider = get_research_provider()
        self.assertIsInstance(provider, TavilyFirecrawlResearchProvider)
        self.assertNotIsInstance(provider, MockResearchProvider)

        with self.assertRaises(ProviderBlockedException):
            asyncio.run(provider.research_role("Data Engineer"))


class TestPipelinePrerequisites(unittest.TestCase, KeyPatchMixin):
    def setUp(self):
        self.patch_keys()

    def test_K2_validate_prerequisites_requires_both_keys_for_tavily_firecrawl(self):
        from question_pipeline.pipeline_runner import QuestionPipelineRunner
        from question_pipeline.state_manager import state_manager

        orig_mock = config.MOCK_MODE
        orig_provider = config.RESEARCH_PROVIDER
        try:
            config.MOCK_MODE = False
            config.RESEARCH_PROVIDER = "tavily_firecrawl"
            config.TAVILY_API_KEY = ""
            config.FIRECRAWL_API_KEY = "fc-still-missing-the-other-one"

            runner = QuestionPipelineRunner()
            with self.assertRaises(ProviderBlockedException) as ctx:
                runner.validate_prerequisites("Isolation Test Role For Prereqs")

            msg = str(ctx.exception).lower()
            self.assertIn("tavily_api_key", msg)
            state = state_manager.get_role_state("Isolation Test Role For Prereqs")
            self.assertEqual(state.status, "BLOCKED")
        finally:
            config.MOCK_MODE = orig_mock
            config.RESEARCH_PROVIDER = orig_provider


if __name__ == "__main__":
    unittest.main()
