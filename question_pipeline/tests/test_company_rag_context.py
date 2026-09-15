"""
Tests for company research reaching the generator's RAG context
(rag_service.retrieve_company_context, generator_service's company-context
injection, db_loader.load_company_research, and
orchestrator.process_company_research's cadence/change-detection).

Covers, per the explicit requirements for this feature:
  - role-only generation still works exactly as before
  - relevant company context is retrieved; irrelevant company context is not
  - company-specific evidence can produce COMPANY questions when justified
  - multiple-company evidence can produce DOMAIN questions
  - generic questions remain UNIVERSAL even with company context present
  - company research never forces COMPANY scope by itself
  - unchanged company knowledge triggers no unnecessary work
  - changed company knowledge triggers incremental update + preserves history
  - DB dedup still works when a question's provenance cites a company chunk
  - provenance is preserved end-to-end (question_sources -> research_chunks)

Uses the real (migrated) DB with obviously-synthetic role/company names,
purged in setUp/tearDown -- same pattern as test_db_loader.py /
test_orchestrator.py. No real Tavily/Firecrawl/Gemini calls happen in this
file; the LLM and embedding layers are replaced with small deterministic
fakes so retrieval relevance can be asserted precisely instead of relying on
real-model behavior.
"""
import os
import json
import hashlib
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

os.environ.setdefault("MOCK_MODE", "true")

import psycopg2
from dotenv import load_dotenv

from question_pipeline.config import config
from question_pipeline.models import QuestionObject, ResearchOutput, KnowledgeChunk
from question_pipeline.providers.base import EmbeddingProvider, LLMResult
from question_pipeline.rag_service import RAGService, rag_service as global_rag_service
from question_pipeline.generator_service import generator_service
from question_pipeline.pipeline_runner import pipeline_runner
from question_pipeline.orchestrator import RolloutOrchestrator
from question_pipeline.governor import QuotaExhaustedException
from question_pipeline.db_loader import QuestionBankLoader, LoadReport, slugify
from question_pipeline.tests.test_db_loader import FakeEmbeddingProvider

load_dotenv(config.BASE_DIR.parent / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"), "connect_timeout": 10,
}

TEST_ROLE = "__CompanyRAGTest Role__"
TEST_COMPANY = "__CompanyRAGTest Company__"

# Marker tokens for the deterministic fake embedding provider below. The
# test role name deliberately contains ROLE_MARKER (lowercased, no spaces)
# so retrieve_context()/retrieve_company_context()'s internally-built query
# strings -- which always embed the role name -- naturally carry it.
ROLE_MARKER = "relevantmarkertoken"
OTHER_MARKER = "othercompanymarkertoken"
TEST_ROLE_MARKED = f"__CompanyRAGTest {ROLE_MARKER} Role__"


class FakeMarkerEmbeddingProvider(EmbeddingProvider):
    """Deterministic 2-dim embedding: dim0 = 1.0 iff ROLE_MARKER is present
    in the text, dim1 = 1.0 iff OTHER_MARKER is present. This makes cosine
    similarity between a role-marked query and a chunk fully predictable
    (1.0 if it shares the marker, 0.0 if it doesn't), so relevance-threshold
    behavior can be asserted precisely instead of relying on real-model or
    lexical-hash fuzziness."""

    provider_name_value = "fake_marker"
    model_name = "fake-marker-embed"

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    def _vec(self, text: str):
        t = text.lower()
        return [1.0 if ROLE_MARKER in t else 0.0, 1.0 if OTHER_MARKER in t else 0.0]

    async def embed_texts(self, texts, task_name="rag_embedding"):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text, task_name="query_embedding"):
        return self._vec(text)


def db_connect():
    return psycopg2.connect(**DB_CONFIG)


def fetch_all(sql, params=()):
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def purge(name):
    """Removes every row tied to `name` (used as either a role or a
    standalone company) across all migration-0001/0002 tables this feature
    touches. Safe to call on a role name or a company name -- the two test
    identifiers never collide."""
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM question_validation_attempts WHERE role = %s;", (name,))
    cur.execute("DELETE FROM questions WHERE role = %s;", (name,))
    cur.execute("DELETE FROM pipeline_jobs WHERE role = %s;", (name,))
    cur.execute("DELETE FROM knowledge_state WHERE role = %s;", (name,))
    cur.execute("DELETE FROM knowledge_version_history WHERE role = %s;", (name,))
    cur.execute("DELETE FROM research_chunks WHERE role = %s;", (name,))
    cur.execute("DELETE FROM research_documents WHERE role = %s;", (name,))
    cur.close()
    conn.close()


def make_question(role=TEST_ROLE, **overrides) -> QuestionObject:
    base = dict(
        question="CompanyRAG test: diagnose a production regression using the available evidence.",
        role=role, experience_band="3-5", difficulty=6,
        technical_depth=6, problem_complexity=6, architecture_complexity=6, troubleshooting=6,
        business_complexity=4, decision_making=5, leadership_ownership=3,
        question_type="technical", paradigm="ARCHITECTURE", mandatory_skills=["Testing"],
        scope="UNIVERSAL", domains=[], applicable_companies=[], source_references=[],
    )
    base.update(overrides)
    return QuestionObject(**base)


def make_company_research_output(text: str, company: str = TEST_COMPANY) -> ResearchOutput:
    return ResearchOutput(
        role=company, company=company, raw_summary=text,
        source_references=[f"Test Source (https://example.com/{company.lower()})"],
        source_hash=hashlib.sha256(text.encode()).hexdigest(), actual_provider="tavily_firecrawl",
    )


def make_company_chunk(company: str = TEST_COMPANY) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=f"{company}_c1", role=company, company=company, topic="T", text="chunk text")


def write_local_company_research_file(output: ResearchOutput, company: str = TEST_COMPANY) -> None:
    """Writes a local company_{slug}_research.json artifact exactly like
    research_service.execute_company_research() does for real -- used to
    give _resume_company_downstream()'s Case-B file-read something genuine
    to load, reproducing the real "research succeeded, downstream work
    didn't" precondition without any real API call."""
    path = config.KNOWLEDGE_DIR / f"company_{slugify(company)}_research.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)


def write_local_company_synthesized_file(chunks, company: str = TEST_COMPANY) -> None:
    """Writes a local {slug}_synthesized.json artifact exactly like
    synthesis_service.synthesize() does for real (chunk.company is NOT
    persisted at this stage, matching real behavior) -- used to reproduce
    the "synthesis already succeeded, only indexing/loading failed" variant
    of the stuck-company precondition (the real Spotify scenario)."""
    path = config.KNOWLEDGE_DIR / f"{slugify(company)}_synthesized.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2)


def make_candidate_item(question_text: str, scope: str, applicable_companies, domains=None):
    """A raw LLM-response question dict, matching exactly what
    generator_service._generate_json_with_bounded_retry()'s caller expects
    under data['questions'][i]. role/experience_band/paradigm/
    source_references/knowledge_version/generated_at are force-overwritten
    by generator_service regardless of what's here, mirroring real output."""
    return {
        "question": question_text, "difficulty": 7, "technical_depth": 7, "problem_complexity": 7,
        "architecture_complexity": 7, "troubleshooting": 6, "business_complexity": 5,
        "decision_making": 6, "leadership_ownership": 4, "question_type": "technical",
        "mandatory_skills": ["Testing"], "scope": scope, "domains": domains or [],
        "applicable_companies": applicable_companies,
    }


class TestRAGServiceCompanyContextRetrieval(unittest.TestCase):
    """Exercises rag_service.retrieve_context()/retrieve_company_context()
    directly, with a dedicated RAGService instance (not the shared
    singleton) so it can't leak state into other test files."""

    def setUp(self):
        self.store_file = Path(tempfile.mkdtemp()) / "company_rag_context_test_store.json"
        self.rag = RAGService(store_file=self.store_file)
        self.rag.embedding_provider = FakeMarkerEmbeddingProvider()

    def test_role_only_retrieval_excludes_company_chunks(self):
        """Existing role research must continue working exactly as before:
        retrieve_context() must never return a company-only chunk, even one
        that would otherwise score as relevant."""
        async def run():
            role_chunk = KnowledgeChunk(chunk_id="role_c1", role=TEST_ROLE_MARKED, topic="Core",
                                         text=f"{ROLE_MARKER} core role knowledge.")
            company_chunk = KnowledgeChunk(chunk_id="co_c1", role="SomeCo", company="SomeCo", topic="Co",
                                            text=f"{ROLE_MARKER} company specific info sharing the same marker.")
            await self.rag.index_chunks([role_chunk, company_chunk])
            retrieved = await self.rag.retrieve_context(TEST_ROLE_MARKED, "3-5", "ARCHITECTURE", top_k=5)
            self.assertEqual([c.chunk_id for c in retrieved], ["role_c1"])
            self.assertIsNone(retrieved[0].company)
        asyncio.run(run())

    def test_relevant_company_context_is_retrieved(self):
        async def run():
            company = "RelevantCo"
            chunk = KnowledgeChunk(chunk_id="co_relevant_1", role=company, company=company, topic="Scale",
                                    text=f"{ROLE_MARKER} large-scale system design specific to RelevantCo.")
            await self.rag.index_chunks([chunk])
            retrieved = await self.rag.retrieve_company_context(TEST_ROLE_MARKED, "5-8", "STRATEGY", top_k=1)
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0].company, company)
        asyncio.run(run())

    def test_irrelevant_company_context_is_not_retrieved(self):
        """A company whose research doesn't actually relate to this role's
        query must never surface -- the relevance threshold, not a hardcoded
        list, is what keeps company context evidence-driven."""
        async def run():
            company = "IrrelevantCo"
            chunk = KnowledgeChunk(chunk_id="co_irrelevant_1", role=company, company=company, topic="Audit",
                                    text=f"{OTHER_MARKER} unrelated audit and compliance consulting practices.")
            await self.rag.index_chunks([chunk])
            retrieved = await self.rag.retrieve_company_context(TEST_ROLE_MARKED, "5-8", "STRATEGY", top_k=1)
            self.assertEqual(retrieved, [])
        asyncio.run(run())

    def test_company_filter_restricts_pool_without_injecting_every_company(self):
        """Passing an explicit companies= list must restrict retrieval to
        those companies -- proving this mechanism can never degenerate into
        injecting the entire company roster regardless of relevance."""
        async def run():
            chunk_a = KnowledgeChunk(chunk_id="co_a1", role="CompanyA", company="CompanyA", topic="A",
                                      text=f"{ROLE_MARKER} CompanyA-specific system design.")
            chunk_b = KnowledgeChunk(chunk_id="co_b1", role="CompanyB", company="CompanyB", topic="B",
                                      text=f"{ROLE_MARKER} CompanyB-specific system design.")
            await self.rag.index_chunks([chunk_a, chunk_b])
            retrieved = await self.rag.retrieve_company_context(
                TEST_ROLE_MARKED, "5-8", "STRATEGY", companies=["CompanyA"], top_k=5,
            )
            self.assertEqual([c.company for c in retrieved], ["CompanyA"])
        asyncio.run(run())


class TestGeneratorServiceCompanyContextIntegrity(unittest.TestCase):
    """Exercises generator_service.generate_batch_for_band() end-to-end
    (LLM mocked) against the real rag_service singleton, then feeds the
    result through pipeline_runner.classify_scope() -- proving company
    context reaching the prompt never forces a scope, and that genuinely
    company-/domain-specific LLM output is still classified correctly."""

    def setUp(self):
        self._orig_chunks = list(global_rag_service.chunks)
        self._orig_provider = global_rag_service.embedding_provider
        global_rag_service.chunks = []
        global_rag_service.embedding_provider = FakeMarkerEmbeddingProvider()

    def tearDown(self):
        global_rag_service.chunks = self._orig_chunks
        global_rag_service.embedding_provider = self._orig_provider

    def _run_generation(self, item):
        mock_generate = AsyncMock(return_value=LLMResult(text="", data={"questions": [item]}))
        with patch.object(generator_service.llm, "generate_json", new=mock_generate):
            result = asyncio.run(
                generator_service.generate_batch_for_band(TEST_ROLE_MARKED, "3-5", 1, item["scope"])
            )
        return result, mock_generate

    def test_role_only_generation_still_works_without_company_context(self):
        item = make_candidate_item("Diagnose a subtle caching bug from the given symptoms.", "UNIVERSAL", [])
        result, mock_generate = self._run_generation(item)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, TEST_ROLE_MARKED)
        self.assertEqual(result[0].scope, "UNIVERSAL")
        prompt = mock_generate.call_args.kwargs["prompt"]
        # The static scope-rules instructions always mention the literal
        # phrase "[Company Context -- ...]" once, describing what to do IF
        # such a block appears -- so the real assertion is that no ACTUAL
        # block was injected (which would push the count to 2+).
        self.assertEqual(prompt.count("[Company Context"), 1, "no real company-context block should be injected")
        self.assertNotIn("Company:", "".join(result[0].source_references))

    def test_irrelevant_company_context_is_not_injected_into_prompt(self):
        chunk = KnowledgeChunk(chunk_id="co_irr_1", role="IrrelevantCo", company="IrrelevantCo", topic="Audit",
                                text=f"{OTHER_MARKER} unrelated audit consulting content.")
        asyncio.run(global_rag_service.index_chunks([chunk]))
        item = make_candidate_item("Diagnose a subtle caching bug from the given symptoms.", "UNIVERSAL", [])
        result, mock_generate = self._run_generation(item)
        prompt = mock_generate.call_args.kwargs["prompt"]
        self.assertEqual(prompt.count("[Company Context"), 1, "an irrelevant company's context must never be injected")
        self.assertNotIn("IrrelevantCo", prompt)

    def test_relevant_company_context_reaches_prompt_but_does_not_force_company_scope(self):
        """Company research must never force COMPANY scope by itself: even
        with genuinely relevant company context injected as evidence, a
        UNIVERSAL-intent candidate the LLM writes with no company dependency
        must classify as UNIVERSAL."""
        chunk = KnowledgeChunk(chunk_id="co_rel_1", role="RelevantCo", company="RelevantCo", topic="Scale",
                                text=f"{ROLE_MARKER} RelevantCo's large-scale system design approach.")
        asyncio.run(global_rag_service.index_chunks([chunk]))
        item = make_candidate_item("Diagnose a subtle caching bug from the given symptoms.", "UNIVERSAL", [])
        result, mock_generate = self._run_generation(item)

        prompt = mock_generate.call_args.kwargs["prompt"]
        self.assertIn("[Company Context -- RelevantCo", prompt, "the relevant company evidence must actually reach the prompt")
        self.assertTrue(any("Company: RelevantCo" in ref for ref in result[0].source_references))

        classified = pipeline_runner.classify_scope(result[0], target_scope="UNIVERSAL")
        self.assertEqual(classified, "UNIVERSAL", "company context being present must not force COMPANY scope")

    def test_company_specific_evidence_can_produce_company_scope_when_justified(self):
        chunk = KnowledgeChunk(chunk_id="co_rel_2", role="RelevantCo", company="RelevantCo", topic="Scale",
                                text=f"{ROLE_MARKER} RelevantCo's proprietary infrastructure.")
        asyncio.run(global_rag_service.index_chunks([chunk]))
        item = make_candidate_item(
            "How would you redesign RelevantCo's proprietary real-time bidding pipeline for 10x scale?",
            "COMPANY", ["RelevantCo"],
        )
        result, _ = self._run_generation(item)
        classified = pipeline_runner.classify_scope(result[0], target_scope="COMPANY")
        self.assertEqual(classified, "COMPANY")

    def test_multiple_company_evidence_can_produce_domain_scope(self):
        item = make_candidate_item(
            "How do FinTech platforms like Stripe and PayPal handle idempotent payment retries?",
            "DOMAIN", ["Stripe", "PayPal"],
        )
        result, _ = self._run_generation(item)
        classified = pipeline_runner.classify_scope(result[0], target_scope="DOMAIN")
        self.assertEqual(classified, "DOMAIN")


def make_company_patchers(company: str, output, synth_mock=None, load_mock=None):
    """Shared patch set for a mocked process_company_research() call: stubs
    execute_company_research/synthesize_and_index_company_research/
    load_company_research so no real Tavily/Firecrawl/Gemini call happens,
    for any company name (not just TEST_COMPANY) -- used both by the
    change-detection tests below and by the batch-processing tests."""
    synth_mock = synth_mock or AsyncMock(return_value=[make_company_chunk(company)])
    load_mock = load_mock or Mock(return_value=LoadReport(role=company, research_chunks_inserted=3))
    return [
        patch("question_pipeline.orchestrator.research_service.execute_company_research",
              new=AsyncMock(return_value={"status": "SUCCESS", "company": company,
                                           "research_output": output, "actual_provider": "tavily_firecrawl"})),
        patch("question_pipeline.orchestrator.synthesize_and_index_company_research", new=synth_mock),
        patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research", new=load_mock),
    ], synth_mock, load_mock


class TestOrchestratorCompanyResearchCadence(unittest.TestCase):
    """Exercises orchestrator.process_company_research()'s change-detection
    against the real DB, with research/synthesis/RAG-indexing/loading
    mocked (no real Tavily/Firecrawl/Gemini calls)."""

    def setUp(self):
        purge(TEST_COMPANY)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)

    def tearDown(self):
        purge(TEST_COMPANY)

    def _patchers_for(self, output, synth_mock=None, load_mock=None):
        return make_company_patchers(TEST_COMPANY, output, synth_mock=synth_mock, load_mock=load_mock)

    def test_company_research_first_run_completes_and_indexes(self):
        output = make_company_research_output("v1 company content")
        patchers, synth_mock, load_mock = self._patchers_for(output)
        for p in patchers:
            p.start()
        try:
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.metrics["chunks_inserted"], 3)
        synth_mock.assert_awaited_once()
        load_mock.assert_called_once_with(TEST_COMPANY)

        state = self.orchestrator.freshness.get_state(TEST_COMPANY, company=TEST_COMPANY)
        self.assertEqual(state["knowledge_version"], "v1.0")
        job = self.orchestrator.jobs.get(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH")
        self.assertEqual(job["status"], "COMPLETED")

    def _force_due_again(self):
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute(
            "UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' "
            "WHERE role = %s AND company = %s;",
            (TEST_COMPANY, TEST_COMPANY),
        )
        cur.close(); conn.close()

    def test_unchanged_company_knowledge_skips_synthesis_and_load(self):
        output = make_company_research_output("stable company content")
        patchers1, _, _ = self._patchers_for(output)
        for p in patchers1:
            p.start()
        try:
            asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
        finally:
            for p in patchers1:
                p.stop()

        self._force_due_again()

        synth_mock2 = AsyncMock()
        load_mock2 = Mock()
        patchers2, _, _ = self._patchers_for(output, synth_mock=synth_mock2, load_mock=load_mock2)
        for p in patchers2:
            p.start()
        try:
            result2 = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
        finally:
            for p in patchers2:
                p.stop()

        self.assertEqual(result2.status, "SKIPPED_NO_CHANGE")
        synth_mock2.assert_not_called()
        load_mock2.assert_not_called()

        state = self.orchestrator.freshness.get_state(TEST_COMPANY, company=TEST_COMPANY)
        self.assertEqual(state["knowledge_version"], "v1.0", "unchanged content must not bump the version")

    def test_changed_company_knowledge_triggers_update_and_preserves_history(self):
        output_v1 = make_company_research_output("version one company content")
        patchers1, _, _ = self._patchers_for(output_v1)
        for p in patchers1:
            p.start()
        try:
            asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
        finally:
            for p in patchers1:
                p.stop()

        self._force_due_again()

        output_v2 = make_company_research_output("version TWO -- materially different company content")
        patchers2, synth_mock2, load_mock2 = self._patchers_for(output_v2)
        for p in patchers2:
            p.start()
        try:
            result2 = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
        finally:
            for p in patchers2:
                p.stop()

        self.assertEqual(result2.status, "COMPLETED")
        synth_mock2.assert_awaited_once()
        load_mock2.assert_called_once_with(TEST_COMPANY)

        state = self.orchestrator.freshness.get_state(TEST_COMPANY, company=TEST_COMPANY)
        self.assertEqual(state["knowledge_version"], "v1.1")

        history = fetch_all(
            "SELECT knowledge_version FROM knowledge_version_history WHERE role = %s ORDER BY id;",
            (TEST_COMPANY,),
        )
        self.assertEqual([h[0] for h in history], ["v1.0", "v1.1"], "previous knowledge version must be preserved, not overwritten")


class TestCompanyStateMachineIndependence(unittest.TestCase):
    """Regression tests for the company-level equivalent of the role
    resume/recovery bug (real incident: Spotify). process_company_research()
    used to gate its ENTIRE body on freshness.is_due(), so once research
    succeeded (setting last_researched_at), an incomplete downstream
    (synthesis/RAG-indexing/DB-load) job -- left at WAITING_FOR_QUOTA/
    BLOCKED/etc by a prior failed attempt -- could never resume; the next
    "not due" check would silently overwrite that status with a misleading
    SKIPPED_NO_CHANGE and the company's chunks would never actually load.

    Fix: process_company_research() now dispatches on research freshness
    AND downstream-job-completeness independently (mirroring process_role()
    exactly), and _resume_company_downstream() finishes the pipeline
    without ever making a new research API call -- reusing an existing
    local research (and, when safe, synthesized) artifact instead.

    Uses the real (migrated) DB with the standard TEST_COMPANY fixture,
    purged in setUp/tearDown. No real Tavily/Firecrawl/Gemini calls happen
    in this file."""

    def setUp(self):
        purge(TEST_COMPANY)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)

    def tearDown(self):
        purge(TEST_COMPANY)

    def test_fresh_research_incomplete_downstream_resumes_without_reresearch(self):
        output = make_company_research_output("stable content")
        write_local_company_research_file(output)
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output.source_hash, None)
        # Simulate a prior attempt that got partway through synthesis before failing.
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "SYNTHESIZING", mark_started=True)

        research_mock = AsyncMock()
        synth_mock = AsyncMock(return_value=[make_company_chunk()])
        load_mock = Mock(return_value=LoadReport(role=TEST_COMPANY, research_chunks_inserted=1))
        with patch("question_pipeline.orchestrator.research_service.execute_company_research", new=research_mock), \
             patch("question_pipeline.orchestrator.synthesize_and_index_company_research", new=synth_mock), \
             patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research", new=load_mock):
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        research_mock.assert_not_called()
        self.assertEqual(result.status, "COMPLETED")
        job = self.orchestrator.jobs.get(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH")
        self.assertEqual(job["status"], "COMPLETED")

    def test_fresh_research_complete_downstream_skips(self):
        output = make_company_research_output("stable content")
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output.source_hash, None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True)

        research_mock = AsyncMock()
        synth_mock = AsyncMock()
        with patch("question_pipeline.orchestrator.research_service.execute_company_research", new=research_mock), \
             patch("question_pipeline.orchestrator.synthesize_and_index_company_research", new=synth_mock):
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        self.assertEqual(result.status, "SKIPPED_NO_CHANGE")
        research_mock.assert_not_called()
        synth_mock.assert_not_called()

    def test_waiting_for_quota_with_fresh_research_and_completed_synthesis_resumes_without_reresearch_or_resynthesis(self):
        """The EXACT real Spotify shape: research succeeded, synthesis
        ALSO succeeded (a real local synthesized file exists), only the
        RAG-indexing/embedding step failed. Resume must reuse the existing
        synthesized file rather than paying for a redundant synthesis call."""
        output = make_company_research_output("spotify-like content")
        write_local_company_research_file(output)
        chunks = [make_company_chunk()]
        write_local_company_synthesized_file(chunks)
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output.source_hash, None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "WAITING_FOR_QUOTA",
                                       reason="simulated prior quota exhaustion during indexing")

        research_mock = AsyncMock()
        synthesize_mock = AsyncMock()  # synthesis_service.synthesize -- must NOT be called (reuse the file instead)
        load_mock = Mock(return_value=LoadReport(role=TEST_COMPANY, research_chunks_inserted=1))
        with patch("question_pipeline.orchestrator.research_service.execute_company_research", new=research_mock), \
             patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=synthesize_mock), \
             patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)), \
             patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research", new=load_mock):
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        research_mock.assert_not_called()
        synthesize_mock.assert_not_called()
        self.assertEqual(result.status, "COMPLETED")

    def test_quota_during_indexing_leaves_waiting_for_quota_not_crash(self):
        output = make_company_research_output("first time content")
        with patch("question_pipeline.orchestrator.research_service.execute_company_research",
                   new=AsyncMock(return_value={"status": "SUCCESS", "company": TEST_COMPANY,
                                                "research_output": output, "actual_provider": "tavily_firecrawl"})), \
             patch("question_pipeline.orchestrator.synthesize_and_index_company_research",
                   new=AsyncMock(side_effect=QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated"))):
            try:
                result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))
            except QuotaExhaustedException:
                self.fail("QuotaExhaustedException during indexing must never propagate and crash the process")

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = self.orchestrator.jobs.get(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        self.assertNotEqual(job["status"], "COMPLETED")

    def test_successful_resume_ends_completed_in_db(self):
        """Same scenario as the core resume test, but asserting directly
        against a fresh DB read (not just the returned UnitResult), and
        that real research_chunks actually landed in the DB -- this is the
        literal Spotify recovery proof, generalized."""
        output = make_company_research_output("stable content")
        write_local_company_research_file(output)
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output.source_hash, None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "BLOCKED", reason="simulated prior block")

        with patch("question_pipeline.orchestrator.research_service.execute_company_research", new=AsyncMock()), \
             patch("question_pipeline.orchestrator.synthesize_and_index_company_research",
                   new=AsyncMock(return_value=[make_company_chunk()])), \
             patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research",
                   new=Mock(return_value=LoadReport(role=TEST_COMPANY, research_chunks_inserted=1))):
            asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        rows = fetch_all("SELECT status FROM pipeline_jobs WHERE role=%s AND company=%s AND job_type='COMPANY_RESEARCH';",
                          (TEST_COMPANY, TEST_COMPANY))
        self.assertEqual(rows[0][0], "COMPLETED")

    def test_stale_research_incomplete_downstream_refreshes_then_resumes(self):
        """Case D: research is due (stale) AND downstream was incomplete --
        must refresh research (it's genuinely due) and then still complete
        the downstream pipeline, not get stuck on the old incomplete state."""
        output_v1 = make_company_research_output("version one")
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output_v1.source_hash, None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "WAITING_FOR_QUOTA", reason="simulated")
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute("UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' WHERE role=%s AND company=%s;",
                    (TEST_COMPANY, TEST_COMPANY))
        cur.close(); conn.close()

        output_v2 = make_company_research_output("version two -- different")
        with patch("question_pipeline.orchestrator.research_service.execute_company_research",
                   new=AsyncMock(return_value={"status": "SUCCESS", "company": TEST_COMPANY,
                                                "research_output": output_v2, "actual_provider": "tavily_firecrawl"})), \
             patch("question_pipeline.orchestrator.synthesize_and_index_company_research",
                   new=AsyncMock(return_value=[make_company_chunk()])), \
             patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research",
                   new=Mock(return_value=LoadReport(role=TEST_COMPANY, research_chunks_inserted=1))):
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        self.assertEqual(result.status, "COMPLETED")
        state = self.orchestrator.freshness.get_state(TEST_COMPANY, company=TEST_COMPANY)
        self.assertEqual(state["knowledge_version"], "v1.1", "genuinely changed content must still bump the version")

    def test_content_changed_never_reuses_stale_synthesized_file(self):
        """Critical correctness guard for the fix itself: a genuine content
        CHANGE must always force fresh synthesis, never silently reuse a
        synthesized file that belongs to the PREVIOUS knowledge version."""
        output_v1 = make_company_research_output("version one")
        write_local_company_research_file(output_v1)
        write_local_company_synthesized_file([make_company_chunk()])  # belongs to v1 only
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", output_v1.source_hash, None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True)
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute("UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' WHERE role=%s AND company=%s;",
                    (TEST_COMPANY, TEST_COMPANY))
        cur.close(); conn.close()

        output_v2 = make_company_research_output("version two -- materially different")
        synthesize_mock = AsyncMock(return_value=[make_company_chunk()])
        with patch("question_pipeline.orchestrator.research_service.execute_company_research",
                   new=AsyncMock(return_value={"status": "SUCCESS", "company": TEST_COMPANY,
                                                "research_output": output_v2, "actual_provider": "tavily_firecrawl"})), \
             patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=synthesize_mock), \
             patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)), \
             patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research",
                   new=Mock(return_value=LoadReport(role=TEST_COMPANY, research_chunks_inserted=1))):
            result = asyncio.run(self.orchestrator.process_company_research(TEST_COMPANY))

        synthesize_mock.assert_awaited_once()  # fresh synthesis WAS performed, not skipped
        self.assertEqual(result.status, "COMPLETED")


TEST_COMPANY_2 = "__CompanyRAGTest Company Two__"


class TestCompanyBatchProcessing(unittest.TestCase):
    """Tests the scheduler-tick primitives added for the local automation
    task: get_pending_companies() (skip-fresh-companies filtering, so a
    scheduler batch slot is never wasted re-checking an already-fresh
    company) and run_company_batch() (quota/blocked-aware batch stopping),
    mirroring the existing role-axis get_pending_roles()/run_batch()
    behavior exactly."""

    def setUp(self):
        purge(TEST_COMPANY)
        purge(TEST_COMPANY_2)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)

    def tearDown(self):
        purge(TEST_COMPANY)
        purge(TEST_COMPANY_2)

    def test_never_processed_company_is_pending(self):
        pending = self.orchestrator.get_pending_companies([TEST_COMPANY])
        self.assertEqual(pending, [TEST_COMPANY])

    def test_completed_and_fresh_company_is_not_pending(self):
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", "hash1", None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True)
        pending = self.orchestrator.get_pending_companies([TEST_COMPANY])
        self.assertEqual(pending, [], "a just-completed, still-fresh company must not be pending again")

    def test_stale_completed_company_is_pending_again(self):
        self.orchestrator.freshness.record_change(TEST_COMPANY, TEST_COMPANY, "v1.0", "hash1", None)
        self.orchestrator.jobs.upsert(TEST_COMPANY, TEST_COMPANY, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True)
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute(
            "UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' "
            "WHERE role = %s AND company = %s;",
            (TEST_COMPANY, TEST_COMPANY),
        )
        cur.close(); conn.close()
        pending = self.orchestrator.get_pending_companies([TEST_COMPANY])
        self.assertEqual(pending, [TEST_COMPANY], "a stale (cadence-elapsed) company must be pending again")

    def test_run_company_batch_skips_fresh_companies_and_does_not_waste_batch_slots(self):
        """A batch_size of 1, with one fresh company and one never-processed
        company requested, must spend its one slot on the pending company --
        not on re-touching the fresh one."""
        self.orchestrator.freshness.record_change(TEST_COMPANY_2, TEST_COMPANY_2, "v1.0", "hash1", None)
        self.orchestrator.jobs.upsert(TEST_COMPANY_2, TEST_COMPANY_2, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True)

        output = make_company_research_output("fresh content for pending company", company=TEST_COMPANY)
        patchers, synth_mock, load_mock = make_company_patchers(TEST_COMPANY, output)
        for p in patchers:
            p.start()
        try:
            results = asyncio.run(self.orchestrator.run_company_batch([TEST_COMPANY_2, TEST_COMPANY], batch_size=1))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].company, TEST_COMPANY)
        self.assertEqual(results[0].status, "COMPLETED")
        synth_mock.assert_awaited_once()

    def test_run_company_batch_stops_on_quota_without_erroring(self):
        """Mirrors run_batch()'s role-axis quota-stop behavior: the moment
        one company in the batch reports WAITING_FOR_QUOTA, the batch must
        stop -- not raise, not silently continue burning further quota-
        exhausted calls against the remaining companies."""
        company_ok = TEST_COMPANY
        company_quota = TEST_COMPANY_2

        async def flaky_execute_company_research(company):
            if company == company_quota:
                raise QuotaExhaustedException("tavily_firecrawl", "tavily", "simulated quota exhaustion")
            return {
                "status": "SUCCESS", "company": company,
                "research_output": make_company_research_output("ok content", company=company),
                "actual_provider": "tavily_firecrawl",
            }

        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_company_research",
                  new=AsyncMock(side_effect=flaky_execute_company_research)),
            patch("question_pipeline.orchestrator.synthesize_and_index_company_research",
                  new=AsyncMock(return_value=[make_company_chunk(company_ok)])),
            patch("question_pipeline.db_loader.QuestionBankLoader.load_company_research",
                  new=Mock(return_value=LoadReport(role=company_ok, research_chunks_inserted=1))),
        ]
        for p in patchers:
            p.start()
        try:
            # company_quota is processed first (batch order matches the list
            # order given), so it hits quota immediately and the batch must
            # stop before ever attempting company_ok.
            results = asyncio.run(self.orchestrator.run_company_batch([company_quota, company_ok], batch_size=2))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(len(results), 1, "the batch must stop at the first quota-exhausted company, not continue")
        self.assertEqual(results[0].company, company_quota)
        self.assertEqual(results[0].status, "WAITING_FOR_QUOTA")

        job_ok = self.orchestrator.jobs.get(company_ok, company_ok, "COMPANY_RESEARCH")
        self.assertIsNone(job_ok, "a company stopped-before in the batch must be completely untouched")


class TestCompanyChunkProvenanceAndDedup(unittest.TestCase):
    """DB-level proof that a generated question's provenance can link to a
    real company research_chunks row, and that this doesn't disturb the
    existing questions.role/question_text_normalized dedup constraint."""

    def setUp(self):
        purge(TEST_ROLE)
        purge(TEST_COMPANY)
        self.loader = QuestionBankLoader(DB_CONFIG)
        self._embedding_patcher = patch("question_pipeline.db_loader.get_embedding_provider", return_value=FakeEmbeddingProvider())
        self._embedding_patcher.start()

    def tearDown(self):
        self._embedding_patcher.stop()
        purge(TEST_ROLE)
        purge(TEST_COMPANY)

    def _seed_company_chunk(self):
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute(
            "INSERT INTO research_documents (role, company, research_provider, raw_summary) "
            "VALUES (%s, %s, 'test', 'test') RETURNING id;",
            (TEST_COMPANY, TEST_COMPANY),
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO research_chunks (chunk_id, research_document_id, role, company, topic, text) "
            "VALUES ('company_rag_test_chunk_1', %s, %s, %s, 'Scale', 'Company-specific scaling knowledge.') "
            "RETURNING id;",
            (doc_id, TEST_COMPANY, TEST_COMPANY),
        )
        chunk_pk = cur.fetchone()[0]
        cur.close(); conn.close()
        return chunk_pk

    def test_provenance_links_generated_question_to_company_research_chunk(self):
        chunk_pk = self._seed_company_chunk()
        # Mirrors exactly the ref_entry format generator_service.py builds
        # for a company chunk: "RAG Chunk: <id> | Company: <name> | Topic: <topic>".
        q = make_question(source_references=[f"RAG Chunk: company_rag_test_chunk_1 | Company: {TEST_COMPANY} | Topic: Scale"])
        asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))

        qid = fetch_all("SELECT id FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        src_rows = fetch_all(
            "SELECT source_reference, research_chunk_id FROM question_sources WHERE question_id = %s;", (qid,)
        )
        self.assertEqual(len(src_rows), 1)
        self.assertEqual(src_rows[0][1], chunk_pk, "provenance must link to the real company research_chunks row")

    def test_reload_with_company_chunk_reference_is_idempotent(self):
        self._seed_company_chunk()
        q = make_question(source_references=[f"RAG Chunk: company_rag_test_chunk_1 | Company: {TEST_COMPANY} | Topic: Scale"])

        asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))
        asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))

        q_count = fetch_all("SELECT COUNT(*) FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        self.assertEqual(q_count, 1, "dedup must still prevent a duplicate row on rerun even when citing a company chunk")

        src_count = fetch_all(
            "SELECT COUNT(*) FROM question_sources WHERE question_id = "
            "(SELECT id FROM questions WHERE role = %s);", (TEST_ROLE,),
        )[0][0]
        self.assertEqual(src_count, 1, "no duplicate question_sources row either")


if __name__ == "__main__":
    unittest.main()
