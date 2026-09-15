"""
Regression tests for the two bugs found during the real Phase 5B rollout:

BUG 1 -- Quota exhaustion during validation crashed the whole process.
    orchestrator._generate_and_load() (formerly inline in process_role())
    caught QuotaExhaustedException/ProviderBlockedException around
    generate_batch_for_band(), but NOT around validator_service.
    validate_question() (which can raise from its internal RAG-context
    embedding call) or around the final db_loader.load_role() call (which
    computes question-bank embeddings before its own DB transaction). A
    real Gemini free-tier daily embedding quota exhaustion during
    validation propagated all the way out of asyncio.run() and killed the
    `rollout_cli.py` process with a traceback instead of checkpointing
    WAITING_FOR_QUOTA.

    Fix: every provider-calling step inside _generate_and_load() (batch
    generation, per-candidate validation, and the final load) is now
    individually wrapped, each checkpointing WAITING_FOR_QUOTA/BLOCKED with
    an `approved_so_far` metric and persisting whatever was already
    approved via _load_partial() before returning -- mirroring the
    generation-phase pattern that already existed.

BUG 2 -- A role whose research succeeded but whose generation never
    finished (crashed, paused for quota, blocked) became permanently stuck:
    process_role() used to gate its ENTIRE body on
    `freshness.is_due(role)`, so once research succeeded (making the role
    "not due" for ~30 days), an incomplete ROLE_GENERATION job (status left
    at e.g. VALIDATING) could never be resumed until the next research
    cadence window -- even though get_pending_roles() correctly kept
    listing the role as pending.

    Fix: process_role() now dispatches on TWO independent signals --
    research freshness (knowledge_state) and generation completeness
    (pipeline_jobs.ROLE_GENERATION status) -- and a fresh-research/
    incomplete-generation role resumes generation directly, without
    re-researching. See orchestrator.py's process_role()/_generate_and_load()
    docstrings for the full case table (A/B/C/D).

Uses the real (migrated) DB with a synthetic role name, purged in
setUp/tearDown -- same pattern as test_orchestrator.py. No real Tavily/
Firecrawl/Gemini calls happen in this file.
"""
import os
import json
import hashlib
import asyncio
import unittest
from unittest.mock import patch, AsyncMock, Mock

os.environ.setdefault("MOCK_MODE", "true")

import psycopg2
from dotenv import load_dotenv

from question_pipeline.config import config
from question_pipeline.models import QuestionObject, ResearchOutput, KnowledgeChunk, QualityGateResult
from question_pipeline.governor import QuotaExhaustedException, ProviderBlockedException
from question_pipeline.orchestrator import RolloutOrchestrator, PipelineJobStore, fetch_existing_bank_from_db
from question_pipeline.deduplicator_service import DeduplicatorService
from question_pipeline.tests.test_db_loader import FakeEmbeddingProvider

_embedding_patcher = patch("question_pipeline.db_loader.get_embedding_provider", return_value=FakeEmbeddingProvider())

load_dotenv(config.BASE_DIR.parent / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"), "connect_timeout": 10,
}

TEST_ROLE = "__StateMachineTest Role__"


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


def purge(role=TEST_ROLE):
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM question_validation_attempts WHERE role = %s;", (role,))
    cur.execute("DELETE FROM questions WHERE role = %s;", (role,))
    cur.execute("DELETE FROM pipeline_jobs WHERE role = %s;", (role,))
    cur.execute("DELETE FROM knowledge_state WHERE role = %s;", (role,))
    cur.execute("DELETE FROM knowledge_version_history WHERE role = %s;", (role,))
    cur.execute("DELETE FROM research_chunks WHERE role = %s;", (role,))
    cur.execute("DELETE FROM research_documents WHERE role = %s;", (role,))
    cur.close()
    conn.close()


def make_research_output(text="Test research content.", role=TEST_ROLE) -> ResearchOutput:
    return ResearchOutput(
        role=role, raw_summary=text, source_references=["Test Source (https://example.com/x)"],
        source_hash=hashlib.sha256(text.encode()).hexdigest(), actual_provider="tavily_firecrawl",
    )


def make_chunks(role=TEST_ROLE):
    return [KnowledgeChunk(chunk_id=f"{role}_c1", role=role, topic="T", text="chunk text")]


def make_question(text, **overrides) -> QuestionObject:
    base = dict(
        question=text, role=TEST_ROLE, experience_band="3-5", difficulty=6,
        technical_depth=6, problem_complexity=6, architecture_complexity=6, troubleshooting=6,
        business_complexity=4, decision_making=5, leadership_ownership=3,
        question_type="technical", paradigm="ARCHITECTURE", mandatory_skills=["Testing"],
        scope="UNIVERSAL", domains=[], applicable_companies=[],
    )
    base.update(overrides)
    return QuestionObject(**base)


async def approve_all(cand, existing_bank, expected_role=None):
    return QualityGateResult(approved=True, quality_score=0.9, critical_failure=False,
                              validation_tier="primary_gemini", validation_provider="gemini", validation_model="fake")


class BaseOrchestratorTest(unittest.TestCase):
    def setUp(self):
        purge(TEST_ROLE)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge(TEST_ROLE)

    def _base_patchers(self, research_output=None, chunks=None, candidates_per_band=None, validate_side_effect=None, load_side_effect=None):
        research_output = research_output or make_research_output()
        chunks = chunks if chunks is not None else make_chunks()
        candidates_per_band = candidates_per_band if candidates_per_band is not None else {
            band: [make_question(f"Question for {band} band, candidate {i}.", experience_band=band)]
            for i, band in enumerate(config.EXPERIENCE_BANDS)
        }

        async def fake_generate(role, band, count, target_scope, domain=None):
            return list(candidates_per_band.get(band, []))

        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_research",
                  new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": research_output})),
            patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=chunks)),
            patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)),
            patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=AsyncMock(side_effect=fake_generate)),
        ]
        if validate_side_effect is not None:
            patchers.append(patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=validate_side_effect)))
        else:
            patchers.append(patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)))
        if load_side_effect is not None:
            patchers.append(patch("question_pipeline.db_loader.QuestionBankLoader.load_role", new=AsyncMock(side_effect=load_side_effect)))
        return patchers

    def _run(self, patchers, coro_factory):
        for p in patchers:
            p.start()
        try:
            return asyncio.run(coro_factory())
        finally:
            for p in patchers:
                p.stop()


class TestQuotaHandledAtEveryPhase(BaseOrchestratorTest):
    """Requirements 1-6: QuotaExhaustedException must never crash the
    process, at any phase, and must always checkpoint WAITING_FOR_QUOTA
    (or degrade gracefully where that's the existing, correct design)."""

    def test_1_quota_during_generation_checkpoints_waiting_for_quota(self):
        async def flaky_generate(role, band, count, target_scope, domain=None):
            if band == "3-5":
                raise QuotaExhaustedException("gemini", "gemini-3.1-flash-lite", "simulated")
            return [make_question(f"Q for {band}.", experience_band=band)]

        with patch("question_pipeline.orchestrator.research_service.execute_research",
                   new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": make_research_output()})), \
             patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=make_chunks())), \
             patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)), \
             patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=AsyncMock(side_effect=flaky_generate)), \
             patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)):
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = self.orchestrator.jobs.get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 2, "the 2 bands processed before quota exhaustion must be preserved")

    def test_2_quota_during_validation_checkpoints_without_crashing(self):
        """THE core regression test for Bug 1: a QuotaExhaustedException
        raised from inside validate_question() (e.g. its RAG-context
        embedding call) must not propagate and crash process_role() -- it
        must be caught, checkpointed, and whatever was already approved
        must be persisted."""
        call_count = {"n": 0}

        async def flaky_validate(cand, existing_bank, expected_role=None):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated embedding quota exhaustion")
            return await approve_all(cand, existing_bank, expected_role)

        patchers = self._base_patchers(validate_side_effect=flaky_validate)
        try:
            result = self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))
        except QuotaExhaustedException:
            self.fail("QuotaExhaustedException from validate_question() must never propagate out of process_role()")

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = self.orchestrator.jobs.get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        self.assertNotEqual(job["status"], "COMPLETED", "incomplete work must never be marked COMPLETED")
        # The 2 candidates validated successfully before the 3rd raised must be preserved.
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 2, "validation progress before the quota hit must not be lost")

    def test_3_quota_during_rag_indexing_checkpoints_waiting_for_quota(self):
        with patch("question_pipeline.orchestrator.research_service.execute_research",
                   new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": make_research_output()})), \
             patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=make_chunks())), \
             patch("question_pipeline.orchestrator.rag_service.index_chunks",
                   new=AsyncMock(side_effect=QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated"))):
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = self.orchestrator.jobs.get(TEST_ROLE, None, "ROLE_RESEARCH")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")

    def test_4_quota_during_dedup_embedding_degrades_gracefully_not_crash(self):
        """The existing (correct) design: deduplicator_service.check_duplicate()
        catches ANY exception from its semantic-embedding tier and falls
        back to the normalized-overlap tier rather than propagating -- this
        proves that behavior holds specifically for QuotaExhaustedException,
        i.e. dedup embedding quota exhaustion degrades a tier, it does not
        crash anything."""
        dedup = DeduplicatorService()
        existing = [make_question("An existing distinct question about caching layers.")]
        candidate = make_question("A totally different question about database indexing strategies.")

        async def raise_quota(text, task_name="dedup_embedding"):
            raise QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated dedup quota exhaustion")

        with patch.object(dedup.embedding_provider, "embed_query", new=AsyncMock(side_effect=raise_quota)):
            is_dup, reason = asyncio.run(dedup.check_duplicate(candidate, existing))

        self.assertFalse(is_dup, "quota exhaustion during the semantic tier must degrade, not crash or falsely flag a duplicate")

    def test_5_quota_during_question_bank_embedding_checkpoints_waiting_for_quota(self):
        """db_loader.load_role() computes question-bank embeddings BEFORE
        opening its DB transaction -- if that raises QuotaExhaustedException,
        nothing from this attempt is persisted (no partial DB write
        occurred), so the orchestrator must just checkpoint and stop, not
        crash and not mark the role COMPLETED."""
        async def raise_quota(*args, **kwargs):
            raise QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated question-bank embedding quota exhaustion")

        patchers = self._base_patchers(load_side_effect=raise_quota)
        result = self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = self.orchestrator.jobs.get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 0, "nothing should be persisted when embedding computation itself fails before any DB write")

    def test_6_validation_quota_leaves_pipeline_jobs_status_waiting_for_quota_in_db(self):
        """Same scenario as test 2, but asserting directly against a fresh
        DB read of pipeline_jobs (not just the returned UnitResult), to
        prove the checkpoint is durably persisted, not just an in-memory
        return value."""
        call_count = {"n": 0}

        async def flaky_validate(cand, existing_bank, expected_role=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated")
            return await approve_all(cand, existing_bank, expected_role)

        patchers = self._base_patchers(validate_side_effect=flaky_validate)
        self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))

        rows = fetch_all("SELECT status FROM pipeline_jobs WHERE role=%s AND job_type='ROLE_GENERATION';", (TEST_ROLE,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "WAITING_FOR_QUOTA")


class TestStateMachineIndependence(BaseOrchestratorTest):
    """Requirements 7-12, 15: research freshness and generation completeness
    are independent axes; an incomplete generation job must resume without
    re-researching, and repeated invocations must be idempotent."""

    def _seed_fresh_research(self, source_hash="stable-hash"):
        self.orchestrator.freshness.record_change(TEST_ROLE, None, "v1.0", source_hash, None)

    def test_7_fresh_research_incomplete_generation_resumes_without_reresearch(self):
        self._seed_fresh_research()
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "KNOWLEDGE_UPDATED", mark_completed=True)
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_GENERATION", "VALIDATING", mark_started=True)

        research_mock = AsyncMock()
        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_research", new=research_mock),
            patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band",
                  new=AsyncMock(side_effect=lambda role, band, count, target_scope, domain=None:
                                 [make_question(f"Resumed Q for {band}.", experience_band=band)])),
            patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)),
        ]
        result = self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))

        research_mock.assert_not_called()
        self.assertEqual(result.status, "COMPLETED")
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 5, "resumed generation must have actually produced and loaded questions")

    def test_8_fresh_research_completed_generation_skips(self):
        self._seed_fresh_research()
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_GENERATION", "COMPLETED", mark_completed=True)

        research_mock = AsyncMock()
        generate_mock = AsyncMock()
        with patch("question_pipeline.orchestrator.research_service.execute_research", new=research_mock), \
             patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=generate_mock):
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        self.assertEqual(result.status, "SKIPPED_NO_CHANGE")
        research_mock.assert_not_called()
        generate_mock.assert_not_called()

    def test_9_stale_research_completed_generation_refreshes_with_topup(self):
        research_v1 = make_research_output("version one content")
        patchers = self._base_patchers(research_output=research_v1)
        self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))
        first_count = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))[0][0]
        self.assertEqual(first_count, 5)

        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute("UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' WHERE role=%s;", (TEST_ROLE,))
        cur.close(); conn.close()

        research_v2 = make_research_output("version TWO -- materially different content")
        new_candidates = {band: [make_question(f"Topup Q for {band}.", experience_band=band)] for band in config.EXPERIENCE_BANDS}
        patchers2 = self._base_patchers(research_output=research_v2, candidates_per_band=new_candidates)
        result = self._run(patchers2, lambda: self.orchestrator.process_role(TEST_ROLE))

        self.assertEqual(result.status, "COMPLETED")
        self.assertFalse(result.metrics["is_first_time"])
        total = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))[0][0]
        self.assertEqual(total, 10, "original 5 + 5 new top-up, not a full regeneration")

    def test_10_interrupted_generation_resumes_without_duplicate_questions(self):
        """Simulates the exact Phase 5B failure mode end-to-end: run 1's
        validation raises quota partway (some work persisted via
        _load_partial), run 2 (research still fresh) resumes and completes
        the rest -- final state must have no duplicates and must include
        the original partial batch's rows untouched."""
        call_count = {"n": 0}

        async def flaky_validate(cand, existing_bank, expected_role=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise QuotaExhaustedException("gemini", "gemini-embedding-2", "simulated")
            return await approve_all(cand, existing_bank, expected_role)

        patchers1 = self._base_patchers(validate_side_effect=flaky_validate)
        result1 = self._run(patchers1, lambda: self.orchestrator.process_role(TEST_ROLE))
        self.assertEqual(result1.status, "WAITING_FOR_QUOTA")
        original_ids = {r[0] for r in fetch_all("SELECT id FROM questions WHERE role=%s;", (TEST_ROLE,))}
        self.assertEqual(len(original_ids), 1, "exactly 1 candidate validated successfully before the quota hit")

        # Run 2: research is still fresh (is_due() is False -- nothing
        # forces a re-research), generation job is WAITING_FOR_QUOTA (not
        # complete) -- must resume via case B, not re-research.
        with patch("question_pipeline.orchestrator.research_service.execute_research", new=AsyncMock()) as research_mock, \
             patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band",
                   new=AsyncMock(side_effect=lambda role, band, count, target_scope, domain=None:
                                  [make_question(f"Resume-2 Q for {band}.", experience_band=band)])), \
             patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)):
            result2 = asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        research_mock.assert_not_called()
        self.assertEqual(result2.status, "COMPLETED")

        all_rows = fetch_all("SELECT id, question_text_normalized FROM questions WHERE role=%s;", (TEST_ROLE,))
        all_ids = {r[0] for r in all_rows}
        self.assertTrue(original_ids.issubset(all_ids), "the original pre-crash approved question must survive untouched")
        normalized_texts = [r[1] for r in all_rows]
        self.assertEqual(len(normalized_texts), len(set(normalized_texts)), "no duplicate question text after resume")

    def test_11_stuck_role_matching_conversational_ai_engineer_pattern_recovers(self):
        """Exact reproduction of the real Phase 5B stuck-role shape:
        knowledge_state=COMPLETED, pipeline_jobs ROLE_RESEARCH=
        KNOWLEDGE_UPDATED / ROLE_GENERATION=VALIDATING, ZERO questions/
        attempts/research rows loaded (the crash happened before
        db_loader.load_role() ever ran). Proves the fix makes this exact
        pattern resumable without re-research and without needing any
        manual DB surgery beyond what orchestrator.py itself does."""
        self._seed_fresh_research(source_hash="conversational-ai-engineer-like-hash")
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "KNOWLEDGE_UPDATED", mark_completed=True)
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_GENERATION", "VALIDATING", mark_started=True)
        # Confirm the precondition matches the real incident exactly.
        self.assertEqual(fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))[0][0], 0)
        self.assertEqual(fetch_all("SELECT COUNT(*) FROM question_validation_attempts WHERE role=%s;", (TEST_ROLE,))[0][0], 0)
        self.assertEqual(fetch_all("SELECT status FROM knowledge_state WHERE role=%s;", (TEST_ROLE,))[0][0], "COMPLETED")

        research_mock = AsyncMock()
        with patch("question_pipeline.orchestrator.research_service.execute_research", new=research_mock), \
             patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band",
                   new=AsyncMock(side_effect=lambda role, band, count, target_scope, domain=None:
                                  [make_question(f"Recovered Q for {band}.", experience_band=band)])), \
             patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)):
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        research_mock.assert_not_called()
        self.assertEqual(result.status, "COMPLETED")
        self.assertTrue(result.metrics["is_first_time"], "zero existing questions must be treated as first-time sizing, not a small top-up")
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 5)

    def test_12_resume_does_not_call_research_service_even_when_available(self):
        """Explicit, isolated assertion (distinct from test 7/11's broader
        flow checks): case B must never call research_service at all."""
        self._seed_fresh_research()
        self.orchestrator.jobs.upsert(TEST_ROLE, None, "ROLE_GENERATION", "FAILED", reason="simulated prior failure")

        research_mock = AsyncMock()
        with patch("question_pipeline.orchestrator.research_service.execute_research", new=research_mock), \
             patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band",
                   new=AsyncMock(side_effect=lambda role, band, count, target_scope, domain=None: [])), \
             patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)):
            asyncio.run(self.orchestrator.process_role(TEST_ROLE))

        research_mock.assert_not_called()

    def test_15_repeated_scheduler_invocation_is_idempotent(self):
        research_output = make_research_output("idempotency content")
        for _ in range(3):
            patchers = self._base_patchers(research_output=research_output)
            result = self._run(patchers, lambda: self.orchestrator.process_role(TEST_ROLE))
        # First call COMPLETED, subsequent calls SKIPPED_NO_CHANGE (fresh + complete).
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 5, "repeated invocations with no change must never duplicate or regenerate")
        job_rows = fetch_all("SELECT COUNT(*) FROM pipeline_jobs WHERE role=%s AND job_type='ROLE_GENERATION';", (TEST_ROLE,))
        self.assertEqual(job_rows[0][0], 1, "repeated invocations must upsert the same job row, not append new ones")


if __name__ == "__main__":
    unittest.main()
