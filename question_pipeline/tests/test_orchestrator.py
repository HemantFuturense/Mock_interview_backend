"""
Tests for the production rollout orchestration layer: config.py's 59-role/
24-company sets, freshness.py's DB-backed cadence/change-detection,
orchestrator.py's PipelineJobStore/RolloutOrchestrator, and migration 0002's
research_sources URL-dedup constraint.

Uses the real (migrated) DB with an obviously-synthetic role/company name,
fully cleaned up in setUp/tearDown -- same pattern as test_db_loader.py.
Expensive external calls (research/synthesis/RAG/generation/validation) are
mocked; no real Tavily/Firecrawl/Gemini calls happen in this file.
"""
import os
import json
import asyncio
import unittest
from unittest.mock import patch, AsyncMock

os.environ.setdefault("MOCK_MODE", "true")

import psycopg2
from dotenv import load_dotenv

from question_pipeline.config import config
from question_pipeline.models import QuestionObject, ResearchOutput, KnowledgeChunk
from question_pipeline.governor import QuotaExhaustedException, ProviderBlockedException
from question_pipeline.freshness import KnowledgeFreshnessTracker, bump_version
from question_pipeline.orchestrator import RolloutOrchestrator, PipelineJobStore, fetch_existing_bank_from_db
from question_pipeline.tests.test_db_loader import FakeEmbeddingProvider

# db_loader computes a real embedding per newly-inserted question. Under
# MOCK_MODE the default embedding provider is 128-dim (LocalEmbeddingProvider)
# which doesn't match the halfvec(3072) column -- patch it globally for this
# module's tests, same fix as test_db_loader.py.
_embedding_patcher = patch("question_pipeline.db_loader.get_embedding_provider", return_value=FakeEmbeddingProvider())

load_dotenv(config.BASE_DIR.parent / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"), "connect_timeout": 10,
}

TEST_ROLE = "__OrchestratorTest Role__"
TEST_COMPANY = "__OrchestratorTest Company__"


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


def purge(role=TEST_ROLE, company=None):
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM question_validation_attempts WHERE role = %s;", (role,))
    cur.execute("DELETE FROM questions WHERE role = %s;", (role,))
    cur.execute("DELETE FROM pipeline_jobs WHERE role = %s;", (role,))
    # Clear BOTH the role-only row and any company-scoped row for this role
    # (e.g. company research tests store knowledge_state with company set) --
    # a partial cleanup here previously let a TEST_COMPANY row leak across
    # test runs and made test_stale_company_research_detection flaky.
    cur.execute("DELETE FROM knowledge_state WHERE role = %s;", (role,))
    cur.execute("DELETE FROM knowledge_state WHERE role = %s;", (TEST_COMPANY,))
    cur.execute("DELETE FROM knowledge_version_history WHERE role = %s;", (role,))
    cur.execute("DELETE FROM knowledge_version_history WHERE role = %s;", (TEST_COMPANY,))
    cur.execute("DELETE FROM research_chunks WHERE role = %s;", (role,))
    cur.execute("DELETE FROM research_documents WHERE role = %s;", (role,))
    cur.close()
    conn.close()


def make_research_output(text="Test research content.") -> ResearchOutput:
    import hashlib
    return ResearchOutput(
        role=TEST_ROLE, raw_summary=text, source_references=["Test Source (https://example.com/x)"],
        source_hash=hashlib.sha256(text.encode()).hexdigest(), actual_provider="tavily_firecrawl",
    )


def make_chunks(role=TEST_ROLE) -> list:
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


class TestConfigValidation(unittest.TestCase):
    def test_59_role_configuration(self):
        self.assertEqual(len(config.PRODUCTION_ROLES), 59)
        self.assertEqual(len(set(config.PRODUCTION_ROLES)), 59, "no duplicate roles")
        self.assertIn("Machine Learning Engineer", config.PRODUCTION_ROLES)
        self.assertIn("DFT (Design for Test) Engineer", config.PRODUCTION_ROLES)

    def test_24_company_configuration(self):
        self.assertEqual(len(config.COMPANIES), 24)
        self.assertEqual(len(set(config.COMPANIES)), 24, "no duplicate companies")
        expected = {"Oracle", "Cognizant", "Adobe", "Flipkart", "Accenture", "PwC", "Tesla",
                    "Alibaba Cloud", "eBay", "EY", "Spotify", "Apple", "Google", "Deloitte",
                    "Tech Mahindra", "Nvidia", "Microsoft", "Netflix", "Amazon", "Walmart",
                    "TCS", "OpenAI", "Meta", "Broadcom"}
        self.assertEqual(set(config.COMPANIES), expected)

    def test_difficulty_and_paradigm_rules_unchanged(self):
        """Regression guard: this feature must not touch the multidimensional
        difficulty architecture or role-specific ceiling logic."""
        self.assertEqual(config.EXPERIENCE_DIFFICULTY_BOUNDS["0-1"], (1, 6))
        self.assertEqual(config.EXPERIENCE_DIFFICULTY_BOUNDS["8+"], (7, 10))
        self.assertEqual(
            config.PARADIGM_BY_EXPERIENCE_BAND,
            {"0-1": "EXECUTION", "1-2": "IMPLEMENTATION", "3-5": "ARCHITECTURE", "5-8": "STRATEGY", "8+": "DOMAIN_OWNERSHIP"},
        )


class TestBumpVersion(unittest.TestCase):
    def test_bump_version_sequence(self):
        self.assertEqual(bump_version(None), "v1.0")
        self.assertEqual(bump_version("v1.0"), "v1.1")
        self.assertEqual(bump_version("v1.9"), "v1.10")
        self.assertEqual(bump_version("not-a-version"), "v1.1")


class TestFreshnessTracker(unittest.TestCase):
    def setUp(self):
        purge()
        self.tracker = KnowledgeFreshnessTracker(DB_CONFIG)

    def tearDown(self):
        purge()

    def test_stale_role_research_detection(self):
        """Never-researched role is due."""
        self.assertTrue(self.tracker.is_due(TEST_ROLE))

    def test_fresh_role_research_skip(self):
        self.tracker.record_change(TEST_ROLE, None, "v1.0", "hash1", None)
        self.assertFalse(self.tracker.is_due(TEST_ROLE), "Just-researched role must not be due again immediately")

    def test_stale_company_research_detection(self):
        self.assertTrue(self.tracker.is_due(TEST_ROLE, company=TEST_COMPANY))
        self.tracker.record_change(TEST_ROLE, TEST_COMPANY, "v1.0", "hash1", None)
        self.assertFalse(self.tracker.is_due(TEST_ROLE, company=TEST_COMPANY))

    def test_no_change_refresh_preserves_knowledge_version(self):
        self.tracker.record_change(TEST_ROLE, None, "v1.0", "hash-same", None)
        self.tracker.record_no_change(TEST_ROLE, None, "hash-same")
        state = self.tracker.get_state(TEST_ROLE)
        self.assertEqual(state["knowledge_version"], "v1.0", "no-change refresh must not bump the version")

    def test_changed_refresh_bumps_version_and_preserves_history(self):
        self.tracker.record_change(TEST_ROLE, None, "v1.0", "hash-a", None)
        self.tracker.record_change(TEST_ROLE, None, "v1.1", "hash-b", None)
        state = self.tracker.get_state(TEST_ROLE)
        self.assertEqual(state["knowledge_version"], "v1.1")
        self.assertEqual(state["source_hash"], "hash-b")

        history = fetch_all(
            "SELECT knowledge_version, source_hash FROM knowledge_version_history WHERE role=%s ORDER BY id;",
            (TEST_ROLE,),
        )
        self.assertEqual(len(history), 2, "both versions must remain in history")
        self.assertEqual(history[0], ("v1.0", "hash-a"))
        self.assertEqual(history[1], ("v1.1", "hash-b"))


class TestResearchSourceDeduplication(unittest.TestCase):
    def setUp(self):
        self._urls = []

    def tearDown(self):
        conn = db_connect()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("DELETE FROM research_sources WHERE url LIKE %s;", ("%orchestrator-dedup-test%",))
        cur.close()
        conn.close()

    def test_repeated_insertion_of_equivalent_url_creates_no_duplicate(self):
        conn = db_connect()
        conn.autocommit = True
        cur = conn.cursor()
        variants = [
            "https://example.com/orchestrator-dedup-test/page",
            "http://example.com/orchestrator-dedup-test/page",
            "https://example.com/orchestrator-dedup-test/page/",
            "https://example.com/orchestrator-dedup-test/page#section",
        ]
        ids = []
        for v in variants:
            cur.execute(
                "INSERT INTO research_sources (url, title, discovery_provider) VALUES (%s, 'T', 'tavily') "
                "ON CONFLICT (url_normalized) DO NOTHING RETURNING id;",
                (v,),
            )
            row = cur.fetchone()
            if row:
                ids.append(row[0])
        cur.execute("SELECT COUNT(*) FROM research_sources WHERE url LIKE %s;", ("%orchestrator-dedup-test%",))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        self.assertEqual(len(ids), 1, "only the first variant should have actually inserted a row")
        self.assertEqual(count, 1, "all 4 equivalent URL variants must resolve to exactly one research_sources row")


class TestPipelineJobStore(unittest.TestCase):
    def setUp(self):
        purge()
        self.jobs = PipelineJobStore(DB_CONFIG)

    def tearDown(self):
        purge()

    def test_checkpoint_progression_and_resume(self):
        self.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "PENDING", mark_started=True)
        self.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "RESEARCHING")
        self.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "RESEARCHED")
        job = self.jobs.get(TEST_ROLE, None, "ROLE_RESEARCH")
        self.assertEqual(job["status"], "RESEARCHED")

        rows = fetch_all("SELECT COUNT(*) FROM pipeline_jobs WHERE role=%s AND job_type='ROLE_RESEARCH';", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 1, "progression must upsert the SAME row, not append new ones")

    def test_quota_pause_marks_waiting_for_quota_with_reason(self):
        self.jobs.upsert(TEST_ROLE, None, "ROLE_GENERATION", "WAITING_FOR_QUOTA", reason="Quota exhausted for gemini/gemini-3.1-flash-lite: simulated")
        job = self.jobs.get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        self.assertIn("Quota exhausted", job["reason"])

    def test_blocked_provider_marks_blocked_with_clear_reason(self):
        self.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "BLOCKED", reason="TAVILY_API_KEY is not configured or is a placeholder in .env.")
        job = self.jobs.get(TEST_ROLE, None, "ROLE_RESEARCH")
        self.assertEqual(job["status"], "BLOCKED")
        self.assertIn("TAVILY_API_KEY", job["reason"])

    def test_failed_role_does_not_affect_other_roles_job_rows(self):
        other_role = TEST_ROLE + "_other"
        try:
            self.jobs.upsert(TEST_ROLE, None, "ROLE_RESEARCH", "FAILED", reason="simulated failure")
            self.jobs.upsert(other_role, None, "ROLE_RESEARCH", "COMPLETED", mark_completed=True)

            self.assertEqual(self.jobs.get(TEST_ROLE, None, "ROLE_RESEARCH")["status"], "FAILED")
            self.assertEqual(self.jobs.get(other_role, None, "ROLE_RESEARCH")["status"], "COMPLETED")
        finally:
            purge(other_role)


class TestFetchExistingBankFromDB(unittest.TestCase):
    def setUp(self):
        purge()

    def tearDown(self):
        purge()

    def test_returns_empty_for_role_with_no_questions(self):
        bank = fetch_existing_bank_from_db(DB_CONFIG, TEST_ROLE)
        self.assertEqual(bank, [])

    def test_reconstructs_questions_from_db(self):
        conn = db_connect()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO questions (role, question_text, experience_band, difficulty, technical_depth,
                problem_complexity, architecture_complexity, troubleshooting, business_complexity,
                decision_making, leadership_ownership, paradigm, scope)
            VALUES (%s, 'A seeded test question for DB bank reconstruction.', '3-5', 6, 6,6,6,6,4,5,3, 'ARCHITECTURE', 'UNIVERSAL');
            """,
            (TEST_ROLE,),
        )
        cur.close()
        conn.close()

        bank = fetch_existing_bank_from_db(DB_CONFIG, TEST_ROLE)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0].question, "A seeded test question for DB bank reconstruction.")
        self.assertEqual(bank[0].role, TEST_ROLE)


class TestOrchestratorEndToEndWithMocks(unittest.TestCase):
    """Exercises process_role() end-to-end with the expensive external calls
    mocked (no real Tavily/Firecrawl/Gemini). Proves: first-run generation,
    no-change skip (no regeneration), changed-knowledge incremental top-up
    with DB dedup, quota pause, provider-blocked behavior, and that rerunning
    with unchanged content produces no duplicates."""

    def setUp(self):
        purge()
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge()

    def _patch_pipeline(self, research_output=None, chunks=None, candidates_per_band=None, validate_side_effect=None):
        """Returns a contextlib.ExitStack-like list of active patchers for
        the full research->synthesis->rag->generate->validate chain."""
        research_output = research_output or make_research_output()
        chunks = chunks if chunks is not None else make_chunks()
        candidates_per_band = candidates_per_band if candidates_per_band is not None else {
            band: [make_question(f"Question for {band} band, candidate {i}.", experience_band=band)]
            for i, band in enumerate(config.EXPERIENCE_BANDS)
        }

        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_research",
                  new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": research_output})),
            patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=chunks)),
            patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)),
        ]

        async def fake_generate(role, band, count, target_scope, domain=None):
            return list(candidates_per_band.get(band, []))

        patchers.append(patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=AsyncMock(side_effect=fake_generate)))

        if validate_side_effect is not None:
            patchers.append(patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=validate_side_effect)))
        else:
            from question_pipeline.models import QualityGateResult

            async def approve_all(candidate, existing_bank, expected_role=None):
                return QualityGateResult(
                    approved=True, quality_score=0.9, critical_failure=False,
                    validation_tier="primary_gemini", validation_provider="gemini", validation_model="fake-model",
                )

            patchers.append(patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)))

        for p in patchers:
            p.start()
        return patchers

    def _stop(self, patchers):
        for p in patchers:
            p.stop()

    def test_first_run_generates_and_loads_questions(self):
        patchers = self._patch_pipeline()
        try:
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            self._stop(patchers)

        self.assertEqual(result.status, "COMPLETED")
        self.assertTrue(result.metrics["is_first_time"])
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 5, "one candidate per band, all approved by the mock validator")

    def test_no_change_refresh_does_not_regenerate_questions(self):
        research_output = make_research_output("stable content")
        patchers = self._patch_pipeline(research_output=research_output)
        try:
            asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            self._stop(patchers)
        first_count = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))[0][0]

        # Second run: IDENTICAL research content (same hash) -> must be a no-op.
        gen_mock = AsyncMock()
        patchers2 = [
            patch("question_pipeline.orchestrator.research_service.execute_research",
                  new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": research_output})),
            patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=gen_mock),
        ]
        for p in patchers2:
            p.start()
        try:
            result2 = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            for p in patchers2:
                p.stop()

        self.assertEqual(result2.status, "SKIPPED_NO_CHANGE")
        gen_mock.assert_not_called()
        second_count = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))[0][0]
        self.assertEqual(first_count, second_count, "no-change refresh must not add or remove any questions")

    def test_changed_knowledge_does_incremental_top_up_not_full_regeneration(self):
        research_v1 = make_research_output("version one content")
        patchers = self._patch_pipeline(research_output=research_v1)
        try:
            asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            self._stop(patchers)
        original_ids = {r[0] for r in fetch_all("SELECT id FROM questions WHERE role=%s;", (TEST_ROLE,))}
        self.assertEqual(len(original_ids), 5)

        # Force it due again regardless of cadence, for this test.
        conn = db_connect(); conn.autocommit = True; cur = conn.cursor()
        cur.execute("UPDATE knowledge_state SET last_researched_at = now() - interval '400 days' WHERE role=%s;", (TEST_ROLE,))
        cur.close(); conn.close()

        research_v2 = make_research_output("version TWO -- materially different content")
        new_candidates = {
            band: [make_question(f"NEW top-up question for {band}.", experience_band=band)]
            for band in config.EXPERIENCE_BANDS
        }
        patchers2 = self._patch_pipeline(research_output=research_v2, candidates_per_band=new_candidates)
        try:
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            self._stop(patchers2)

        self.assertEqual(result.status, "COMPLETED")
        self.assertFalse(result.metrics["is_first_time"])

        all_rows = fetch_all("SELECT id FROM questions WHERE role=%s;", (TEST_ROLE,))
        all_ids = {r[0] for r in all_rows}
        self.assertTrue(original_ids.issubset(all_ids), "the original approved questions must still be present, untouched")
        self.assertEqual(len(all_ids), 10, "original 5 + 5 new top-up questions, NOT a full regeneration")

        history = fetch_all("SELECT knowledge_version FROM knowledge_version_history WHERE role=%s ORDER BY id;", (TEST_ROLE,))
        self.assertEqual([h[0] for h in history], ["v1.0", "v1.1"])

    def test_rerunning_unchanged_rollout_produces_no_duplicates(self):
        research_output = make_research_output("idempotency check content")
        for _ in range(2):
            patchers = self._patch_pipeline(research_output=research_output)
            try:
                asyncio.run(self.orchestrator.process_role(TEST_ROLE))
            finally:
                self._stop(patchers)
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 5, "running the same unchanged content twice must not duplicate anything")

    def test_quota_exhaustion_pauses_and_checkpoints_partial_progress(self):
        call_count = {"n": 0}

        async def flaky_generate(role, band, count, target_scope, domain=None):
            call_count["n"] += 1
            if band == "3-5":
                raise QuotaExhaustedException("gemini", "gemini-3.1-flash-lite", "simulated 429")
            return [make_question(f"Question for {band}.", experience_band=band)]

        research_output = make_research_output()
        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_research",
                  new=AsyncMock(return_value={"status": "SUCCESS", "role": TEST_ROLE, "research_output": research_output})),
            patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=make_chunks())),
            patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)),
            patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=AsyncMock(side_effect=flaky_generate)),
        ]
        from question_pipeline.models import QualityGateResult

        async def approve_all(candidate, existing_bank, expected_role=None):
            return QualityGateResult(approved=True, quality_score=0.9, critical_failure=False,
                                      validation_tier="primary_gemini", validation_provider="gemini", validation_model="fake")
        patchers.append(patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)))
        for p in patchers:
            p.start()
        try:
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(result.status, "WAITING_FOR_QUOTA")
        job = PipelineJobStore(DB_CONFIG).get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "WAITING_FOR_QUOTA")
        # Bands processed before the quota hit (0-1, 1-2) must have been
        # checkpointed via the partial load, not discarded.
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 2, "the 2 bands processed before quota exhaustion must be preserved, not lost or faked")

    def test_blocked_provider_does_not_silently_substitute(self):
        blocked_exc = ProviderBlockedException("tavily_firecrawl", "tavily", "TAVILY_API_KEY is not configured or is a placeholder in .env.")
        patchers = [
            patch("question_pipeline.orchestrator.research_service.execute_research", new=AsyncMock(side_effect=blocked_exc)),
        ]
        for p in patchers:
            p.start()
        try:
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("TAVILY_API_KEY", result.reason)
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 0, "a blocked role must never produce fabricated questions")

    def test_dry_run_makes_no_db_writes_and_no_mocked_calls(self):
        gen_mock = AsyncMock()
        with patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=gen_mock):
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE, dry_run=True))
        self.assertEqual(result.status, "DRY_RUN")
        gen_mock.assert_not_called()
        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 0)
        jobs = fetch_all("SELECT COUNT(*) FROM pipeline_jobs WHERE role=%s;", (TEST_ROLE,))
        self.assertEqual(jobs[0][0], 0, "dry-run must not write any job checkpoint either")


if __name__ == "__main__":
    unittest.main()
