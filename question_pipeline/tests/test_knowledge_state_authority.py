"""
Regression tests for the knowledge_state.status authority bug found during
the real Phase 5A rollout: db_loader.QuestionBankLoader.
_load_research_and_knowledge()'s role-only state.json -> knowledge_state
sync ran unconditionally, so when the production orchestrator called it
(via load_role(), from process_role()) AFTER freshness.record_change() had
already correctly set knowledge_state.status='COMPLETED', the sync
clobbered it back to 'RUNNING' -- the value research_service.
execute_research() (shared by both the legacy CLI and the orchestrator)
always leaves in the LOCAL state.json, since nothing in the orchestrator
path ever advances that local file any further.

Fix: load_role()/_load_research_and_knowledge() take a `sync_legacy_state`
flag (default True, preserving the original standalone "run the legacy CLI,
then load its output" workflow's behavior exactly). orchestrator.py now
passes sync_legacy_state=False at both of its load_role() call sites, since
it has already written the authoritative DB status itself via freshness.py
BEFORE the loader ever runs.

Uses the real (migrated) DB with a synthetic role name, purged in
setUp/tearDown -- same pattern as test_orchestrator.py /
test_company_rag_context.py. One test deliberately does NOT mock
research_service.execute_research (it runs for real against
MockResearchProvider under MOCK_MODE -- free, deterministic) because that
is what genuinely reproduces the bug precondition: a real local state.json
side effect left at status="RUNNING".
"""
import os
import asyncio
import unittest
from unittest.mock import patch, AsyncMock

os.environ.setdefault("MOCK_MODE", "true")

import psycopg2
from dotenv import load_dotenv

from question_pipeline.config import config
from question_pipeline.models import QuestionObject, KnowledgeChunk, QualityGateResult
from question_pipeline.state_manager import state_manager
from question_pipeline.db_loader import QuestionBankLoader
from question_pipeline.orchestrator import RolloutOrchestrator
from question_pipeline.tests.test_db_loader import FakeEmbeddingProvider

_embedding_patcher = patch("question_pipeline.db_loader.get_embedding_provider", return_value=FakeEmbeddingProvider())

load_dotenv(config.BASE_DIR.parent / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"), "connect_timeout": 10,
}

TEST_ROLE = "__KnowledgeStateAuthorityTest Role__"


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


class TestLegacyStateSyncFlagBackwardCompatibility(unittest.TestCase):
    """Requirement 4 ('legacy single-role CLI still works') at the
    db_loader level: the standalone-loader workflow (default
    sync_legacy_state=True, unchanged from before this fix) must still sync
    status from state.json exactly as it always has -- pipeline_runner.py
    itself never calls the loader, but the documented standalone "run the
    legacy CLI, then load its approved output" workflow does, and this must
    keep working unmodified."""

    def setUp(self):
        purge(TEST_ROLE)
        self.loader = QuestionBankLoader(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge(TEST_ROLE)
        state_manager.reset_role_state(TEST_ROLE)

    def test_default_sync_legacy_state_still_updates_knowledge_state_from_state_json(self):
        state_manager.update_role_state(TEST_ROLE, status="COMPLETED", accepted_questions_count=5)
        asyncio.run(self.loader.load_role(TEST_ROLE, [], []))  # sync_legacy_state defaults to True
        rows = fetch_all("SELECT status FROM knowledge_state WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "COMPLETED")

    def test_default_sync_reflects_a_non_completed_local_status_too(self):
        """'Failed/incomplete work must retain the appropriate non-completed
        status': for the legacy standalone workflow, that status genuinely
        comes from state.json, so a BLOCKED local state must sync through
        faithfully, not be silently upgraded to COMPLETED."""
        state_manager.update_role_state(TEST_ROLE, status="BLOCKED", reason="missing credentials")
        asyncio.run(self.loader.load_role(TEST_ROLE, [], []))
        rows = fetch_all("SELECT status, reason FROM knowledge_state WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], "BLOCKED")
        self.assertEqual(rows[0][1], "missing credentials")


class TestStaleLocalStateCannotOverwriteAuthoritativeDBStatus(unittest.TestCase):
    """The literal regression test for the bug (requirement 3): with
    sync_legacy_state=False (what the orchestrator now passes), a stale
    local state.json must be powerless to change the DB's already-
    authoritative status."""

    def setUp(self):
        purge(TEST_ROLE)
        self.loader = QuestionBankLoader(DB_CONFIG)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge(TEST_ROLE)
        state_manager.reset_role_state(TEST_ROLE)

    def test_sync_disabled_preserves_db_completed_status_despite_stale_running_local_state(self):
        # DB-authoritative status set first, exactly as freshness.record_change() does.
        self.orchestrator.freshness.record_change(TEST_ROLE, None, "v1.0", "hash1", None)
        rows = fetch_all("SELECT status FROM knowledge_state WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], "COMPLETED", "freshness.record_change() must set COMPLETED")

        # Reproduce the exact bug precondition: local state.json still says
        # RUNNING for this role, as research_service.execute_research()
        # always leaves it regardless of which caller invoked it.
        state_manager.update_role_state(TEST_ROLE, status="RUNNING")

        asyncio.run(self.loader.load_role(TEST_ROLE, [], [], sync_legacy_state=False))

        rows = fetch_all("SELECT status FROM knowledge_state WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], "COMPLETED", "a stale local state.json must never overwrite the authoritative DB status")

    def test_sync_enabled_default_would_still_reproduce_the_original_clobber(self):
        """Sanity check proving the test above is a genuine regression
        guard, not a tautology: leaving the flag at its old (True) default
        against the exact same stale-local-state scenario DOES still
        clobber the DB status -- demonstrating sync_legacy_state is truly
        what controls this, and that orchestrator.py's opt-out is load-
        bearing, not decorative."""
        self.orchestrator.freshness.record_change(TEST_ROLE, None, "v1.0", "hash1", None)
        state_manager.update_role_state(TEST_ROLE, status="RUNNING")

        asyncio.run(self.loader.load_role(TEST_ROLE, [], []))  # sync_legacy_state left at default True

        rows = fetch_all("SELECT status FROM knowledge_state WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], "RUNNING", "documents the pre-fix behavior orchestrator.py must opt out of")


class TestOrchestratorEndToEndKnowledgeStateAndJobStatus(unittest.TestCase):
    """Full process_role() flow WITHOUT mocking research_service.
    execute_research (it runs for real against MockResearchProvider under
    MOCK_MODE -- free, deterministic -- which is what genuinely reproduces
    the bug precondition: a real local state.json side effect left at
    status=RUNNING). Proves requirements 1 and 2: successful role research
    ends with knowledge_state.status=COMPLETED, and successful generation
    ends with the correct pipeline_jobs status."""

    def setUp(self):
        purge(TEST_ROLE)
        state_manager.reset_role_state(TEST_ROLE)
        self.orchestrator = RolloutOrchestrator(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge(TEST_ROLE)
        state_manager.reset_role_state(TEST_ROLE)

    def test_successful_role_research_and_generation_leaves_knowledge_state_completed(self):
        chunks = make_chunks()
        candidate = make_question("Full flow regression question.", experience_band="0-1")

        async def fake_generate(role, band, count, target_scope, domain=None):
            return [candidate] if band == "0-1" else []

        async def approve_all(cand, existing_bank, expected_role=None):
            return QualityGateResult(
                approved=True, quality_score=0.9, critical_failure=False,
                validation_tier="primary_gemini", validation_provider="gemini", validation_model="fake",
            )

        patchers = [
            # Deliberately NOT mocking research_service.execute_research --
            # it must run for real (against the shared MockResearchProvider
            # singleton) so it genuinely leaves local state.json at
            # status="RUNNING" for TEST_ROLE, exactly as it does in a real
            # production run. This is what makes the test a genuine
            # reproduction of the bug, not just an assertion on the fix.
            patch("question_pipeline.orchestrator.synthesis_service.synthesize", new=AsyncMock(return_value=chunks)),
            patch("question_pipeline.orchestrator.rag_service.index_chunks", new=AsyncMock(return_value=None)),
            patch("question_pipeline.orchestrator.generator_service.generate_batch_for_band", new=AsyncMock(side_effect=fake_generate)),
            patch("question_pipeline.orchestrator.validator_service.validate_question", new=AsyncMock(side_effect=approve_all)),
        ]
        for p in patchers:
            p.start()
        try:
            result = asyncio.run(self.orchestrator.process_role(TEST_ROLE))
        finally:
            for p in patchers:
                p.stop()

        self.assertEqual(result.status, "COMPLETED")

        # Sanity check: confirms the bug precondition genuinely occurred
        # (local state.json really was left at RUNNING by the real
        # execute_research() call) -- otherwise this test wouldn't actually
        # be exercising the fix at all.
        local_state = state_manager.get_role_state(TEST_ROLE)
        self.assertEqual(local_state.status, "RUNNING", "sanity check: confirms the bug precondition was genuinely reproduced")

        # Requirement 1: knowledge_state.status must be COMPLETED, not
        # clobbered back to RUNNING by db_loader's legacy-state sync.
        db_state = self.orchestrator.freshness.get_state(TEST_ROLE)
        self.assertEqual(db_state["status"], "COMPLETED")

        # Requirement 2: pipeline_jobs must show the correct COMPLETED status.
        job = self.orchestrator.jobs.get(TEST_ROLE, None, "ROLE_GENERATION")
        self.assertEqual(job["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
