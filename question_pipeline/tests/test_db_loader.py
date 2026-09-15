"""
Tests for question_pipeline/db_loader.py.

These are integration tests against the REAL migration-0001 schema on the
real database (no mock DB exists, and psycopg2/SQL/ON CONFLICT/halfvec
behavior can't be meaningfully verified against a fake connection). To keep
this safe:
  - Every test uses an obviously-synthetic role name (prefixed
    "__LoaderTest") that can never collide with a real pilot role.
  - setUp AND tearDown both purge that role from every migration-0001
    table, so a failed test never leaves residue and the DB is left exactly
    as found regardless of pass/fail.
  - No legacy table is ever touched by this file.
  - MOCK_MODE=true (set by tests/__init__.py) means embeddings use the
    local TF-IDF provider -- no real API calls, no cost, deterministic.
"""
import os
import json
import asyncio
import unittest
from unittest.mock import patch

os.environ.setdefault("MOCK_MODE", "true")

import psycopg2
from dotenv import load_dotenv

from question_pipeline.config import config
from question_pipeline.models import QuestionObject, ValidationAttemptRecord
from question_pipeline.db_loader import QuestionBankLoader, normalize_question_text
from question_pipeline.providers.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dimension-correct (3072, matching the real
    gemini-embedding-2 column width) fake embedder -- avoids both real API
    cost and the default LocalEmbeddingProvider's 128-dim mismatch against
    the halfvec(3072) column."""

    provider_name_value = "fake_embedding"
    model_name = "fake-embedding-3072"

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    async def embed_texts(self, texts, task_name="rag_embedding"):
        return [[0.001 * ((hash((t, i)) % 1000)) for i in range(3072)] for t in texts]

    async def embed_query(self, text, task_name="query_embedding"):
        return (await self.embed_texts([text]))[0]


# Patch at the point db_loader imports it from, for every test in this file.
_embedding_patcher = patch("question_pipeline.db_loader.get_embedding_provider", return_value=FakeEmbeddingProvider())

load_dotenv(config.BASE_DIR.parent / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"), "connect_timeout": 10,
}

TEST_ROLE = "__LoaderTest Machine Learning Engineer__"


def make_question(**overrides) -> QuestionObject:
    base = dict(
        question="Loader test: diagnose why validation accuracy is high but production precision collapsed.",
        role=TEST_ROLE, experience_band="3-5", difficulty=7,
        technical_depth=7, problem_complexity=7, architecture_complexity=6,
        troubleshooting=7, business_complexity=4, decision_making=5, leadership_ownership=3,
        question_type="technical", paradigm="ARCHITECTURE",
        mandatory_skills=["Monitoring", "Data Validation"],
        scope="DOMAIN", domains=["FinTech"], applicable_companies=["Stripe", "PayPal"],
        source_references=["Loader Test Source (https://example.com/loader-test-source)"],
    )
    base.update(overrides)
    return QuestionObject(**base)


def make_attempt(question_text: str, approved: bool, **overrides) -> ValidationAttemptRecord:
    base = dict(
        role=TEST_ROLE, candidate_question_text=question_text, approved=approved,
        quality_score=0.9 if approved else 0.4, critical_failure=not approved,
        rejection_reasons=[] if approved else ["trivial_content"],
        validation_tier="primary_gemini", validation_provider="gemini", validation_model="fake-model",
    )
    base.update(overrides)
    return ValidationAttemptRecord(**base)


def db_connect():
    return psycopg2.connect(**DB_CONFIG)


def purge_test_role(role: str = TEST_ROLE):
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DELETE FROM question_validation_attempts WHERE role = %s;", (role,))
    cur.execute("DELETE FROM questions WHERE role = %s;", (role,))  # cascades to companies/domains/skills/sources/embeddings
    cur.execute("DELETE FROM knowledge_state WHERE role = %s;", (role,))
    cur.execute("DELETE FROM knowledge_version_history WHERE role = %s;", (role,))
    cur.execute("DELETE FROM research_chunks WHERE role = %s;", (role,))
    cur.execute("DELETE FROM research_documents WHERE role = %s;", (role,))
    cur.close()
    conn.close()


def fetch_all(sql, params=()):
    conn = db_connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


class TestDBLoader(unittest.TestCase):
    def setUp(self):
        purge_test_role()
        self.loader = QuestionBankLoader(DB_CONFIG)
        _embedding_patcher.start()

    def tearDown(self):
        _embedding_patcher.stop()
        purge_test_role()

    def test_approved_question_is_inserted(self):
        q = make_question()
        report = asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))

        self.assertEqual(report.questions_inserted, 1)
        # There is no `approved` column on `questions` -- presence in this
        # table IS the approval signal (rejected candidates never reach it).
        rows = fetch_all("SELECT question_text, role FROM questions WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], q.question)

    def test_rejected_question_is_excluded_from_questions_table(self):
        approved = make_question(question="Approved loader-test question about drift monitoring.")
        rejected_text = "Rejected loader-test question that never became a QuestionObject."

        report = asyncio.run(self.loader.load_role(
            TEST_ROLE, [approved],
            [make_attempt(approved.question, approved=True), make_attempt(rejected_text, approved=False)],
        ))

        self.assertEqual(report.questions_inserted, 1)
        rows = fetch_all("SELECT question_text FROM questions WHERE role = %s;", (TEST_ROLE,))
        texts = [r[0] for r in rows]
        self.assertIn(approved.question, texts)
        self.assertNotIn(rejected_text, texts, "A rejected candidate must never appear in `questions`")

        attempt_rows = fetch_all(
            "SELECT candidate_question_text, approved, question_id FROM question_validation_attempts WHERE role = %s;",
            (TEST_ROLE,),
        )
        rejected_row = next(r for r in attempt_rows if r[0] == rejected_text)
        self.assertFalse(rejected_row[1])
        self.assertIsNone(rejected_row[2], "A rejected attempt must not reference any questions.id")

    def test_companies_domains_skills_relationships(self):
        q = make_question(applicable_companies=["Stripe", "PayPal", "Adyen"], domains=["FinTech", "Payments"],
                           mandatory_skills=["Feature Engineering", "Python"])
        asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))

        qid = fetch_all("SELECT id FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        companies = {r[0] for r in fetch_all("SELECT company_name FROM question_companies WHERE question_id = %s;", (qid,))}
        domains = {r[0] for r in fetch_all("SELECT domain_name FROM question_domains WHERE question_id = %s;", (qid,))}
        skills = {r[0] for r in fetch_all("SELECT skill_name FROM question_skills WHERE question_id = %s;", (qid,))}

        self.assertEqual(companies, {"Stripe", "PayPal", "Adyen"})
        self.assertEqual(domains, {"FinTech", "Payments"})
        self.assertEqual(skills, {"Feature Engineering", "Python"})

    def test_provenance_raw_reference_and_chunk_linkage(self):
        # 1. Raw string provenance (no matching research_chunks row exists)
        q_raw = make_question(source_references=["Some Interview Guide (https://example.com/guide)"])
        asyncio.run(self.loader.load_role(TEST_ROLE, [q_raw], []))
        qid = fetch_all("SELECT id FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        src_rows = fetch_all("SELECT source_reference, research_chunk_id FROM question_sources WHERE question_id = %s;", (qid,))
        self.assertEqual(len(src_rows), 1)
        self.assertEqual(src_rows[0][0], "Some Interview Guide (https://example.com/guide)")
        self.assertIsNone(src_rows[0][1])
        purge_test_role()

        # 2. Chunk-linked provenance: manually seed a research_chunks row
        # (bypassing file loading, which is exercised separately) then
        # confirm the loader links to it via chunk_id parsed from the
        # QuestionObject's source_references string.
        conn = db_connect()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO research_documents (role, research_provider, raw_summary) VALUES (%s, 'test', 'test') RETURNING id;",
            (TEST_ROLE,),
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO research_chunks (chunk_id, research_document_id, role, topic, text) "
            "VALUES ('loader_test_chunk_1', %s, %s, 'Test Topic', 'Test chunk text.') RETURNING id;",
            (doc_id, TEST_ROLE),
        )
        chunk_pk = cur.fetchone()[0]
        cur.close()
        conn.close()

        q_chunk = make_question(source_references=["RAG Chunk: loader_test_chunk_1 | Topic: Test Topic"])
        asyncio.run(self.loader.load_role(TEST_ROLE, [q_chunk], []))
        qid2 = fetch_all("SELECT id FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        src_rows2 = fetch_all("SELECT source_reference, research_chunk_id FROM question_sources WHERE question_id = %s;", (qid2,))
        self.assertEqual(len(src_rows2), 1)
        self.assertEqual(src_rows2[0][1], chunk_pk, "Provenance must link to the real research_chunks row via chunk_id")

    def test_validation_attempts_logged_for_both_outcomes(self):
        q = make_question()
        attempts = [
            make_attempt(q.question, approved=True, quality_score=0.95, validation_tier="primary_gemini"),
            make_attempt("A different rejected candidate text.", approved=False, quality_score=0.3,
                          rejection_reasons=["wrong_role", "hallucinated_technology"]),
        ]
        report = asyncio.run(self.loader.load_role(TEST_ROLE, [q], attempts))
        self.assertEqual(report.validation_attempts_inserted, 2)

        rows = fetch_all(
            "SELECT candidate_question_text, approved, quality_score, rejection_reasons, validation_tier, "
            "validation_provider, validation_model FROM question_validation_attempts WHERE role = %s ORDER BY id;",
            (TEST_ROLE,),
        )
        self.assertEqual(len(rows), 2)
        approved_row = next(r for r in rows if r[1])
        rejected_row = next(r for r in rows if not r[1])
        self.assertAlmostEqual(float(approved_row[2]), 0.95, places=2)
        self.assertEqual(json.loads(rejected_row[3]) if isinstance(rejected_row[3], str) else rejected_row[3],
                          ["wrong_role", "hallucinated_technology"])
        self.assertEqual(approved_row[4], "primary_gemini")
        self.assertEqual(approved_row[5], "gemini")

    def test_embeddings_are_stored_for_new_questions(self):
        q = make_question()
        report = asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))
        self.assertEqual(report.question_embeddings_inserted, 1)

        qid = fetch_all("SELECT id FROM questions WHERE role = %s;", (TEST_ROLE,))[0][0]
        rows = fetch_all(
            "SELECT embedding_provider, embedding_model, (embedding IS NOT NULL) FROM question_embeddings WHERE question_id = %s;",
            (qid,),
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0][0])
        self.assertIsNotNone(rows[0][1])
        self.assertTrue(rows[0][2])

    def test_duplicate_reload_is_idempotent(self):
        q = make_question()
        report1 = asyncio.run(self.loader.load_role(TEST_ROLE, [q], [make_attempt(q.question, approved=True)]))
        self.assertEqual(report1.questions_inserted, 1)
        self.assertEqual(report1.question_embeddings_inserted, 1)

        # Re-run with the SAME question (no `since` cutoff this time means
        # we'd naively re-log the attempt too if we passed it again -- so
        # to test question-level idempotency specifically we reload with an
        # empty attempts list, and separately confirm attempt idempotency
        # via the `since` parameter in the next test.
        report2 = asyncio.run(self.loader.load_role(TEST_ROLE, [q], []))
        self.assertEqual(report2.questions_inserted, 0, "Re-inserting the same question must be a no-op")
        self.assertEqual(report2.questions_already_present, 1)
        self.assertEqual(report2.question_companies_inserted, 0)
        self.assertEqual(report2.question_embeddings_inserted, 0)

        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 1, "Must still be exactly one row, not two")

    def test_since_cutoff_prevents_duplicate_attempt_logging_on_rerun(self):
        q = make_question()
        cutoff_before_first_load = "2020-01-01T00:00:00+00:00"
        attempt = make_attempt(q.question, approved=True)

        asyncio.run(self.loader.load_role(TEST_ROLE, [q], [attempt], since=cutoff_before_first_load))
        rows_after_first = fetch_all("SELECT COUNT(*) FROM question_validation_attempts WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows_after_first[0][0], 1)

        # Re-run scoped to "since now" (after the attempt's own validated_at) -- must load nothing new.
        import datetime
        future_cutoff = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report2 = asyncio.run(self.loader.load_role(TEST_ROLE, [q], [attempt], since=future_cutoff))
        self.assertEqual(report2.validation_attempts_inserted, 0)
        rows_after_second = fetch_all("SELECT COUNT(*) FROM question_validation_attempts WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows_after_second[0][0], 1, "since cutoff must prevent re-logging the same attempt")

    def test_transaction_rolls_back_completely_on_failure(self):
        good = make_question(question="A perfectly valid loader-test question that should NOT persist.")
        # QuestionObject itself doesn't constrain `paradigm` to the 5 valid
        # values (only the DB CHECK does), so this passes Pydantic but must
        # be rejected by questions_paradigm_check at insert time -- a
        # genuine DB-level failure, not a fixture-construction error.
        bad = make_question(question="A loader-test question with an invalid paradigm value.", paradigm="NOT_A_REAL_PARADIGM")

        with self.assertRaises(Exception):
            asyncio.run(self.loader.load_role(TEST_ROLE, [good, bad], []))

        rows = fetch_all("SELECT COUNT(*) FROM questions WHERE role = %s;", (TEST_ROLE,))
        self.assertEqual(rows[0][0], 0, "A failure partway through must roll back the ENTIRE load, including the earlier valid question")

    def test_legacy_tables_are_never_touched(self):
        """Static guard: the loader's SQL must never reference a legacy
        table (FROM/INTO/UPDATE/JOIN). Mentions in comments/docstrings
        explaining what NOT to touch are fine and expected."""
        import inspect
        import re as _re
        import question_pipeline.db_loader as loader_module
        source = inspect.getsource(loader_module)
        for legacy_table in ["interview_questions", "pre_generated_questions", "interview_data",
                              "students", "student_resumes", "session_metadata", "admin_users"]:
            pattern = rf"\b(FROM|INTO|UPDATE|JOIN)\s+{legacy_table}\b"
            self.assertIsNone(_re.search(pattern, source, _re.IGNORECASE),
                               f"Found SQL referencing legacy table {legacy_table!r}")


if __name__ == "__main__":
    unittest.main()
