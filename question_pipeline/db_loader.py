"""
Loads one role's local pipeline artifacts (approved questions, validation
attempts, research/synthesis/knowledge state) into the Postgres schema
created by migrations/0001_question_bank_and_knowledge_layer.sql.

Design:
  - Reads from the SAME local artifacts the pipeline already produces
    (question_pipeline/data/question_bank.json, validation_attempts.json,
    knowledge_base/{slug}_research.json, {slug}_synthesized.json,
    vector_store.json, state.json) rather than being wired into the live
    generation loop. This keeps the loader a standalone, rerunnable step --
    "run the pipeline, then load whatever's approved" -- decoupled from
    pipeline internals, and testable with plain fixtures.
  - Writes ONLY to the 13 migration-0001 tables. Never references, reads,
    or writes any legacy table (interview_questions, pre_generated_questions,
    interview_data, students, etc.).
  - One transaction per load_role() call: any failure rolls back the whole
    load for that role, so the DB is never left half-populated.
  - Idempotent for questions/chunks/companies/domains/skills/sources: these
    rely on the schema's own UNIQUE constraints (ON CONFLICT DO NOTHING),
    so re-running the loader against unchanged artifacts inserts nothing
    new and is always safe to repeat.
  - Idempotent for question_validation_attempts via the `since` parameter:
    that table intentionally has no natural unique key (it's meant to log
    every attempt, including ones an LLM might genuinely repeat), so the
    caller scopes each load to the attempts produced by ONE pipeline run
    (pass `since=<a timestamp captured right before that run started>`)
    rather than the loader guessing which historical rows are "new".
  - question_embeddings: the pipeline does not persist a per-question
    embedding anywhere today (dedup only keeps one in memory, transiently).
    The loader computes one embedding per newly-inserted question via the
    configured EMBEDDING_PROVIDER/EMBEDDING_MODEL (the same real provider
    RAG already uses) and stores it -- this is embedding text that was
    already approved, not generating new question content.
"""
import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import psycopg2

from .config import config
from .models import QuestionObject, ValidationAttemptRecord
from .providers.factory import get_embedding_provider


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).lower()


def normalize_question_text(text: str) -> str:
    """Mirrors the DB's own `questions.question_text_normalized` generated
    column exactly, so lookups against already-inserted rows agree with it."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def vector_literal(values: List[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


_SOURCE_REF_RE = re.compile(r"^(.*?)\s*\((https?://[^\s)]+)\)\s*$")
_CHUNK_ID_RE = re.compile(r"RAG Chunk:\s*(\S+)")


@dataclass
class LoadReport:
    role: str
    questions_inserted: int = 0
    questions_already_present: int = 0
    question_companies_inserted: int = 0
    question_domains_inserted: int = 0
    question_skills_inserted: int = 0
    question_sources_inserted: int = 0
    question_embeddings_inserted: int = 0
    validation_attempts_inserted: int = 0
    research_documents_inserted: int = 0
    research_chunks_inserted: int = 0
    research_sources_inserted: int = 0
    knowledge_state_upserted: bool = False
    inserted_question_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class QuestionBankLoader:
    """Loads one role's approved questions + provenance + validation
    attempts into the migration-0001 schema. Legacy tables are never
    touched -- this class contains no SQL referencing them."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config

    async def load_role(
        self,
        role: str,
        question_bank: List[QuestionObject],
        validation_attempts: List[ValidationAttemptRecord],
        since: Optional[str] = None,
        sync_legacy_state: bool = True,
    ) -> LoadReport:
        """
        question_bank: QuestionObjects to consider for insertion into
            `questions` (typically the CURRENT approved bank for `role`;
            entries for other roles are ignored).
        validation_attempts: ValidationAttemptRecords to log into
            `question_validation_attempts` (approved AND rejected).
        since: ISO timestamp; only attempts with validated_at > since are
            loaded. Pass the time you captured right before running the
            pipeline so a load only covers that run's output, not the
            whole historical attempts file.
        sync_legacy_state: if True (default -- preserves the original
            standalone-loader behavior for "run the legacy single-role CLI,
            then load its output"), also syncs question_pipeline/data/
            state.json's status/cadence for this role into knowledge_state.
            Pass False when the caller has ALREADY written the
            authoritative DB-backed status for this role via freshness.py
            (this is what orchestrator.py's RolloutOrchestrator does) --
            the local state.json is a side effect of the research_service
            call shared by both the legacy CLI and the orchestrator, and it
            does NOT reflect the orchestrator's own final status (it's left
            at "RUNNING" since nothing in the orchestrator path ever
            advances the *local* file further), so syncing it here would
            silently clobber a correct COMPLETED status back to a stale
            RUNNING one. See db_loader's module docstring and
            _load_research_and_knowledge() for the full rationale.
        """
        report = LoadReport(role=role)
        role_questions = [q for q in question_bank if q.role == role]
        role_attempts = [a for a in validation_attempts if a.role == role and (since is None or a.validated_at > since)]

        # Compute embeddings for candidate-new questions BEFORE opening the
        # DB transaction (network I/O should not hold a transaction open).
        # We embed all of them; ones that turn out to already exist in the
        # DB simply have their freshly-computed embedding discarded below.
        embeddings_by_text: Dict[str, Tuple[List[float], str, str]] = {}
        if role_questions:
            embeddings_by_text = await self._compute_embeddings(role_questions)

        conn = psycopg2.connect(**self.db_config)
        try:
            conn.autocommit = False
            cur = conn.cursor()

            research_document_id, chunk_id_to_pk = self._load_research_and_knowledge(
                cur, role, report, sync_legacy_state=sync_legacy_state
            )

            question_id_by_norm_text: Dict[str, int] = {}
            for q in role_questions:
                qid, was_new = self._load_question(cur, q, report)
                question_id_by_norm_text[normalize_question_text(q.question)] = qid
                if was_new:
                    self._load_question_relations(cur, qid, q, chunk_id_to_pk, report)
                    emb = embeddings_by_text.get(q.question)
                    if emb:
                        self._load_question_embedding(cur, qid, emb, report)

            self._load_validation_attempts(cur, role, role_attempts, question_id_by_norm_text, report)

            conn.commit()
        except Exception as e:
            conn.rollback()
            report.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            conn.close()
        return report

    def load_company_research(self, company: str) -> LoadReport:
        """Loads a standalone company research profile (research_documents,
        research_sources, research_chunks tagged company=<name>) into the DB.
        No questions/embeddings/validation attempts are touched by this call
        -- it only makes company KNOWLEDGE available for RAG retrieval during
        role generation; question generation itself still only happens via
        load_role() for an actual role."""
        report = LoadReport(role=company)
        conn = psycopg2.connect(**self.db_config)
        try:
            conn.autocommit = False
            cur = conn.cursor()
            self._load_research_and_knowledge(cur, company, report, company=company)
            conn.commit()
        except Exception as e:
            conn.rollback()
            report.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            conn.close()
        return report

    # ------------------------------------------------------------------
    async def _compute_embeddings(self, questions: List[QuestionObject]) -> Dict[str, Tuple[List[float], str, str]]:
        provider = get_embedding_provider()
        texts = [q.question for q in questions]
        vectors = await provider.embed_texts(texts, task_name="question_bank_embedding")
        model_name = getattr(provider, "model_name", None) or config.EMBEDDING_MODEL
        return {q.question: (vec, provider.provider_name, model_name) for q, vec in zip(questions, vectors)}

    def _load_question(self, cur, q: QuestionObject, report: LoadReport):
        cur.execute(
            """
            INSERT INTO questions
                (role, question_text, question_type, experience_band, difficulty,
                 technical_depth, problem_complexity, architecture_complexity, troubleshooting,
                 business_complexity, decision_making, leadership_ownership, paradigm, scope,
                 knowledge_version, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (role, question_text_normalized) DO NOTHING
            RETURNING id;
            """,
            (
                q.role, q.question, q.question_type, q.experience_band, q.difficulty,
                q.technical_depth, q.problem_complexity, q.architecture_complexity, q.troubleshooting,
                q.business_complexity, q.decision_making, q.leadership_ownership, q.paradigm, q.scope,
                q.knowledge_version, q.generated_at,
            ),
        )
        row = cur.fetchone()
        if row:
            report.questions_inserted += 1
            report.inserted_question_ids.append(row[0])
            return row[0], True

        report.questions_already_present += 1
        cur.execute(
            "SELECT id FROM questions WHERE role = %s AND question_text_normalized = %s;",
            (q.role, normalize_question_text(q.question)),
        )
        return cur.fetchone()[0], False

    def _load_question_relations(self, cur, qid: int, q: QuestionObject, chunk_id_to_pk: Dict[str, int], report: LoadReport):
        for company in q.applicable_companies:
            if not company or not company.strip():
                continue
            cur.execute(
                "INSERT INTO question_companies (question_id, company_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (qid, company.strip()),
            )
            report.question_companies_inserted += cur.rowcount

        for domain in q.domains:
            if not domain or not domain.strip():
                continue
            cur.execute(
                "INSERT INTO question_domains (question_id, domain_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (qid, domain.strip()),
            )
            report.question_domains_inserted += cur.rowcount

        for skill in q.mandatory_skills:
            if not skill or not skill.strip():
                continue
            cur.execute(
                "INSERT INTO question_skills (question_id, skill_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (qid, skill.strip()),
            )
            report.question_skills_inserted += cur.rowcount

        for ref in q.source_references:
            chunk_match = _CHUNK_ID_RE.search(ref or "")
            chunk_pk = None
            if chunk_match:
                chunk_key = chunk_match.group(1)
                chunk_pk = chunk_id_to_pk.get(chunk_key)
                if chunk_pk is None:
                    # Not loaded earlier in THIS call (e.g. research for this
                    # role was loaded in a previous, separate load_role()
                    # call) -- fall back to looking it up directly, so
                    # provenance linkage still works across incremental runs.
                    cur.execute("SELECT id FROM research_chunks WHERE chunk_id = %s;", (chunk_key,))
                    row = cur.fetchone()
                    chunk_pk = row[0] if row else None
            if chunk_pk:
                cur.execute(
                    """
                    INSERT INTO question_sources (question_id, research_chunk_id)
                    VALUES (%s, %s)
                    ON CONFLICT (question_id, research_chunk_id) WHERE research_chunk_id IS NOT NULL
                    DO NOTHING;
                    """,
                    (qid, chunk_pk),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO question_sources (question_id, source_reference)
                    VALUES (%s, %s)
                    ON CONFLICT (question_id, source_reference)
                        WHERE source_reference IS NOT NULL AND research_chunk_id IS NULL AND research_document_id IS NULL
                    DO NOTHING;
                    """,
                    (qid, ref),
                )
            report.question_sources_inserted += cur.rowcount

    def _load_question_embedding(self, cur, qid: int, emb: Tuple[List[float], str, str], report: LoadReport):
        vector, provider_name, model_name = emb
        cur.execute(
            """
            INSERT INTO question_embeddings (question_id, embedding_provider, embedding_model, embedding)
            VALUES (%s, %s, %s, %s::halfvec)
            ON CONFLICT (question_id, embedding_model) DO NOTHING;
            """,
            (qid, provider_name, model_name, vector_literal(vector)),
        )
        report.question_embeddings_inserted += cur.rowcount

    def _load_validation_attempts(
        self, cur, role: str, attempts: List[ValidationAttemptRecord],
        question_id_by_norm_text: Dict[str, int], report: LoadReport,
    ):
        for a in attempts:
            qid = question_id_by_norm_text.get(normalize_question_text(a.candidate_question_text)) if a.approved else None
            cur.execute(
                """
                INSERT INTO question_validation_attempts
                    (question_id, role, candidate_question_text, approved, quality_score, critical_failure,
                     rejection_reasons, validation_tier, validation_provider, validation_model, validated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s);
                """,
                (
                    qid, a.role, a.candidate_question_text, a.approved, a.quality_score, a.critical_failure,
                    json.dumps(a.rejection_reasons), a.validation_tier, a.validation_provider,
                    a.validation_model, a.validated_at,
                ),
            )
            report.validation_attempts_inserted += 1

    @staticmethod
    def _split_source_reference(ref: str) -> Tuple[Optional[str], Optional[str]]:
        m = _SOURCE_REF_RE.match(ref.strip())
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return ref.strip() or None, None

    def _load_vector_store_chunk_embeddings(self, role: Optional[str] = None, company: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Filters vector_store.json chunks by role (existing behavior) OR
        by company (new: company-only chunks have role=None, company=<name>,
        via synthesize_and_index_company_research())."""
        if not config.VECTOR_STORE_FILE.exists():
            return {}
        with open(config.VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
            store = json.load(f)
        result = {}
        for c in store.get("chunks", []):
            matches = (c.get("company") == company) if company else (c.get("role") == role)
            if matches and c.get("embedding"):
                result[c["chunk_id"]] = {
                    "embedding": c["embedding"],
                    "provider": store.get("embedding_provider"),
                    "model": store.get("embedding_model"),
                }
        return result

    def _load_research_and_knowledge(
        self, cur, role: str, report: LoadReport, company: Optional[str] = None, sync_legacy_state: bool = True,
    ):
        """Loads research_documents + research_sources + research_chunks +
        knowledge_state for one unit. `company=None` is the original
        role-level path (unchanged). `company=<name>` loads a standalone
        company research profile instead: reads the company-prefixed research
        file and the company-slugged synthesis file (synthesis_service names
        output files by research.role, which IS the company name for a
        company ResearchOutput -- no separate company synthesis code path was
        needed), and tags the resulting chunks with role=NULL, company=<name>
        so they never leak into role-only RAG retrieval by accident.

        sync_legacy_state gates the role-only state.json->knowledge_state
        sync below -- see load_role()'s docstring for why the production
        orchestrator must pass False here."""
        is_company = company is not None
        slug = slugify(company) if is_company else slugify(role)
        research_document_id = None
        chunk_id_to_pk: Dict[str, int] = {}

        research_file = (
            config.KNOWLEDGE_DIR / f"company_{slug}_research.json" if is_company
            else config.KNOWLEDGE_DIR / f"{slug}_research.json"
        )
        if research_file.exists():
            with open(research_file, "r", encoding="utf-8") as f:
                research = json.load(f)
            cur.execute(
                """
                INSERT INTO research_documents
                    (role, company, research_provider, knowledge_version, source_hash, raw_summary,
                     technologies, engineering_practices, responsibilities, interview_topics, trends, researched_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                RETURNING id;
                """,
                (
                    research.get("role", company if is_company else role), research.get("company"),
                    research.get("actual_provider", "unknown"), research.get("knowledge_version", "v1.0"),
                    research.get("source_hash"), research.get("raw_summary"),
                    json.dumps(research.get("technologies", [])),
                    json.dumps(research.get("engineering_practices", [])),
                    json.dumps(research.get("responsibilities", [])),
                    json.dumps(research.get("interview_topics", [])),
                    json.dumps(research.get("trends", [])),
                    research.get("researched_at"),
                ),
            )
            research_document_id = cur.fetchone()[0]
            report.research_documents_inserted += 1

            for ref in research.get("source_references", []):
                title, url = self._split_source_reference(ref)
                if not url:
                    continue
                # Idempotent upsert keyed on research_sources.url_normalized
                # (migration 0002) -- re-loading the same source (even with a
                # trailing-slash/scheme variation) never creates a duplicate
                # row; it resolves to the existing one instead.
                cur.execute(
                    """
                    INSERT INTO research_sources (url, title, discovery_provider)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (url_normalized) DO NOTHING
                    RETURNING id;
                    """,
                    (url, title, research.get("actual_provider")),
                )
                row = cur.fetchone()
                if row:
                    source_id = row[0]
                    report.research_sources_inserted += 1
                else:
                    cur.execute(
                        """
                        SELECT id FROM research_sources
                        WHERE url_normalized = lower(regexp_replace(regexp_replace(regexp_replace(trim(%s), '#.*$', ''), '/+$', ''), '^https?://', ''));
                        """,
                        (url,),
                    )
                    source_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO research_document_sources (research_document_id, research_source_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                    (research_document_id, source_id),
                )

        # synthesis_service.synthesize() names its output file by
        # research.role, which for a company ResearchOutput IS the company
        # name -- so this is the same filename shape as the role path, just
        # slugified from `company` instead of `role`.
        synth_file = config.KNOWLEDGE_DIR / f"{slug}_synthesized.json"
        if synth_file.exists():
            with open(synth_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            vs_embeddings = (
                self._load_vector_store_chunk_embeddings(company=company) if is_company
                else self._load_vector_store_chunk_embeddings(role=role)
            )
            # research_chunks.role is NOT NULL (migration 0001), so a
            # standalone company chunk reuses the role slot for the company
            # name -- same convention knowledge_state/research_documents
            # already use for company-only units. `company` being non-null
            # is the real discriminator; rag_service.retrieve_context()
            # explicitly excludes any chunk with company set, so this can
            # never leak into role-only retrieval regardless of what ends up
            # in the role column.
            chunk_role = company if is_company else role
            chunk_company = company if is_company else None
            for c in chunks:
                emb = vs_embeddings.get(c["chunk_id"])
                cur.execute(
                    """
                    INSERT INTO research_chunks
                        (chunk_id, research_document_id, role, company, topic, text,
                         synthesis_provider, embedding_provider, embedding_model, embedding, knowledge_version)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::halfvec,%s)
                    ON CONFLICT (chunk_id) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        c["chunk_id"], research_document_id, chunk_role, chunk_company, c.get("topic"), c.get("text"),
                        "gemini",
                        emb["provider"] if emb else None,
                        emb["model"] if emb else None,
                        vector_literal(emb["embedding"]) if emb else None,
                        c.get("knowledge_version", "v1.0"),
                    ),
                )
                row = cur.fetchone()
                if row:
                    chunk_id_to_pk[c["chunk_id"]] = row[0]
                    report.research_chunks_inserted += 1
                else:
                    cur.execute("SELECT id FROM research_chunks WHERE chunk_id = %s;", (c["chunk_id"],))
                    chunk_id_to_pk[c["chunk_id"]] = cur.fetchone()[0]

        # The role-only state.json sync below mirrors the legacy single-role
        # CLI path (state_manager.py) -- it is ONLY correct for the
        # standalone "run the legacy CLI, then load its output" workflow,
        # where state.json's status genuinely IS that role's final result.
        # It must be skipped (sync_legacy_state=False) when the caller is
        # the production orchestrator, which writes its OWN authoritative
        # status via freshness.record_change()/mark_status() -- BEFORE this
        # method runs -- and whose local state.json (touched only as a side
        # effect of the research_service call both paths share) is left at
        # "RUNNING" regardless of the orchestrator's real outcome. Company
        # research has no equivalent local file-based state at all -- its
        # knowledge_state is written directly by freshness.py -- so this is
        # always skipped for a company load regardless of the flag.
        if not is_company and sync_legacy_state and config.STATE_FILE.exists():
            with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                states = json.load(f)
            s = states.get(role)
            if s:
                cur.execute(
                    """
                    INSERT INTO knowledge_state
                        (role, company, current_knowledge_version, current_research_document_id,
                         source_hash, status, reason, last_researched_at, next_research_at)
                    VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (role, COALESCE(company, '')) DO UPDATE SET
                        current_knowledge_version = EXCLUDED.current_knowledge_version,
                        current_research_document_id = EXCLUDED.current_research_document_id,
                        source_hash = EXCLUDED.source_hash,
                        status = EXCLUDED.status,
                        reason = EXCLUDED.reason,
                        last_researched_at = EXCLUDED.last_researched_at,
                        next_research_at = EXCLUDED.next_research_at,
                        updated_at = now();
                    """,
                    (
                        role, s.get("knowledge_version"), research_document_id, s.get("source_hash"),
                        s.get("status", "PENDING"), s.get("reason"),
                        s.get("last_researched_at"), s.get("next_research_at"),
                    ),
                )
                report.knowledge_state_upserted = True

        return research_document_id, chunk_id_to_pk
