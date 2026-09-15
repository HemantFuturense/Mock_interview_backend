"""
Production rollout orchestrator for the 59-role / 24-company question
pipeline.

This module introduces NO new research, synthesis, generation, validation,
or deduplication logic. It coordinates the EXISTING components
(research_service, synthesis_service, rag_service, generator_service,
validator_service, db_loader, pipeline_runner.classify_scope) against
DB-authoritative state (knowledge_state, pipeline_jobs from migration 0002)
instead of the local JSON files those components were originally built
around for single-role/dev-mode use.

Flow per role (process_role):
    is_due(role)?  -- DB knowledge_state cadence check
        no  -> SKIPPED_NO_CHANGE, nothing touched
        yes -> research (reused verbatim, force=True since the DB check
               already decided "due")
               -> compare new source_hash to DB knowledge_state.source_hash
                  no change -> record_no_change(), question bank untouched
                  changed   -> bump knowledge_version, record history,
                               synthesize + index RAG (reused verbatim),
                               generate a batch (full pilot size on a role's
                               first run, a small top-up on a refresh),
                               validate + dedup against the DB's CURRENT
                               approved bank for this role (not a local
                               file), classify scope (reused), load only
                               the newly-approved questions via db_loader
                               (idempotent by construction).

Every state transition is written to `pipeline_jobs` before/after the work
it describes, so a crash mid-role leaves an inspectable, resumable status
rather than silence. A WAITING_FOR_QUOTA/BLOCKED/FAILED role never touches
another role's job row or question bank.
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import psycopg2

from .config import config
from .models import QuestionObject, ValidationAttemptRecord, ResearchOutput, KnowledgeChunk
from .research_service import research_service
from .synthesis_service import synthesis_service
from .rag_service import rag_service
from .generator_service import generator_service
from .validator_service import validator_service
from .db_loader import QuestionBankLoader, slugify
from .freshness import KnowledgeFreshnessTracker, bump_version
from .governor import QuotaExhaustedException, ProviderBlockedException


VALID_JOB_STATUSES = {
    "PENDING", "RESEARCHING", "RESEARCHED", "SYNTHESIZING", "KNOWLEDGE_UPDATED",
    "GENERATING", "VALIDATING", "LOADING", "COMPLETED", "SKIPPED_NO_CHANGE",
    "WAITING_FOR_QUOTA", "BLOCKED", "FAILED",
}


@dataclass
class UnitResult:
    role: str
    company: Optional[str]
    job_type: str
    status: str
    reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineJobStore:
    """Thin read/write wrapper around the `pipeline_jobs` table
    (migration 0002). One current-state row per (role, company, job_type);
    upserting is always safe to repeat."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config

    def _connect(self):
        return psycopg2.connect(**self.db_config)

    def upsert(
        self, role: str, company: Optional[str], job_type: str, status: str,
        reason: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None,
        mark_started: bool = False, mark_completed: bool = False,
    ) -> None:
        assert status in VALID_JOB_STATUSES, f"Unknown job status: {status}"
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pipeline_jobs
                    (role, company, job_type, status, reason, metrics, started_at, completed_at, attempt_count)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb,
                        CASE WHEN %s THEN now() ELSE NULL END,
                        CASE WHEN %s THEN now() ELSE NULL END,
                        1)
                ON CONFLICT (role, COALESCE(company, ''), job_type) DO UPDATE SET
                    status = EXCLUDED.status,
                    reason = EXCLUDED.reason,
                    metrics = CASE WHEN %s::jsonb <> '{}'::jsonb THEN %s::jsonb ELSE pipeline_jobs.metrics END,
                    started_at = COALESCE(pipeline_jobs.started_at, EXCLUDED.started_at),
                    completed_at = CASE WHEN %s THEN now() ELSE pipeline_jobs.completed_at END,
                    attempt_count = pipeline_jobs.attempt_count + (CASE WHEN %s THEN 1 ELSE 0 END),
                    updated_at = now();
                """,
                (
                    role, company, job_type, status, reason, json.dumps(metrics or {}),
                    mark_started, mark_completed,
                    json.dumps(metrics or {}), json.dumps(metrics or {}),
                    mark_completed, mark_started,
                ),
            )
        finally:
            conn.close()

    def get(self, role: str, company: Optional[str], job_type: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "SELECT status, reason, metrics, attempt_count FROM pipeline_jobs "
                "WHERE role=%s AND company IS NOT DISTINCT FROM %s AND job_type=%s;",
                (role, company, job_type),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"status": row[0], "reason": row[1], "metrics": row[2], "attempt_count": row[3]}
        finally:
            conn.close()

    def list_all(self, roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            if roles:
                cur.execute(
                    "SELECT role, company, job_type, status, reason, attempt_count, updated_at "
                    "FROM pipeline_jobs WHERE role = ANY(%s) ORDER BY role, job_type;",
                    (roles,),
                )
            else:
                cur.execute(
                    "SELECT role, company, job_type, status, reason, attempt_count, updated_at "
                    "FROM pipeline_jobs ORDER BY role, job_type;"
                )
            return [
                {"role": r[0], "company": r[1], "job_type": r[2], "status": r[3],
                 "reason": r[4], "attempt_count": r[5], "updated_at": r[6]}
                for r in cur.fetchall()
            ]
        finally:
            conn.close()


async def synthesize_and_index_company_research(company: str, research_output) -> List:
    """Synthesizes a company ResearchOutput into KnowledgeChunks (reuses
    synthesis_service.synthesize() verbatim -- it doesn't care whether the
    ResearchOutput came from a role or a company) and indexes them into
    rag_service so retrieve_company_context() can find them. The only new
    logic here is remapping each chunk's role/company fields: synthesis
    always sets chunk.role = research.role, which for a company
    ResearchOutput IS the company name -- that's exactly the tagging
    convention company-only chunks are meant to use (see db_loader.py's
    docstring), so no remapping is actually required, just documented."""
    chunks = await synthesis_service.synthesize(research_output)
    for c in chunks:
        c.company = company  # research.role already equals company; this just makes the company field explicit too
    await rag_service.index_chunks(chunks)
    return chunks


def fetch_existing_bank_from_db(db_config: Dict[str, Any], role: str) -> List[QuestionObject]:
    """Reconstructs a dedup-usable QuestionObject list from the DB's CURRENT
    approved bank for `role`. This is what makes semantic dedup during a
    refresh DB-authoritative instead of dependent on the local file/process
    memory -- deduplicator_service.check_duplicate() only ever reads
    `.question` off each item, so the other required fields are filled with
    the row's own real values where available (a faithful reconstruction,
    not placeholder data)."""
    conn = psycopg2.connect(**db_config)
    try:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT question_text, experience_band, difficulty, paradigm, scope FROM questions WHERE role = %s;",
            (role,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    bank = []
    for text, band, diff, paradigm, scope in rows:
        bank.append(QuestionObject(
            question=text, role=role, experience_band=band, difficulty=diff,
            technical_depth=diff, problem_complexity=diff, architecture_complexity=diff,
            troubleshooting=diff, business_complexity=diff, decision_making=diff, leadership_ownership=diff,
            paradigm=paradigm, scope=scope,
        ))
    return bank


class RolloutOrchestrator:
    """Coordinates role research refresh, change detection, incremental
    generation, and DB loading across the production role set."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.jobs = PipelineJobStore(db_config)
        self.freshness = KnowledgeFreshnessTracker(db_config)
        self.loader = QuestionBankLoader(db_config)

    def get_pending_roles(self, roles: List[str]) -> List[str]:
        """Roles whose research is due per DB knowledge_state cadence, OR
        that have never completed a ROLE_GENERATION job."""
        pending = []
        for role in roles:
            job = self.jobs.get(role, None, "ROLE_GENERATION")
            already_done_and_fresh = (
                job and job["status"] in ("COMPLETED", "SKIPPED_NO_CHANGE")
                and not self.freshness.is_due(role)
            )
            if not already_done_and_fresh:
                pending.append(role)
        return pending

    def get_pending_companies(self, companies: List[str]) -> List[str]:
        """Companies whose research is due per DB knowledge_state cadence
        (14-day company cadence), OR that have never completed a
        COMPANY_RESEARCH job. Mirrors get_pending_roles() exactly so a
        scheduler tick never wastes a batch slot re-checking an already-fresh
        company -- this is the "find company research that is due / skip
        fresh items" step for the company axis."""
        pending = []
        for company in companies:
            job = self.jobs.get(company, company, "COMPANY_RESEARCH")
            already_done_and_fresh = (
                job and job["status"] in ("COMPLETED", "SKIPPED_NO_CHANGE")
                and not self.freshness.is_due(company, company=company)
            )
            if not already_done_and_fresh:
                pending.append(company)
        return pending

    def status(self, roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.jobs.list_all(roles)

    async def process_role(self, role: str, dry_run: bool = False) -> UnitResult:
        """Dispatches on TWO independent axes -- research freshness
        (knowledge_state, ~30-day cadence) and generation completeness
        (pipeline_jobs.ROLE_GENERATION status) -- rather than treating
        research freshness as a blanket gate over the whole role. This is
        the fix for the bug where a role whose research succeeded but whose
        generation crashed/paused mid-way (job status left at e.g.
        VALIDATING/WAITING_FOR_QUOTA/BLOCKED, never COMPLETED) became stuck:
        the old code's `if not due: skip everything` meant an already-fresh
        research timestamp permanently prevented ever finishing that
        role's generation until the next ~30-day cadence window, even
        though get_pending_roles() correctly kept including it as pending.

        Cases:
          A. research fresh AND generation complete   -> SKIP entirely.
          B. research fresh BUT generation incomplete -> RESUME generation
             only, WITHOUT re-researching (research freshness must never
             gate an incomplete generation job).
          C. research due AND generation complete     -> refresh research,
             then run an incremental top-up generation pass.
          D. research due AND generation incomplete   -> refresh research
             (it's genuinely stale and due regardless), then generation
             runs via the same _generate_and_load() path, which determines
             its own target size from the DB's actual current bank size
             (not from research history), so an incomplete-then-refreshed
             role still gets a full-size pass if it has no approved
             questions yet, and a top-up otherwise.
        """
        research_due = self.freshness.is_due(role)
        gen_job = self.jobs.get(role, None, "ROLE_GENERATION")
        generation_complete = bool(gen_job and gen_job["status"] in ("COMPLETED", "SKIPPED_NO_CHANGE"))

        if dry_run:
            if not research_due and generation_complete:
                reason = "within cadence window and generation complete -- would be skipped"
            elif not research_due and not generation_complete:
                reason = (f"research fresh but generation incomplete "
                          f"(last status={gen_job['status'] if gen_job else 'NONE'}) -- would resume generation only")
            else:
                reason = "research due -- would run"
            print(f"[DRY-RUN] {role}: research_due={research_due}, generation_complete={generation_complete}, "
                  f"last_job_status={gen_job['status'] if gen_job else 'NONE'} -> {reason}")
            return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="DRY_RUN", reason=reason)

        # ---- Case A: nothing to do ----
        if not research_due and generation_complete:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "SKIPPED_NO_CHANGE",
                              reason="Within research cadence window", mark_started=True, mark_completed=True)
            print(f"[ORCHESTRATOR] '{role}' is within its research cadence window and generation is already complete -- skipping.")
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="SKIPPED_NO_CHANGE")

        # ---- Case B: resume generation only, no re-research ----
        if not research_due and not generation_complete:
            print(f"[ORCHESTRATOR] '{role}': research is fresh but generation is incomplete "
                  f"(last status={gen_job['status'] if gen_job else 'NONE'}) -- resuming generation only.")
            return await self._generate_and_load(role)

        # ---- Cases C & D: research is due -- refresh it ----
        self.jobs.upsert(role, None, "ROLE_RESEARCH", "RESEARCHING", mark_started=True)
        try:
            research_result = await research_service.execute_research(role, force=True)
        except QuotaExhaustedException as qe:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "WAITING_FOR_QUOTA", reason=str(qe))
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="WAITING_FOR_QUOTA", reason=str(qe))
        except ProviderBlockedException as pbe:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "BLOCKED", reason=str(pbe))
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="BLOCKED", reason=str(pbe))
        except Exception as e:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "FAILED", reason=str(e))
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="FAILED", reason=str(e))

        if research_result.get("status") != "SUCCESS":
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "FAILED",
                              reason=f"Unexpected research status: {research_result.get('status')}")
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="FAILED")

        research_output = research_result["research_output"]
        self.jobs.upsert(role, None, "ROLE_RESEARCH", "RESEARCHED")

        # ---- Change detection: DB knowledge_state is authoritative ----
        existing_state = self.freshness.get_state(role)
        is_first_research = existing_state is None or not existing_state.get("last_researched_at")
        unchanged = (not is_first_research) and existing_state["source_hash"] == research_output.source_hash

        if unchanged:
            self.freshness.record_no_change(role, None, research_output.source_hash)
            # Research being unchanged does NOT by itself mean nothing needs
            # to happen: a prior generation attempt may still be incomplete
            # (case D nuance). Only skip generation too if it was already
            # complete; otherwise fall through to resume it.
            if generation_complete:
                self.jobs.upsert(role, None, "ROLE_GENERATION", "SKIPPED_NO_CHANGE",
                                  reason="Research content unchanged since last knowledge version",
                                  mark_started=True, mark_completed=True)
                print(f"[ORCHESTRATOR] '{role}': no meaningful knowledge change -- question bank left untouched.")
                return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="SKIPPED_NO_CHANGE")
            print(f"[ORCHESTRATOR] '{role}': research content unchanged, but generation was incomplete "
                  f"(last status={gen_job['status'] if gen_job else 'NONE'}) -- resuming generation.")
            return await self._generate_and_load(role)

        self.jobs.upsert(role, None, "ROLE_RESEARCH", "SYNTHESIZING")
        try:
            chunks = await synthesis_service.synthesize(research_output)
            await rag_service.index_chunks(chunks)
        except QuotaExhaustedException as e:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "WAITING_FOR_QUOTA", reason=str(e))
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="WAITING_FOR_QUOTA", reason=str(e))
        except ProviderBlockedException as e:
            self.jobs.upsert(role, None, "ROLE_RESEARCH", "BLOCKED", reason=str(e))
            return UnitResult(role=role, company=None, job_type="ROLE_RESEARCH", status="BLOCKED", reason=str(e))

        new_version = bump_version(existing_state["knowledge_version"] if existing_state else None)
        self.freshness.record_change(role, None, new_version, research_output.source_hash, None)
        self.jobs.upsert(role, None, "ROLE_RESEARCH", "KNOWLEDGE_UPDATED", mark_completed=True)

        return await self._generate_and_load(role)

    async def _generate_and_load(self, role: str) -> UnitResult:
        """The generation -> validation -> dedup -> load phase, factored out
        so it can run either right after a fresh research cycle OR as a
        direct RESUME of an incomplete generation job while research stays
        fresh (process_role()'s case B) -- generation completeness is
        tracked and resumed independently of research freshness.

        target_count (full pilot size vs. a small top-up) is derived from
        the DB's ACTUAL current approved-question count for this role, not
        from research history -- this is what makes a role recovering from
        a crash with ZERO existing questions correctly get a full-size
        pass, even though its research has been done before (see the
        Phase 5B "Conversational AI Engineer" recovery).

        QuotaExhaustedException/ProviderBlockedException are caught around
        EVERY provider-calling step in this method (generation, validation,
        and the final DB load, which computes real question-bank
        embeddings before its own transaction) -- this is the fix for the
        crash where validate_question()'s internal RAG-context embedding
        call could raise uncaught and kill the whole process. Every catch
        checkpoints WAITING_FOR_QUOTA/BLOCKED with whatever was already
        approved this call persisted via _load_partial() first, so no
        already-completed work is lost and nothing is marked COMPLETED
        prematurely.
        """
        from .pipeline_runner import pipeline_runner  # reuse classify_scope() verbatim, no reimplementation

        existing_bank = fetch_existing_bank_from_db(self.db_config, role)
        is_first_time = len(existing_bank) == 0
        target_count = config.PILOT_TARGET_QUESTIONS_PER_ROLE if is_first_time else config.REFRESH_TOP_UP_QUESTIONS_PER_ROLE
        self.jobs.upsert(role, None, "ROLE_GENERATION", "GENERATING", mark_started=True)

        bands = config.EXPERIENCE_BANDS
        per_band = max(target_count // len(bands), 1)
        approved: List[QuestionObject] = []
        attempts: List[ValidationAttemptRecord] = []

        for batch_idx, band in enumerate(bands):
            target_scope = "DOMAIN" if batch_idx == 1 else ("COMPANY" if batch_idx == 4 else "UNIVERSAL")
            try:
                candidates = await generator_service.generate_batch_for_band(role, band, per_band, target_scope)
            except QuotaExhaustedException as e:
                self.jobs.upsert(role, None, "ROLE_GENERATION", "WAITING_FOR_QUOTA", reason=str(e),
                                  metrics={"approved_so_far": len(approved)})
                await self._load_partial(role, approved, attempts)
                return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="WAITING_FOR_QUOTA", reason=str(e))
            except ProviderBlockedException as e:
                self.jobs.upsert(role, None, "ROLE_GENERATION", "BLOCKED", reason=str(e),
                                  metrics={"approved_so_far": len(approved)})
                await self._load_partial(role, approved, attempts)
                return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="BLOCKED", reason=str(e))
            except Exception:
                # This band's own bounded single-retry (generator_service) is
                # already exhausted by the time this is reached -- treated as
                # a partial batch, not a reason to abandon the whole role.
                continue

            self.jobs.upsert(role, None, "ROLE_GENERATION", "VALIDATING")
            for cand in candidates:
                cand.scope = pipeline_runner.classify_scope(cand, target_scope)
                try:
                    gate_result = await validator_service.validate_question(cand, existing_bank + approved, expected_role=role)
                except QuotaExhaustedException as e:
                    self.jobs.upsert(role, None, "ROLE_GENERATION", "WAITING_FOR_QUOTA", reason=str(e),
                                      metrics={"approved_so_far": len(approved)})
                    await self._load_partial(role, approved, attempts)
                    return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="WAITING_FOR_QUOTA", reason=str(e))
                except ProviderBlockedException as e:
                    self.jobs.upsert(role, None, "ROLE_GENERATION", "BLOCKED", reason=str(e),
                                      metrics={"approved_so_far": len(approved)})
                    await self._load_partial(role, approved, attempts)
                    return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="BLOCKED", reason=str(e))
                attempts.append(gate_result.to_attempt_record(role=cand.role, candidate_question_text=cand.question))
                if gate_result.approved:
                    approved.append(cand)

        self.jobs.upsert(role, None, "ROLE_GENERATION", "LOADING")
        # sync_legacy_state=False: this orchestrator has ALREADY written the
        # authoritative DB status for this role via freshness.record_change()
        # above -- the loader must not clobber it back to "RUNNING" from the
        # local state.json that research_service.execute_research() (shared
        # by both the legacy CLI and this orchestrator) always leaves behind.
        try:
            load_report = await self.loader.load_role(role, approved, attempts, sync_legacy_state=False)
        except QuotaExhaustedException as e:
            # load_role() computes question-bank embeddings BEFORE opening
            # its DB transaction -- if that call itself exhausts quota nothing
            # from this attempt was persisted (no partial DB write occurred),
            # so there is nothing further to load; just checkpoint and stop.
            self.jobs.upsert(role, None, "ROLE_GENERATION", "WAITING_FOR_QUOTA", reason=str(e),
                              metrics={"approved_so_far": len(approved)})
            return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="WAITING_FOR_QUOTA", reason=str(e))
        except ProviderBlockedException as e:
            self.jobs.upsert(role, None, "ROLE_GENERATION", "BLOCKED", reason=str(e),
                              metrics={"approved_so_far": len(approved)})
            return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="BLOCKED", reason=str(e))

        metrics = {
            "generated": len(attempts), "approved": len(approved),
            "rejected": len(attempts) - len(approved), "inserted": load_report.questions_inserted,
            "is_first_time": is_first_time,
        }
        self.jobs.upsert(role, None, "ROLE_GENERATION", "COMPLETED", metrics=metrics, mark_completed=True)
        print(f"[ORCHESTRATOR] '{role}': {len(approved)}/{len(attempts)} approved, "
              f"{load_report.questions_inserted} newly inserted into DB.")
        return UnitResult(role=role, company=None, job_type="ROLE_GENERATION", status="COMPLETED", metrics=metrics)

    async def _load_partial(self, role: str, approved: List[QuestionObject], attempts: List[ValidationAttemptRecord]) -> None:
        if approved or attempts:
            await self.loader.load_role(role, approved, attempts, sync_legacy_state=False)

    async def process_company_research(self, company: str, dry_run: bool = False) -> UnitResult:
        """COMPANY_RESEARCH job: keeps company-level knowledge fresh on its
        own (shorter) cadence. Does not itself trigger question generation --
        company context already reaches the question bank through the
        existing scope-classification mechanism inside process_role().

        Dispatches on the SAME two independent axes process_role() does --
        research freshness (knowledge_state) and downstream-work
        completeness (this job's own status) -- rather than treating
        freshness as a blanket gate. This is the company-level fix for the
        bug where a company whose research succeeded but whose synthesis/
        RAG-indexing/DB-load never finished (job left at e.g.
        WAITING_FOR_QUOTA/BLOCKED, never COMPLETED) became stuck: research
        succeeding sets last_researched_at, so a bare `if not due: skip`
        would silently overwrite that WAITING_FOR_QUOTA/BLOCKED status with
        a misleading SKIPPED_NO_CHANGE and never actually finish the
        pipeline -- the company axis had no equivalent of the
        ROLE_RESEARCH/ROLE_GENERATION split that already fixed this for
        roles, since research->synthesis->index->load are all tracked under
        one COMPANY_RESEARCH job type. A COMPANY_RESEARCH job is only
        "complete" once the ENTIRE research->synthesis->RAG-indexing->load
        pipeline has finished -- not merely because the research step alone
        succeeded.

        Cases (mirroring process_role()'s A/B/C/D exactly):
          A. research fresh AND downstream complete   -> SKIP.
          B. research fresh BUT downstream incomplete  -> RESUME downstream
             only (synthesis/index/load), WITHOUT re-researching.
          C. research due AND downstream complete      -> refresh research,
             then run the downstream pipeline.
          D. research due AND downstream incomplete     -> refresh research
             (it's genuinely due regardless), then run the downstream
             pipeline -- which safely reuses an already-synthesized local
             file only when the research content is confirmed unchanged
             (see _resume_company_downstream's allow_reuse_synth), so a
             genuine content change never resumes from stale synthesis.
        """
        # knowledge_state is shaped (role, company); a standalone company
        # profile (not tied to one specific role) is stored with both set
        # to the company name, since the table has no role-less concept.
        due = self.freshness.is_due(role=company, company=company)
        job = self.jobs.get(company, company, "COMPANY_RESEARCH")
        downstream_complete = bool(job and job["status"] in ("COMPLETED", "SKIPPED_NO_CHANGE"))

        if dry_run:
            if not due and downstream_complete:
                reason = "within cadence window and downstream work complete -- would be skipped"
            elif not due and not downstream_complete:
                reason = (f"research fresh but downstream work incomplete "
                          f"(last status={job['status'] if job else 'NONE'}) -- would resume downstream only")
            else:
                reason = "research due -- would run"
            print(f"[DRY-RUN] company={company}: research_due={due}, downstream_complete={downstream_complete}, "
                  f"last_job_status={job['status'] if job else 'NONE'} -> {reason}")
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="DRY_RUN", reason=reason)

        # ---- Case A: nothing to do ----
        if not due and downstream_complete:
            self.jobs.upsert(company, company, "COMPANY_RESEARCH", "SKIPPED_NO_CHANGE",
                              reason="Within company research cadence window", mark_started=True, mark_completed=True)
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="SKIPPED_NO_CHANGE")

        # ---- Case B: resume downstream work only, no re-research ----
        if not due and not downstream_complete:
            print(f"[ORCHESTRATOR] Company research for '{company}': research is fresh but downstream "
                  f"work is incomplete (last status={job['status'] if job else 'NONE'}) -- resuming, no re-research.")
            return await self._resume_company_downstream(company)

        # ---- Cases C & D: research is due -- refresh it ----
        self.jobs.upsert(company, company, "COMPANY_RESEARCH", "RESEARCHING", mark_started=True)
        try:
            result = await research_service.execute_company_research(company)
        except QuotaExhaustedException as e:
            self.jobs.upsert(company, company, "COMPANY_RESEARCH", "WAITING_FOR_QUOTA", reason=str(e))
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="WAITING_FOR_QUOTA", reason=str(e))
        except ProviderBlockedException as e:
            self.jobs.upsert(company, company, "COMPANY_RESEARCH", "BLOCKED", reason=str(e))
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="BLOCKED", reason=str(e))
        except Exception as e:
            self.jobs.upsert(company, company, "COMPANY_RESEARCH", "FAILED", reason=str(e))
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="FAILED", reason=str(e))

        output = result["research_output"]
        existing_state = self.freshness.get_state(company, company=company)
        if existing_state and existing_state["source_hash"] == output.source_hash:
            self.freshness.record_no_change(company, company, output.source_hash)
            # Research being unchanged does NOT by itself mean downstream
            # work is complete (case D nuance, mirroring the role fix) --
            # only skip if it genuinely already finished.
            if downstream_complete:
                self.jobs.upsert(company, company, "COMPANY_RESEARCH", "SKIPPED_NO_CHANGE",
                                  reason="Company research content unchanged", mark_completed=True)
                return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="SKIPPED_NO_CHANGE")
            print(f"[ORCHESTRATOR] Company research for '{company}': content unchanged, but downstream "
                  f"work was incomplete (last status={job['status'] if job else 'NONE'}) -- resuming.")
            # Content confirmed unchanged -> safe to reuse an existing
            # synthesized file if one exists, no wasted re-synthesis call.
            return await self._resume_company_downstream(company, research_output=output, allow_reuse_synth=True)

        new_version = bump_version(existing_state["knowledge_version"] if existing_state else None)
        self.freshness.record_change(company, company, new_version, output.source_hash, None)
        self.jobs.upsert(company, company, "COMPANY_RESEARCH", "KNOWLEDGE_UPDATED")

        # Content genuinely changed -> any existing synthesized file belongs
        # to the PREVIOUS knowledge version and must never be silently
        # reused; force fresh synthesis from the new research output.
        return await self._resume_company_downstream(company, research_output=output, allow_reuse_synth=False)

    async def _resume_company_downstream(
        self, company: str, research_output: Optional[ResearchOutput] = None, allow_reuse_synth: bool = True,
    ) -> UnitResult:
        """Runs (or resumes) the synthesis -> RAG-indexing -> DB-load steps
        for a company whose research is already valid, WITHOUT making any
        new research API call. `research_output`, if not given, is loaded
        from the existing local `company_{slug}_research.json` artifact
        (Case B: research is fresh, nothing new was just fetched).

        allow_reuse_synth=True permits reusing an already-synthesized local
        `{slug}_synthesized.json` file (if present) instead of re-running
        synthesis -- safe ONLY when the research content is confirmed
        unchanged from what produced that file (Case B, or an unchanged-
        content refresh). Callers that just recorded a genuine content
        CHANGE must pass allow_reuse_synth=False so a stale prior-version
        synthesis is never silently reused."""
        slug = slugify(company)

        if research_output is None:
            research_file = config.KNOWLEDGE_DIR / f"company_{slug}_research.json"
            if not research_file.exists():
                reason = (f"No local research artifact found for '{company}' to resume downstream work from -- "
                          f"cannot safely resume without a new research call.")
                self.jobs.upsert(company, company, "COMPANY_RESEARCH", "FAILED", reason=reason)
                return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="FAILED", reason=reason)
            with open(research_file, "r", encoding="utf-8") as f:
                research_output = ResearchOutput(**json.load(f))

        self.jobs.upsert(company, company, "COMPANY_RESEARCH", "SYNTHESIZING")
        synth_file = config.KNOWLEDGE_DIR / f"{slug}_synthesized.json"
        try:
            if allow_reuse_synth and synth_file.exists():
                with open(synth_file, "r", encoding="utf-8") as f:
                    chunks = [KnowledgeChunk(**c) for c in json.load(f)]
                for c in chunks:
                    c.company = company
                await rag_service.index_chunks(chunks)
            else:
                await synthesize_and_index_company_research(company, research_output)
        except (QuotaExhaustedException, ProviderBlockedException) as e:
            status = "WAITING_FOR_QUOTA" if isinstance(e, QuotaExhaustedException) else "BLOCKED"
            self.jobs.upsert(company, company, "COMPANY_RESEARCH", status, reason=str(e))
            return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status=status, reason=str(e))

        load_report = self.loader.load_company_research(company)
        self.jobs.upsert(
            company, company, "COMPANY_RESEARCH", "COMPLETED", mark_completed=True,
            metrics={"chunks_inserted": load_report.research_chunks_inserted},
        )
        print(f"[ORCHESTRATOR] Company research for '{company}': downstream work complete, "
              f"{load_report.research_chunks_inserted} chunk(s) now available to RAG.")
        return UnitResult(role=company, company=company, job_type="COMPANY_RESEARCH", status="COMPLETED",
                           metrics={"chunks_inserted": load_report.research_chunks_inserted})

    async def run_batch(self, roles: List[str], batch_size: int, dry_run: bool = False) -> List[UnitResult]:
        pending = self.get_pending_roles(roles)
        todo = pending[:batch_size]
        print(f"[ORCHESTRATOR] {len(pending)} role(s) pending out of {len(roles)} requested; processing batch of {len(todo)}"
              f"{' (dry-run)' if dry_run else ''}.")
        results = []
        for role in todo:
            result = await self.process_role(role, dry_run=dry_run)
            results.append(result)
            if result.status in ("WAITING_FOR_QUOTA", "BLOCKED") and not dry_run:
                print(f"[ORCHESTRATOR] Stopping batch: '{role}' is {result.status}. "
                      f"Remaining {len(todo) - len(results)} role(s) in this batch, and all other pending roles, "
                      f"are untouched and safe to resume later.")
                break
        return results

    async def run_company_batch(self, companies: List[str], batch_size: int, dry_run: bool = False) -> List[UnitResult]:
        """Company-axis counterpart of run_batch(): pre-filters to companies
        that are actually pending (get_pending_companies) so a batch slot is
        never spent re-checking an already-fresh company, then stops the
        batch -- without erroring -- the moment any company reports
        WAITING_FOR_QUOTA/BLOCKED, leaving every other pending company
        untouched and safe to resume later."""
        pending = self.get_pending_companies(companies)
        todo = pending[:batch_size]
        print(f"[ORCHESTRATOR] {len(pending)} compan{'y' if len(pending) == 1 else 'ies'} pending out of "
              f"{len(companies)} requested; processing batch of {len(todo)}{' (dry-run)' if dry_run else ''}.")
        results = []
        for company in todo:
            result = await self.process_company_research(company, dry_run=dry_run)
            results.append(result)
            if result.status in ("WAITING_FOR_QUOTA", "BLOCKED") and not dry_run:
                remaining = len(todo) - len(results)
                print(f"[ORCHESTRATOR] Stopping company batch: '{company}' is {result.status}. "
                      f"Remaining {remaining} compan{'y' if remaining == 1 else 'ies'} in this batch, and all "
                      f"other pending companies, are untouched and safe to resume later.")
                break
        return results
