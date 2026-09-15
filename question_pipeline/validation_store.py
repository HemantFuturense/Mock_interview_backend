"""
Persistence for question_validation_attempts (Phase 2 quality gate).

Two implementations:
  - InMemoryValidationAttemptStore: used by tests and by the pipeline until
    migration 0001 has actually been applied to the live database. Enforces
    the SAME fail-closed invariant as the DB's chk_qva_* CHECK constraints,
    in code, as defense-in-depth -- so the invariant holds even before the
    schema exists.
  - PostgresValidationAttemptStore: targets the real `question_validation_attempts`
    table column-for-column. This is NOT exercised anywhere yet (no test uses
    it, pipeline_runner does not default to it) because migration 0001 has not
    been executed against the live database. It exists so that once the
    migration IS applied, wiring it in is a one-line change
    (QuestionPipelineRunner.validation_store = PostgresValidationAttemptStore()).
"""
import json
import threading
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from .models import ValidationAttemptRecord


class FailClosedViolation(ValueError):
    """Raised when a caller attempts to record a row that would violate the
    fail-closed invariant (mirrors the DB's chk_qva_fail_closed /
    chk_qva_approved_has_question CHECK constraints)."""
    pass


def _assert_fail_closed_invariants(attempt: ValidationAttemptRecord) -> None:
    if attempt.critical_failure and attempt.approved:
        raise FailClosedViolation(
            f"Refusing to record attempt for role='{attempt.role}': "
            f"critical_failure=True and approved=True cannot both hold "
            f"(fail-closed violation)."
        )
    if attempt.approved and attempt.question_id is None:
        raise FailClosedViolation(
            f"Refusing to record attempt for role='{attempt.role}': "
            f"approved=True requires a question_id (an approved attempt must "
            f"reference a real questions row)."
        )


class ValidationAttemptStore(ABC):
    """Storage abstraction for question_validation_attempts rows."""

    @abstractmethod
    def record(self, attempt: ValidationAttemptRecord) -> ValidationAttemptRecord:
        """Persist one validation attempt (approved or rejected). Returns the
        record with `id` populated."""
        raise NotImplementedError

    @abstractmethod
    def list_for_role(self, role: str) -> List[ValidationAttemptRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_for_question(self, question_id: int) -> List[ValidationAttemptRecord]:
        raise NotImplementedError


class InMemoryValidationAttemptStore(ValidationAttemptStore):
    """
    Process-local store. Each `record()` call appends immediately (no
    batching), which is what makes per-candidate checkpointing meaningful:
    if a later candidate in the same batch raises QuotaExhaustedException,
    every attempt recorded before that point is already durable in
    `self._records` and, if `persist_path` is set, already flushed to disk.

    Honest limitation: unlike PostgresValidationAttemptStore, this does NOT
    survive a process crash unless `persist_path` is provided (in which case
    it writes a JSON file after every record() call, mirroring the existing
    question_bank.json / state.json incremental-write pattern used elsewhere
    in this pipeline).
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._records: List[ValidationAttemptRecord] = []
        self._lock = threading.RLock()
        self._persist_path = persist_path
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._records = [ValidationAttemptRecord(**item) for item in data]
        except Exception as e:
            print(f"[VALIDATION_STORE] Warning: could not load {self._persist_path}: {e}")
            self._records = []

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in self._records], f, indent=2)
        except Exception as e:
            print(f"[VALIDATION_STORE] Warning: failed to persist attempts: {e}")

    def record(self, attempt: ValidationAttemptRecord) -> ValidationAttemptRecord:
        _assert_fail_closed_invariants(attempt)
        with self._lock:
            next_id = len(self._records) + 1
            stored = attempt.model_copy(update={"id": next_id})
            self._records.append(stored)
            self._save()
            return stored

    def list_for_role(self, role: str) -> List[ValidationAttemptRecord]:
        return [r for r in self._records if r.role == role]

    def list_for_question(self, question_id: int) -> List[ValidationAttemptRecord]:
        return [r for r in self._records if r.question_id == question_id]

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self._save()


class PostgresValidationAttemptStore(ValidationAttemptStore):
    """
    Real implementation targeting the `question_validation_attempts` table
    created by migrations/0001_question_bank_and_knowledge_layer.sql.

    NOT used by any test and NOT the default store on QuestionPipelineRunner
    today -- migration 0001 has not been executed against the live database,
    so this table does not exist yet. This class is provided so that once the
    migration is applied for real, switching the pipeline over is a one-line
    change rather than new code written under time pressure.
    """

    def __init__(self, db_config: dict):
        self._db_config = db_config

    def _connect(self):
        import psycopg2
        return psycopg2.connect(**self._db_config)

    def record(self, attempt: ValidationAttemptRecord) -> ValidationAttemptRecord:
        # Deliberately NOT pre-validated in Python beyond what the DB itself
        # will enforce -- chk_qva_fail_closed and chk_qva_approved_has_question
        # are the authoritative guard here once this store is live. A raised
        # psycopg2.errors.CheckViolation on an invalid row is the correct,
        # intended failure mode, not a bug to be caught and hidden.
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO question_validation_attempts
                        (question_id, role, candidate_question_text, approved,
                         quality_score, critical_failure, rejection_reasons,
                         validation_tier, validation_provider, validation_model,
                         validated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        attempt.question_id,
                        attempt.role,
                        attempt.candidate_question_text,
                        attempt.approved,
                        attempt.quality_score,
                        attempt.critical_failure,
                        json.dumps(attempt.rejection_reasons),
                        attempt.validation_tier,
                        attempt.validation_provider,
                        attempt.validation_model,
                        attempt.validated_at,
                    ),
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return attempt.model_copy(update={"id": new_id})
        finally:
            conn.close()

    def list_for_role(self, role: str) -> List[ValidationAttemptRecord]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, question_id, role, candidate_question_text, approved,
                           quality_score, critical_failure, rejection_reasons,
                           validation_tier, validation_provider, validation_model, validated_at
                    FROM question_validation_attempts WHERE role = %s ORDER BY validated_at;
                    """,
                    (role,),
                )
                return [self._row_to_record(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def list_for_question(self, question_id: int) -> List[ValidationAttemptRecord]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, question_id, role, candidate_question_text, approved,
                           quality_score, critical_failure, rejection_reasons,
                           validation_tier, validation_provider, validation_model, validated_at
                    FROM question_validation_attempts WHERE question_id = %s ORDER BY validated_at;
                    """,
                    (question_id,),
                )
                return [self._row_to_record(row) for row in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row) -> ValidationAttemptRecord:
        (id_, question_id, role, candidate_question_text, approved, quality_score,
         critical_failure, rejection_reasons, validation_tier, validation_provider,
         validation_model, validated_at) = row
        return ValidationAttemptRecord(
            id=id_,
            question_id=question_id,
            role=role,
            candidate_question_text=candidate_question_text,
            approved=approved,
            quality_score=float(quality_score),
            critical_failure=critical_failure,
            rejection_reasons=rejection_reasons or [],
            validation_tier=validation_tier,
            validation_provider=validation_provider,
            validation_model=validation_model,
            validated_at=str(validated_at),
        )
