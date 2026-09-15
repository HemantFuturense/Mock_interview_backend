"""
DB-backed freshness and change-detection for the production rollout
orchestrator (question_pipeline/orchestrator.py).

Mirrors state_manager.py's cadence/hash-check pattern
(is_role_due_for_research, is_hash_unchanged, mark_waiting_for_quota,
mark_blocked) exactly -- same semantics, same cadence constants from
config.py -- but reads/writes `knowledge_state` / `knowledge_version_history`
in Postgres instead of the local state.json file, because production
orchestration state must be DB-authoritative, not filesystem-dependent.

state_manager.py itself is untouched and keeps working exactly as before for
the existing single-role CLI path (pipeline_runner.py's __main__).
"""
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import psycopg2

from .config import config


def bump_version(current: Optional[str]) -> str:
    """v1.0 -> v1.1, v1.9 -> v1.10, unparseable/None -> v1.0 (or v1.1 if a
    version string existed but didn't match the vX.Y pattern)."""
    if not current:
        return "v1.0"
    m = re.match(r"^v(\d+)\.(\d+)$", current.strip())
    if not m:
        return "v1.1"
    major, minor = int(m.group(1)), int(m.group(2))
    return f"v{major}.{minor + 1}"


class KnowledgeFreshnessTracker:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config

    def _connect(self):
        return psycopg2.connect(**self.db_config)

    def get_state(self, role: str, company: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                SELECT current_knowledge_version, source_hash, status, reason,
                       last_researched_at, next_research_at
                FROM knowledge_state WHERE role = %s AND company IS NOT DISTINCT FROM %s;
                """,
                (role, company),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "knowledge_version": row[0], "source_hash": row[1], "status": row[2],
                "reason": row[3], "last_researched_at": row[4], "next_research_at": row[5],
            }
        finally:
            conn.close()

    def is_due(self, role: str, company: Optional[str] = None) -> bool:
        """True if this (role[, company]) unit has never been researched, or
        its cadence window (role: ~30d, company: ~14d) has elapsed."""
        state = self.get_state(role, company)
        if not state or not state["last_researched_at"]:
            return True
        cadence_days = config.COMPANY_RESEARCH_CADENCE_DAYS if company else config.ROLE_RESEARCH_CADENCE_DAYS
        last_dt = state["last_researched_at"]
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last_dt) >= timedelta(days=cadence_days)

    def is_hash_unchanged(self, role: str, new_hash: str, company: Optional[str] = None) -> bool:
        state = self.get_state(role, company)
        return bool(state and state["source_hash"] and state["source_hash"] == new_hash)

    def record_no_change(
        self, role: str, company: Optional[str], source_hash: str,
        reason: str = "Source content hash unchanged",
    ) -> None:
        """Refresh freshness metadata WITHOUT bumping knowledge_version and
        WITHOUT touching the question bank -- the previous knowledge version
        is still current."""
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            now = datetime.now(timezone.utc)
            cadence_days = config.COMPANY_RESEARCH_CADENCE_DAYS if company else config.ROLE_RESEARCH_CADENCE_DAYS
            next_research = now + timedelta(days=cadence_days)
            cur.execute(
                """
                INSERT INTO knowledge_state (role, company, source_hash, status, reason, last_researched_at, next_research_at)
                VALUES (%s, %s, %s, 'COMPLETED', %s, %s, %s)
                ON CONFLICT (role, COALESCE(company, '')) DO UPDATE SET
                    source_hash = EXCLUDED.source_hash, status = EXCLUDED.status, reason = EXCLUDED.reason,
                    last_researched_at = EXCLUDED.last_researched_at, next_research_at = EXCLUDED.next_research_at,
                    updated_at = now();
                """,
                (role, company, source_hash, reason, now, next_research),
            )
            cur.execute(
                """
                INSERT INTO knowledge_version_history (role, company, source_hash, status, reason, recorded_at)
                VALUES (%s, %s, %s, 'SKIPPED', %s, now());
                """,
                (role, company, source_hash, reason),
            )
        finally:
            conn.close()

    def record_change(
        self, role: str, company: Optional[str], new_knowledge_version: str,
        source_hash: str, research_document_id: Optional[int],
        reason: str = "Meaningful knowledge change detected",
    ) -> None:
        """Bump the current knowledge version pointer AND append an
        immutable history row -- knowledge_version_history is never updated
        or deleted, so every prior version stays inspectable even after this
        call moves `knowledge_state` forward."""
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            now = datetime.now(timezone.utc)
            cadence_days = config.COMPANY_RESEARCH_CADENCE_DAYS if company else config.ROLE_RESEARCH_CADENCE_DAYS
            next_research = now + timedelta(days=cadence_days)
            cur.execute(
                """
                INSERT INTO knowledge_state (role, company, current_knowledge_version, current_research_document_id,
                                              source_hash, status, reason, last_researched_at, next_research_at)
                VALUES (%s, %s, %s, %s, %s, 'COMPLETED', %s, %s, %s)
                ON CONFLICT (role, COALESCE(company, '')) DO UPDATE SET
                    current_knowledge_version = EXCLUDED.current_knowledge_version,
                    current_research_document_id = COALESCE(EXCLUDED.current_research_document_id, knowledge_state.current_research_document_id),
                    source_hash = EXCLUDED.source_hash, status = EXCLUDED.status, reason = EXCLUDED.reason,
                    last_researched_at = EXCLUDED.last_researched_at, next_research_at = EXCLUDED.next_research_at,
                    updated_at = now();
                """,
                (role, company, new_knowledge_version, research_document_id, source_hash, reason, now, next_research),
            )
            cur.execute(
                """
                INSERT INTO knowledge_version_history (role, company, knowledge_version, research_document_id, source_hash, status, reason, recorded_at)
                VALUES (%s, %s, %s, %s, %s, 'COMPLETED', %s, now());
                """,
                (role, company, new_knowledge_version, research_document_id, source_hash, reason),
            )
        finally:
            conn.close()

    def mark_status(self, role: str, company: Optional[str], status: str, reason: Optional[str] = None) -> None:
        """Mirrors state_manager.mark_waiting_for_quota / mark_blocked, DB-side."""
        conn = self._connect()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO knowledge_state (role, company, status, reason)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (role, COALESCE(company, '')) DO UPDATE SET
                    status = EXCLUDED.status, reason = EXCLUDED.reason, updated_at = now();
                """,
                (role, company, status, reason),
            )
        finally:
            conn.close()
