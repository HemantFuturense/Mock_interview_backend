import json
import threading
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .config import config
from .models import RolePipelineState

class StateManager:
    """Manages resumable pipeline checkpoints across roles and companies."""
    def __init__(self, state_file: Optional[Path] = None):
        self._state_file = state_file or config.STATE_FILE
        self._states: Dict[str, RolePipelineState] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for role, s in data.items():
                        self._states[role] = RolePipelineState(**s)
            except Exception as e:
                print(f"[STATE_MANAGER] Warning: Could not load state file ({e}). Starting fresh.")
                self._states = {}

    def save(self) -> None:
        with self._lock:
            try:
                with open(self._state_file, "w", encoding="utf-8") as f:
                    json.dump({k: v.model_dump() for k, v in self._states.items()}, f, indent=2)
            except Exception as e:
                print(f"[STATE_MANAGER] Warning: Failed to save state file: {e}")

    def get_role_state(self, role: str) -> RolePipelineState:
        with self._lock:
            if role not in self._states:
                self._states[role] = RolePipelineState(role=role, status="PENDING")
            return self._states[role]

    def update_role_state(
        self,
        role: str,
        status: Optional[str] = None,
        current_batch: Optional[int] = None,
        total_batches: Optional[int] = None,
        target_questions: Optional[int] = None,
        accepted_questions_count: Optional[int] = None,
        last_researched_at: Optional[str] = None,
        next_research_at: Optional[str] = None,
        knowledge_version: Optional[str] = None,
        source_hash: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> RolePipelineState:
        with self._lock:
            state = self.get_role_state(role)
            if status is not None:
                state.status = status
            if target_questions is not None:
                state.target_questions = target_questions
            if current_batch is not None:
                state.current_batch = current_batch
            if total_batches is not None:
                state.total_batches = total_batches
            if accepted_questions_count is not None:
                state.accepted_questions_count = accepted_questions_count
            if last_researched_at is not None:
                state.last_researched_at = last_researched_at
            if next_research_at is not None:
                state.next_research_at = next_research_at
            if knowledge_version is not None:
                state.knowledge_version = knowledge_version
            if source_hash is not None:
                state.source_hash = source_hash
            if reason is not None:
                state.reason = reason
            state.updated_at = datetime.now(timezone.utc).isoformat()
            self._states[role] = state

        self.save()
        return state

    def mark_waiting_for_quota(self, role: str, reason: str) -> RolePipelineState:
        return self.update_role_state(role, status="WAITING_FOR_QUOTA", reason=reason)

    def mark_blocked(self, role: str, reason: str) -> RolePipelineState:
        return self.update_role_state(role, status="BLOCKED", reason=reason)

    def reset_role_state(self, role: str) -> RolePipelineState:
        """Reset state ONLY for the specified role for a forced fresh rerun."""
        with self._lock:
            state = RolePipelineState(
                role=role,
                status="PENDING",
                current_batch=0,
                total_batches=0,
                target_questions=0,
                accepted_questions_count=0,
                last_researched_at=None,
                next_research_at=None,
                knowledge_version="v1.0",
                source_hash="",
                reason=None,
                updated_at=datetime.now(timezone.utc).isoformat()
            )
            self._states[role] = state
        self.save()
        return state

    def is_role_due_for_research(self, role: str) -> bool:
        state = self.get_role_state(role)
        if not state.last_researched_at:
            return True
        try:
            last_dt = datetime.fromisoformat(state.last_researched_at)
            now = datetime.now(timezone.utc)
            cadence = timedelta(days=config.ROLE_RESEARCH_CADENCE_DAYS)
            return (now - last_dt) >= cadence
        except Exception:
            return True

    def is_hash_unchanged(self, role: str, new_hash: str) -> bool:
        state = self.get_role_state(role)
        if state.source_hash and state.source_hash == new_hash:
            return True
        return False

    def list_all_states(self) -> Dict[str, Dict]:
        with self._lock:
            return {k: v.model_dump() for k, v in self._states.items()}

state_manager = StateManager()
