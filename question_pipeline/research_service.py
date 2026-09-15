import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
from .config import config
from .models import ResearchOutput, RolePipelineState
from .state_manager import state_manager
from .governor import ProviderBlockedException, QuotaExhaustedException
from .providers.factory import get_research_provider

def slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).lower()

class ResearchService:
    """Orchestrates cadence-aware, hash-checked web research."""
    def __init__(self):
        self.provider = get_research_provider()

    async def execute_research(self, role: str, force: bool = False) -> Dict[str, Any]:
        """Run web research for a role if cadence or force dictates."""
        state = state_manager.get_role_state(role)

        # Check cadence
        if not force and not state_manager.is_role_due_for_research(role):
            print(f"[RESEARCH] Role '{role}' was recently researched on {state.last_researched_at}. Cadence: {config.ROLE_RESEARCH_CADENCE_DAYS}d. Skipping research.")
            return {"status": "SKIPPED_CADENCE", "role": role}

        print(f"[RESEARCH] Initiating research for '{role}' using provider '{self.provider.provider_name}'...")

        try:
            output = await self.provider.research_role(role)
        except QuotaExhaustedException as qe:
            print(f"[RESEARCH] Quota exhausted for {qe.provider}/{qe.model}: {qe.message}")
            state_manager.mark_waiting_for_quota(role, f"Research quota exhausted: {qe.message}")
            raise
        except ProviderBlockedException as pbe:
            print(f"[RESEARCH] Provider blocked for {pbe.provider}/{pbe.model}: {pbe.message}")
            state_manager.mark_blocked(role, f"Research blocked: {pbe.message}")
            raise
        except Exception as e:
            print(f"[RESEARCH] Unexpected research error for '{role}': {e}")
            state_manager.update_role_state(role, status="FAILED", reason=str(e))
            raise

        # Check if research found anything meaningfully new (source hash check)
        if not force and state_manager.is_hash_unchanged(role, output.source_hash):
            print(f"[RESEARCH] Source content hash for '{role}' unchanged ({output.source_hash[:8]}...). No new information found.")
            state_manager.update_role_state(
                role,
                status="SKIPPED",
                last_researched_at=datetime.now(timezone.utc).isoformat(),
                reason="Source content hash unchanged"
            )
            return {"status": "SKIPPED_HASH_UNCHANGED", "role": role, "hash": output.source_hash}

        # Save research output file
        slug = slugify(role)
        file_path = config.KNOWLEDGE_DIR / f"{slug}_research.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)

        # Update state
        now = datetime.now(timezone.utc)
        next_research = (now + timedelta(days=config.ROLE_RESEARCH_CADENCE_DAYS)).isoformat()
        state_manager.update_role_state(
            role,
            status="RUNNING",
            last_researched_at=now.isoformat(),
            next_research_at=next_research,
            source_hash=output.source_hash,
            knowledge_version=output.knowledge_version,
            reason=None
        )

        return {
            "status": "SUCCESS",
            "role": role,
            "research_output": output,
            "actual_provider": output.actual_provider
        }

    async def execute_company_research(self, company: str) -> Dict[str, Any]:
        """Company-level counterpart to execute_research(), for the
        production orchestrator's COMPANY_RESEARCH jobs. Reuses the exact
        same provider abstraction, governor, and error types -- the only
        difference is calling provider.research_company() instead of
        provider.research_role(). Cadence/hash gating for company research is
        DB-authoritative (see freshness.py) and is the orchestrator's
        responsibility, done BEFORE this is called -- this method always
        executes when invoked, the same way state_manager-gated
        execute_research(..., force=True) does."""
        print(f"[RESEARCH] Initiating company research for '{company}' using provider '{self.provider.provider_name}'...")
        try:
            output = await self.provider.research_company(company)
        except QuotaExhaustedException as qe:
            print(f"[RESEARCH] Quota exhausted for {qe.provider}/{qe.model}: {qe.message}")
            raise
        except ProviderBlockedException as pbe:
            print(f"[RESEARCH] Provider blocked for {pbe.provider}/{pbe.model}: {pbe.message}")
            raise

        slug = slugify(company)
        file_path = config.KNOWLEDGE_DIR / f"company_{slug}_research.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)

        return {
            "status": "SUCCESS",
            "company": company,
            "research_output": output,
            "actual_provider": output.actual_provider,
        }


research_service = ResearchService()
