import json
import asyncio
import argparse
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from .config import config, is_valid_key
from .models import QuestionObject, PilotQualityReport, RolePipelineState
from .state_manager import state_manager
from .cost_tracker import cost_tracker
from .research_service import research_service
from .synthesis_service import synthesis_service
from .rag_service import rag_service
from .generator_service import generator_service
from .validator_service import validator_service
from .governor import QuotaExhaustedException, ProviderBlockedException
from .validation_store import ValidationAttemptStore, InMemoryValidationAttemptStore

class QuestionPipelineRunner:
    """End-to-end orchestrator for automated interview question generation."""

    def __init__(self, validation_store: Optional[ValidationAttemptStore] = None):
        self.question_bank: List[QuestionObject] = []
        self._load_question_bank()
        # Defaults to an in-memory store because migration 0001 (which creates
        # question_validation_attempts) has not been executed against the live
        # database yet. Once it has, pass validation_store=PostgresValidationAttemptStore(...)
        # here -- no other code in this file needs to change.
        self.validation_store: ValidationAttemptStore = validation_store or InMemoryValidationAttemptStore(
            persist_path=config.DATA_DIR / "validation_attempts.json"
        )

    def _load_question_bank(self):
        if config.QUESTION_BANK_FILE.exists():
            try:
                with open(config.QUESTION_BANK_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.question_bank = [QuestionObject(**item) for item in data]
            except Exception as e:
                print(f"[PIPELINE] Warning loading existing question bank: {e}")
                self.question_bank = []

    def _save_question_bank(self):
        try:
            with open(config.QUESTION_BANK_FILE, "w", encoding="utf-8") as f:
                json.dump([q.model_dump() for q in self.question_bank], f, indent=2)
        except Exception as e:
            print(f"[PIPELINE] Error saving question bank: {e}")

    def clear_role_questions(self, role: str) -> int:
        """Remove questions ONLY for the specified role from the question bank."""
        before = len(self.question_bank)
        self.question_bank = [q for q in self.question_bank if q.role.lower() != role.lower()]
        removed = before - len(self.question_bank)
        if removed > 0:
            self._save_question_bank()
        return removed

    def print_diagnostics(self, role: str, force: bool = False) -> None:
        """Print clear startup diagnostics without revealing secret values."""
        mode_str = "MOCK" if config.MOCK_MODE else "REAL"
        force_str = "true" if force else "false"
        print("\n" + "=" * 60)
        print(" PIPELINE STARTUP DIAGNOSTICS")
        print("=" * 60)
        print(f"ROLE:                {role}")
        print(f"MODE:                {mode_str}")
        print(f"FORCE:               {force_str}")
        print(f"RESEARCH PROVIDER:   {config.RESEARCH_PROVIDER} ({config.RESEARCH_MODEL})")
        print(f"LLM PROVIDER:        {config.LLM_PROVIDER} ({config.LLM_MODEL})")
        print(f"EMBEDDING PROVIDER:  {config.EMBEDDING_PROVIDER} ({config.EMBEDDING_MODEL})")
        print(f"GENERATION MODEL:    {config.LLM_MODEL}")
        print(f"EMBEDDING MODEL:     {config.EMBEDDING_MODEL}")
        # Informational only -- the fallback/escalation tiers are optional
        # and best-effort by design (see validator_service.py's ladder), so
        # their absence never blocks the pipeline from starting. This just
        # gives visibility into whether a real second opinion is actually
        # reachable before a run, since "escalation_unavailable_uncertain_result"
        # rejections are otherwise easy to misread as question-quality problems.
        fallback_ready = is_valid_key(config.OPENAI_API_KEY) if config.FALLBACK_LLM_PROVIDER == "openai" \
            else is_valid_key(config.GEMINI_API_KEY) if config.FALLBACK_LLM_PROVIDER == "gemini" \
            else is_valid_key(config.ANTHROPIC_API_KEY) if config.FALLBACK_LLM_PROVIDER == "anthropic" else False
        escalation_ready = is_valid_key(config.ANTHROPIC_API_KEY) if config.ESCALATION_PROVIDER == "anthropic" \
            else is_valid_key(config.OPENAI_API_KEY) if config.ESCALATION_PROVIDER == "openai" \
            else is_valid_key(config.GEMINI_API_KEY) if config.ESCALATION_PROVIDER == "gemini" else False
        print(f"FALLBACK TIER (T2):  {config.FALLBACK_LLM_PROVIDER} ({config.FALLBACK_LLM_MODEL}) -- {'configured' if fallback_ready else 'NOT configured (uncertain validations will fail closed, not escalate)'}")
        print(f"ESCALATION TIER (T3): {config.ESCALATION_PROVIDER} ({config.ESCALATION_MODEL}) -- {'configured' if escalation_ready else 'NOT configured (uncertain validations will fail closed, not escalate)'}")
        print("=" * 60 + "\n")

    def validate_prerequisites(self, role: str) -> None:
        """Validate required configuration and credentials before making expensive API calls."""
        if config.MOCK_MODE:
            return

        missing = []

        # 1. Research provider credential check
        res_provider = config.RESEARCH_PROVIDER.lower()
        if res_provider == "perplexity":
            if not is_valid_key(config.PERPLEXITY_API_KEY):
                missing.append("PERPLEXITY_API_KEY is not configured or is a placeholder in .env (Required for research)")
        elif res_provider == "tavily_firecrawl":
            if not is_valid_key(config.TAVILY_API_KEY):
                missing.append("TAVILY_API_KEY is not configured or is a placeholder in .env (Required for research discovery)")
            if not is_valid_key(config.FIRECRAWL_API_KEY):
                missing.append("FIRECRAWL_API_KEY is not configured or is a placeholder in .env (Required for research extraction)")
        elif res_provider == "gemini":
            if not is_valid_key(config.GEMINI_API_KEY):
                missing.append("GEMINI_API_KEY is not configured or is a placeholder in .env (Required for research)")
        else:
            missing.append(f"Unsupported RESEARCH_PROVIDER '{res_provider}'. Must be 'perplexity' or 'tavily_firecrawl'.")

        # 2. LLM generation provider credential check
        llm_provider = config.LLM_PROVIDER.lower()
        if llm_provider == "gemini":
            if not is_valid_key(config.GEMINI_API_KEY):
                missing.append("GEMINI_API_KEY is not configured or is a placeholder in .env (Required for generation & synthesis)")
        elif llm_provider == "openai":
            if not is_valid_key(config.OPENAI_API_KEY):
                missing.append("OPENAI_API_KEY is not configured or is a placeholder in .env")
        elif llm_provider == "anthropic":
            if not is_valid_key(config.ANTHROPIC_API_KEY):
                missing.append("ANTHROPIC_API_KEY is not configured or is a placeholder in .env")

        # 3. Embedding provider credential check
        emb_provider = config.EMBEDDING_PROVIDER.lower()
        if emb_provider == "gemini":
            if not is_valid_key(config.GEMINI_API_KEY):
                missing.append("GEMINI_API_KEY is not configured or is a placeholder in .env (Required for embedding)")
        elif emb_provider != "local":
            missing.append(f"Unsupported EMBEDDING_PROVIDER '{emb_provider}'. Must be 'gemini' or 'local'.")

        if missing:
            msg = "Missing or placeholder credentials for real mode:\n  - " + "\n  - ".join(missing)
            state_manager.mark_blocked(role, msg)
            raise ProviderBlockedException(
                provider="credentials",
                model=config.LLM_MODEL,
                message=msg
            )

    def classify_scope(self, candidate: QuestionObject, target_scope: str = "UNIVERSAL") -> str:
        """Step 7: Explicitly classify and verify question scope (UNIVERSAL, DOMAIN, COMPANY).

        Fix (see migration/pilot review): the previous version treated the
        mere PRESENCE of a non-generic `domains` tag as sufficient evidence
        for DOMAIN scope, regardless of target_scope. Since real generator
        output almost always includes a specific domain tag, this made
        genuine UNIVERSAL questions unreachable -- every batch, including
        ones explicitly targeting UNIVERSAL, was silently promoted to
        DOMAIN. A `domains` tag is topical categorization; it is NOT, by
        itself, evidence that company- or industry-level context is
        required.

        The corrected logic trusts genuine content signals instead:
          - `applicable_companies` (now generated from the actual question
            content, not a copied example list -- see generator_service.py)
            is the primary signal: exactly one named company -> COMPANY,
            two or more -> DOMAIN.
          - With no company signal, an explicit UNIVERSAL target_scope is
            authoritative (a universal question may still be illustratively
            associated with several companies without becoming DOMAIN).
          - A self-reported COMPANY scope with no company actually named is
            not defensible and is downgraded rather than trusted blindly.
        """
        companies = [
            c.strip() for c in candidate.applicable_companies
            if c and c.strip().lower() not in ("all", "any", "general", "n/a", "none")
        ]

        # A single, genuinely-named company is decisive regardless of batch intent.
        if len(companies) == 1:
            return "COMPANY"

        # Respect an explicit UNIVERSAL generation intent as authoritative
        # when there is no company-specific signal to override it.
        if target_scope == "UNIVERSAL" and candidate.scope != "COMPANY":
            return candidate.scope if candidate.scope in config.SCOPES else "UNIVERSAL"

        if len(companies) >= 2:
            return "DOMAIN"

        # No genuine company signal and not a UNIVERSAL-targeted batch:
        # fall back to the candidate's own self-reported scope, except a
        # self-reported COMPANY with zero companies named is not defensible.
        if candidate.scope == "COMPANY":
            return "UNIVERSAL"
        if candidate.scope in config.SCOPES:
            return candidate.scope
        return "UNIVERSAL"

    async def run_role_pipeline(
        self,
        role: str,
        target_questions: int = 20,
        force: bool = False
    ) -> Dict[str, Any]:
        """Execute full pipeline for a single role with incremental checkpointing."""
        self.print_diagnostics(role, force=force)

        state = state_manager.get_role_state(role)
        if not force and state.status == "COMPLETED" and state.accepted_questions_count >= target_questions:
            print(f"[PIPELINE] Role '{role}' already COMPLETED with {state.accepted_questions_count} questions. Skipping.")
            return {"role": role, "status": "ALREADY_COMPLETED", "accepted": state.accepted_questions_count}

        metrics = {
            "role": role,
            "generated": 0,
            "accepted": 0,
            "rejected_validation": 0,
            "rejected_duplicate": 0,
            "errors": []
        }

        # Validate configuration and credentials before making any expensive calls
        try:
            self.validate_prerequisites(role)
        except ProviderBlockedException as pbe:
            print(f"\n[PIPELINE BLOCKED] {pbe.message}")
            metrics["errors"].append(pbe.message)
            return {"role": role, "status": "BLOCKED", "metrics": metrics}

        # If forced, reset ONLY this role's state, question bank entries, and RAG chunks
        if force:
            print(f"[PIPELINE] --force flag supplied. Resetting state and bank entries ONLY for role '{role}'...")
            state = state_manager.reset_role_state(role)
            removed_q = self.clear_role_questions(role)
            removed_c = rag_service.clear_role_chunks(role)
            print(f"[PIPELINE] Reset role '{role}': cleared {removed_q} questions from bank, {removed_c} chunks from RAG store.")

        state_manager.update_role_state(role, status="RUNNING", target_questions=target_questions)

        # Step 1, 2, 3: Research, Synthesis & RAG Indexing (Only if starting from batch 0)
        start_batch = state.current_batch
        if start_batch == 0:
            try:
                research_res = await research_service.execute_research(role, force=force)
            except (QuotaExhaustedException, ProviderBlockedException) as e:
                metrics["errors"].append(str(e))
                return {"role": role, "status": state.status, "metrics": metrics}
            except Exception as e:
                metrics["errors"].append(f"Research failed: {e}")
                state_manager.update_role_state(role, status="FAILED", reason=str(e))
                return {"role": role, "status": "FAILED", "metrics": metrics}

            if research_res.get("status") == "SKIPPED_HASH_UNCHANGED":
                print(f"[PIPELINE] Research content unchanged for '{role}'. Skipping regeneration.")
                return {"role": role, "status": "SKIPPED_UNCHANGED", "metrics": metrics}

            research_output = research_res.get("research_output")
            # If skipped due to cadence, load existing research file
            if not research_output and research_res.get("status") == "SKIPPED_CADENCE":
                from .research_service import slugify
                res_file = config.KNOWLEDGE_DIR / f"{slugify(role)}_research.json"
                if res_file.exists():
                    try:
                        with open(res_file, "r", encoding="utf-8") as f:
                            from .models import ResearchOutput
                            research_output = ResearchOutput(**json.load(f))
                    except Exception as e:
                        print(f"[PIPELINE] Could not load existing research file: {e}")

            if research_output:
                # Step 2: Synthesis
                try:
                    chunks = await synthesis_service.synthesize(research_output)
                except (QuotaExhaustedException, ProviderBlockedException) as e:
                    metrics["errors"].append(str(e))
                    return {"role": role, "status": state.status, "metrics": metrics}
                except Exception as e:
                    metrics["errors"].append(f"Synthesis failed: {e}")
                    state_manager.update_role_state(role, status="FAILED", reason=str(e))
                    return {"role": role, "status": "FAILED", "metrics": metrics}

                # Step 3: Vector RAG Indexing
                try:
                    await rag_service.index_chunks(chunks)
                except (QuotaExhaustedException, ProviderBlockedException) as e:
                    metrics["errors"].append(str(e))
                    return {"role": role, "status": state.status, "metrics": metrics}
                except Exception as e:
                    metrics["errors"].append(f"RAG indexing failed: {e}")
                    state_manager.update_role_state(role, status="FAILED", reason=str(e))
                    return {"role": role, "status": "FAILED", "metrics": metrics}
        else:
            print(f"[PIPELINE] Resuming role '{role}' from batch {start_batch + 1}. Skipping research/synthesis.")

        # Step 4: Batch Question Generation, Validation & Deduplication
        bands = config.EXPERIENCE_BANDS # ["0-1", "1-2", "3-5", "5-8", "8+"]
        questions_per_band = max(target_questions // len(bands), config.BATCH_SIZE)

        total_batches = len(bands)
        state_manager.update_role_state(role, total_batches=total_batches)

        for batch_idx in range(start_batch, total_batches):
            band = bands[batch_idx]
            print(f"\n[PIPELINE] Batch {batch_idx + 1}/{total_batches}: Generating for band '{band}' (Target: {questions_per_band})...")

            # Determine target scope distribution: ~75% UNIVERSAL, ~15% DOMAIN, ~10% COMPANY
            target_scope = "UNIVERSAL"
            if batch_idx == 1:
                target_scope = "DOMAIN"
            elif batch_idx == 4:
                target_scope = "COMPANY"

            try:
                candidates = await generator_service.generate_batch_for_band(
                    role=role,
                    experience_band=band,
                    count=questions_per_band,
                    target_scope=target_scope
                )
            except (QuotaExhaustedException, ProviderBlockedException) as e:
                metrics["errors"].append(str(e))
                # Checkpoint current batch
                state_manager.update_role_state(role, current_batch=batch_idx)
                return {"role": role, "status": state.status, "metrics": metrics}
            except Exception as e:
                print(f"[PIPELINE] Batch generation error: {e}")
                metrics["errors"].append(str(e))
                continue

            metrics["generated"] += len(candidates)

            # Step 5, 6, 7 & 8: Scope Classification -> Quality Gate (validation +
            # dedup + provenance/grounding, fail-closed) -> Persistence.
            #
            # Scope is classified BEFORE the quality gate (not after, as in the
            # pre-Phase-2 ordering) so the gate's correct_scope check evaluates
            # the actual final scope, not a placeholder the generator echoed back.
            #
            # Every attempt -- approved or rejected -- is recorded to
            # self.validation_store immediately, before moving to the next
            # candidate. This is the checkpointing guarantee: if candidate N+1
            # raises QuotaExhaustedException, attempts 1..N are already durable.
            try:
                for cand in candidates:
                    # Step 5: Scope Classification (deterministic, cheap)
                    cand.scope = self.classify_scope(cand, target_scope)

                    # Step 6, 7 & Provenance checks: unified Quality Gate
                    # (role invariance, difficulty/paradigm/scope correctness,
                    # dimension consistency, 3-tier dedup, RAG grounding, then
                    # Gemini -> OpenAI -> Claude rubric ladder). Fail-closed:
                    # a high score can never override a critical failure.
                    gate_result = await validator_service.validate_question(
                        cand, self.question_bank, expected_role=role
                    )

                    question_id: Optional[int] = None
                    if gate_result.approved:
                        self.question_bank.append(cand)
                        self._save_question_bank()
                        # File-mode placeholder id (question_bank.json has no
                        # real serial id yet). Once migration 0001 is applied
                        # and PostgresValidationAttemptStore/questions table
                        # are wired in, this becomes the real `questions.id`
                        # from INSERT ... RETURNING id.
                        question_id = len(self.question_bank)
                        metrics["accepted"] += 1
                        print(
                            f"  [APPROVED] Score: {gate_result.quality_score:.2f} | Tier: {gate_result.validation_tier} | "
                            f"Scope: {cand.scope} | Band: {cand.experience_band} | Diff: {cand.difficulty} -> \"{cand.question[:60]}...\""
                        )
                    else:
                        is_duplicate_rejection = any(
                            r.startswith("duplicate_question:") for r in gate_result.rejection_reasons
                        )
                        if is_duplicate_rejection:
                            metrics["rejected_duplicate"] += 1
                        else:
                            metrics["rejected_validation"] += 1

                        if gate_result.critical_failure:
                            print(f"  [REJECTED - CRITICAL] {gate_result.rejection_reasons} -> \"{cand.question[:60]}...\"")
                        else:
                            print(
                                f"  [REJECTED] Score: {gate_result.quality_score:.2f} ({gate_result.reasoning}) -> \"{cand.question[:60]}...\""
                            )

                    attempt = gate_result.to_attempt_record(
                        role=cand.role, candidate_question_text=cand.question, question_id=question_id
                    )
                    self.validation_store.record(attempt)

            except (QuotaExhaustedException, ProviderBlockedException) as e:
                metrics["errors"].append(str(e))
                total_role_accepted = len([q for q in self.question_bank if q.role.lower() == role.lower()])
                state_manager.update_role_state(
                    role,
                    current_batch=batch_idx,
                    accepted_questions_count=total_role_accepted
                )
                return {"role": role, "status": state.status, "metrics": metrics}

            # Accurate cumulative role count from persistent bank
            total_role_accepted = len([q for q in self.question_bank if q.role.lower() == role.lower()])

            # Checkpoint batch completion
            state_manager.update_role_state(
                role,
                current_batch=batch_idx + 1,
                accepted_questions_count=total_role_accepted
            )

        # Mark role completed
        total_role_accepted = len([q for q in self.question_bank if q.role.lower() == role.lower()])
        final_status = "COMPLETED" if total_role_accepted >= (target_questions * 0.7) else "PARTIAL"
        state_manager.update_role_state(
            role,
            status=final_status,
            accepted_questions_count=total_role_accepted,
            reason=None
        )

        return {"role": role, "status": final_status, "metrics": metrics}

    async def run_pilot(self, force: bool = False) -> PilotQualityReport:
        """Run the 6-role pilot (~120 questions) and generate pilot_quality_report.json."""
        print("\n" + "=" * 65)
        print("  STARTING 6-ROLE PILOT QUESTION GENERATION PIPELINE")
        print("=" * 65)
        print(f"Roles: {config.PILOT_ROLES}")
        print(f"Target per role: {config.PILOT_TARGET_QUESTIONS_PER_ROLE}")
        print(f"Total target questions: ~{len(config.PILOT_ROLES) * config.PILOT_TARGET_QUESTIONS_PER_ROLE}")
        print(f"Force Rerun: {force}")
        print("=" * 65 + "\n")

        all_metrics = []
        for role in config.PILOT_ROLES:
            res = await self.run_role_pipeline(
                role=role,
                target_questions=config.PILOT_TARGET_QUESTIONS_PER_ROLE,
                force=force
            )
            all_metrics.append(res)

            # Check if execution was halted due to quota or block
            status = res.get("status")
            if status in ("WAITING_FOR_QUOTA", "BLOCKED"):
                print(f"\n[PILOT PAUSED] Pipeline halted safely because role '{role}' status is '{status}'. Progress saved.")
                break

        # Generate Comprehensive Pilot Quality Report
        report = self._build_pilot_report(all_metrics)
        self._save_pilot_report(report)

        print("\n" + "=" * 65)
        print("  PILOT EXECUTION COMPLETE- REPORT GENERATED")
        print("=" * 65)
        print(f"Total Questions Generated: {report.total_generated}")
        print(f"Accepted Questions:        {report.accepted}")
        print(f"Duplicates Rejected:       {report.duplicates}")
        print(f"Validation Failures:       {report.validation_failures}")
        print(f"Scope Distribution:        {report.scope_distribution}")
        print(f"Estimated API Cost:        ${report.estimated_api_cost:.6f} USD")
        print("=" * 65 + "\n")

        return report

    def _build_pilot_report(self, results: List[Dict[str, Any]]) -> PilotQualityReport:
        pilot_questions = [q for q in self.question_bank if q.role in config.PILOT_ROLES]

        total_gen = sum(r.get("metrics", {}).get("generated", 0) for r in results)
        accepted = len(pilot_questions)
        duplicates = sum(r.get("metrics", {}).get("rejected_duplicate", 0) for r in results)
        val_fails = sum(r.get("metrics", {}).get("rejected_validation", 0) for r in results)

        scope_dist: Dict[str, int] = {}
        for q in pilot_questions:
            scope_dist[q.scope] = scope_dist.get(q.scope, 0) + 1

        diff_dist: Dict[str, int] = {}
        for q in pilot_questions:
            diff_dist[str(q.difficulty)] = diff_dist.get(str(q.difficulty), 0) + 1

        exp_dist: Dict[str, int] = {}
        for q in pilot_questions:
            exp_dist[q.experience_band] = exp_dist.get(q.experience_band, 0) + 1

        # Role correctness check (verify role invariant across all questions)
        role_correctness = {}
        for role in config.PILOT_ROLES:
            role_qs = [q for q in pilot_questions if q.role == role]
            role_correctness[role] = all(q.role == role for q in role_qs) if role_qs else True

        # Cost tracking summary
        cost_summary = cost_tracker.get_summary()

        all_errors = []
        for r in results:
            errs = r.get("metrics", {}).get("errors", [])
            all_errors.extend(errs)

        return PilotQualityReport(
            total_generated=max(total_gen, accepted),
            accepted=accepted,
            rejected=val_fails + duplicates,
            duplicates=duplicates,
            validation_failures=val_fails,
            scope_distribution=scope_dist,
            difficulty_distribution=diff_dist,
            experience_distribution=exp_dist,
            role_correctness=role_correctness,
            api_usage=cost_summary.get("usage_by_provider", {}),
            estimated_api_cost=cost_summary.get("total_estimated_cost_usd", 0.0),
            cost_breakdown_by_provider=cost_summary.get("cost_by_provider", {}),
            cost_breakdown_by_task=cost_summary.get("cost_by_task", {}),
            failures_or_errors=all_errors
        )

    def _save_pilot_report(self, report: PilotQualityReport):
        try:
            with open(config.PILOT_REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(report.model_dump(), f, indent=2)
            print(f"[REPORT] Saved pilot report to: {config.PILOT_REPORT_FILE}")
        except Exception as e:
            print(f"[REPORT] Error saving pilot report: {e}")

pipeline_runner = QuestionPipelineRunner()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Interview Question Generation Pipeline")
    parser.add_argument("--pilot", action="store_true", help="Run the 6-role pilot (~120 questions)")
    parser.add_argument("--role", type=str, help="Run generation for a single role")
    parser.add_argument("--force", action="store_true", help="Force fresh rerun of role, resetting its state and artifacts")
    parser.add_argument("--resume", action="store_true", help="Resume pipeline from existing checkpoints")
    parser.add_argument("--status", action="store_true", help="Display current pipeline states and cost summary")
    parser.add_argument("--count", type=int, default=20, help="Target questions per role (default: 20)")
    args = parser.parse_args()

    if args.status:
        print("\n--- Pipeline State Summary ---")
        print(json.dumps(state_manager.list_all_states(), indent=2))
        print("\n--- Cost & Usage Summary ---")
        print(json.dumps(cost_tracker.get_summary(), indent=2))
    elif args.pilot or args.resume:
        asyncio.run(pipeline_runner.run_pilot(force=args.force))
    elif args.role:
        asyncio.run(pipeline_runner.run_role_pipeline(args.role, target_questions=args.count, force=args.force))
    else:
        parser.print_help()
