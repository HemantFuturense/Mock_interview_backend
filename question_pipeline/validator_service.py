"""
Phase 2 Quality Gate.

Every generated QuestionObject candidate passes through here before it is
allowed anywhere near the approved question bank. Two kinds of checks run:

  1. DETERMINISTIC checks (no LLM call, no ambiguity, cannot be argued with):
     role invariance, experience/difficulty band bounds, paradigm-for-band,
     scope validity, difficulty/dimension self-consistency, duplicate/near-
     duplicate (via the existing 3-tier deduplicator_service), and RAG
     grounding availability (via the existing rag_service).

  2. LLM rubric checks (Gemini primary -> OpenAI fallback -> Claude
     escalation, same ladder shape as before): role/technical/experience
     correctness, clarity, real-interview relevance, reasoning depth,
     hallucination, triviality, and whether the question is actually grounded
     in the RAG context it was given (the retrieved chunks are placed directly
     in the prompt so this is a judgment about real text, not a guess).

FAIL-CLOSED CONTRACT: if ANY deterministic check fails, the candidate is
rejected -- full stop. The LLM's own score/verdict is still computed (so the
audit trail captures its opinion), but it can never override a deterministic
failure. This is the literal enforcement of "a high overall score must never
override a critical failure": see `_merge()` below, and the fail-closed test
suite in tests/test_quality_gate.py.
"""
from typing import Dict, Any, List, Optional, Tuple
from .config import config
from .models import QuestionObject, QualityGateResult
from .providers.factory import get_llm_provider
from .rag_service import rag_service as _default_rag_service, RAGService
from .deduplicator_service import deduplicator_service
from .governor import QuotaExhaustedException, ProviderBlockedException


class ValidatorService:
    """Multi-tier quality gate: deterministic pre-checks + Gemini (primary) ->
    OpenAI (fallback) -> Claude (escalation) LLM rubric, fail-closed throughout."""

    def __init__(self, rag_service_instance: Optional[RAGService] = None):
        self.primary_llm = get_llm_provider(config.VALIDATION_PROVIDER)
        self.rag = rag_service_instance or _default_rag_service

    # ------------------------------------------------------------------
    # Deterministic checks
    # ------------------------------------------------------------------

    def _check_role_invariance(self, q: QuestionObject, expected_role: Optional[str]) -> Optional[str]:
        """Requirement: experience changes difficulty/depth, NEVER the role."""
        if expected_role and q.role != expected_role:
            return f"role_mismatch: expected role '{expected_role}', candidate has role '{q.role}'"
        return None

    def _check_difficulty_band(self, q: QuestionObject) -> Optional[str]:
        min_d, max_d = config.EXPERIENCE_DIFFICULTY_BOUNDS.get(q.experience_band, (1, 10))
        if not (min_d <= q.difficulty <= max_d):
            return (
                f"difficulty_out_of_band: difficulty {q.difficulty} outside calibrated "
                f"range [{min_d}, {max_d}] for experience band '{q.experience_band}'"
            )
        return None

    def _check_paradigm(self, q: QuestionObject) -> Optional[str]:
        expected = config.PARADIGM_BY_EXPERIENCE_BAND.get(q.experience_band)
        if expected and q.paradigm != expected:
            return f"incorrect_paradigm: expected '{expected}' for band '{q.experience_band}', got '{q.paradigm}'"
        return None

    def _check_scope(self, q: QuestionObject) -> Optional[str]:
        if q.scope not in config.SCOPES:
            return f"invalid_scope: '{q.scope}' is not one of {config.SCOPES}"
        return None

    def _check_dimension_consistency(self, q: QuestionObject) -> Optional[str]:
        dims = [
            q.technical_depth, q.problem_complexity, q.architecture_complexity,
            q.troubleshooting, q.business_complexity, q.decision_making,
            q.leadership_ownership,
        ]
        avg_dim = sum(dims) / len(dims)
        tolerance = config.DIFFICULTY_DIMENSION_CONSISTENCY_TOLERANCE
        if abs(q.difficulty - avg_dim) > tolerance:
            return (
                f"difficulty_dimension_inconsistency: overall difficulty {q.difficulty} "
                f"is inconsistent with the average of the 7 difficulty dimensions "
                f"({avg_dim:.1f}, tolerance {tolerance})"
            )
        return None

    async def _check_duplicate(self, q: QuestionObject, existing_bank: List[QuestionObject]) -> Optional[str]:
        is_dup, reason = await deduplicator_service.check_duplicate(q, existing_bank)
        if is_dup:
            return f"duplicate_question: {reason}"
        return None

    async def _retrieve_rag_context(self, q: QuestionObject):
        return await self.rag.retrieve_context(q.role, q.experience_band, q.paradigm, top_k=3)

    async def _run_deterministic_checks(
        self, q: QuestionObject, existing_bank: List[QuestionObject], expected_role: Optional[str]
    ) -> Tuple[List[str], list]:
        """Returns (rejection_reasons, rag_chunks). rag_chunks is returned too
        so the caller doesn't have to retrieve it a second time for the LLM prompt."""
        reasons: List[str] = []

        for check in (
            self._check_role_invariance(q, expected_role),
            self._check_difficulty_band(q),
            self._check_paradigm(q),
            self._check_scope(q),
            self._check_dimension_consistency(q),
        ):
            if check:
                reasons.append(check)

        dup_reason = await self._check_duplicate(q, existing_bank)
        if dup_reason:
            reasons.append(dup_reason)

        rag_chunks = await self._retrieve_rag_context(q)
        if not rag_chunks:
            reasons.append("insufficient_rag_grounding: no research/RAG context available to support this question")

        return reasons, rag_chunks

    # ------------------------------------------------------------------
    # LLM rubric (ladder)
    # ------------------------------------------------------------------

    def _build_prompt(self, q: QuestionObject, rag_chunks: list) -> str:
        context_text = (
            "\n\n".join(f"[{c.topic}]: {c.text}" for c in rag_chunks)
            if rag_chunks else "NO RESEARCH CONTEXT WAS RETRIEVED FOR THIS CANDIDATE."
        )
        dims = (
            f"technical_depth={q.technical_depth}, problem_complexity={q.problem_complexity}, "
            f"architecture_complexity={q.architecture_complexity}, troubleshooting={q.troubleshooting}, "
            f"business_complexity={q.business_complexity}, decision_making={q.decision_making}, "
            f"leadership_ownership={q.leadership_ownership}"
        )
        return (
            f"You are an uncompromising Bar Raiser and Principal Interview Evaluator. "
            f"Evaluate this candidate interview question against the retrieved research context. "
            f"Be strict: your job is to catch problems, not to be encouraging.\n\n"
            f"CANDIDATE QUESTION: \"{q.question}\"\n"
            f"TARGET ROLE (must not be altered): \"{q.role}\"\n"
            f"EXPERIENCE BAND: \"{q.experience_band}\"\n"
            f"CLAIMED DIFFICULTY: {q.difficulty}/10\n"
            f"CLAIMED PARADIGM: {q.paradigm}\n"
            f"CLAIMED SCOPE: {q.scope}\n"
            f"CLAIMED 7 DIFFICULTY DIMENSIONS: {dims}\n"
            f"MANDATORY SKILLS: {', '.join(q.mandatory_skills) if q.mandatory_skills else 'none listed'}\n\n"
            f"RETRIEVED RESEARCH / RAG CONTEXT (the ONLY source of truth for factual grounding):\n"
            f"{context_text}\n\n"
            "Evaluate EVERY one of these dimensions explicitly:\n"
            "1. role_correctness: Is this genuinely a question for the stated role, not a different role?\n"
            "2. technical_correctness: Is everything the question asserts or implies technically accurate?\n"
            "3. experience_seniority_fit: Is the expected depth of answer appropriate for this experience band "
            "   (note: for junior/EXECUTION-level questions, low business/decision/leadership dimensions are "
            "   CORRECT, not a defect -- judge fit for the band, not uniform 'importance')?\n"
            "4. difficulty_fit: Does the question's actual difficulty match the claimed difficulty and band?\n"
            "5. clarity: Is the wording precise, with no ambiguous interpretation?\n"
            "6. real_interview_relevance: Would a real interviewer at a real company actually ask this?\n"
            "7. reasoning_depth: Does it require reasoning, judgment, or applied understanding -- not memorization?\n"
            "8. no_hallucination: Does the question rely on any technology, statistic, or claim NOT supported by "
            "   the research context and not universally common knowledge? If so this is a hallucination.\n"
            "9. grounded_in_research: Is the question's subject matter substantively grounded in the retrieved "
            "   research context above (not just generically plausible)?\n"
            "10. not_trivial: Is this more than search-engine trivia (a fact lookup with no reasoning required)?\n"
            "11. correct_paradigm: Does the question's actual content match its claimed paradigm "
            f"    ({q.paradigm})?\n"
            "12. correct_dimensions: Are the 7 claimed difficulty dimensions a believable, internally consistent "
            "    self-rating for what this question actually asks?\n\n"
            "Return ONLY this JSON structure:\n"
            "{\n"
            '  "score": 0.85,\n'
            '  "passed": true,\n'
            '  "uncertain": false,\n'
            '  "reasoning": "one or two sentence justification",\n'
            '  "checks": {\n'
            '    "role_correctness": true, "technical_correctness": true, "experience_seniority_fit": true,\n'
            '    "difficulty_fit": true, "clarity": true, "real_interview_relevance": true,\n'
            '    "reasoning_depth": true, "no_hallucination": true, "grounded_in_research": true,\n'
            '    "not_trivial": true, "correct_paradigm": true, "correct_dimensions": true\n'
            "  },\n"
            '  "critical_failures": []\n'
            "}\n"
            "Put a short machine-readable reason string into critical_failures for EVERY check above that is "
            "false (e.g. \"technically_incorrect\", \"hallucinated_technology\", \"wrong_role\", "
            "\"materially_ambiguous\"). Set 'uncertain': true only if the overall score is genuinely borderline "
            f"(between {config.VALIDATION_UNCERTAIN_LOW} and {config.VALIDATION_UNCERTAIN_HIGH})."
        )

    async def _run_llm_tier(self, q: QuestionObject, rag_chunks: list, llm, tier_name: str) -> Dict[str, Any]:
        prompt = self._build_prompt(q, rag_chunks)
        try:
            res = await llm.generate_json(
                prompt=prompt,
                system_prompt="You are an uncompromising Bar Raiser and Principal Interview Evaluator.",
                task_name=f"validation_{tier_name}",
            )
        except (QuotaExhaustedException, ProviderBlockedException):
            # Never silently absorb these -- the caller (pipeline_runner) must
            # see them to checkpoint and pause, not have a fabricated result
            # returned in their place.
            raise
        except Exception as e:
            # Genuine unexpected error (bad JSON, provider hiccup, etc.):
            # fail closed -- reject rather than guess.
            return {
                "score": 0.0, "passed": False, "uncertain": False,
                "reasoning": f"Validation call failed: {e}",
                "checks": {}, "critical_failures": ["validation_call_error"],
                "provider": getattr(llm, "provider_name", tier_name),
                "model": getattr(llm, "model_name", tier_name),
            }

        data = res.data or {}
        score = float(data.get("score", 0.0))
        uncertain = bool(
            data.get("uncertain", False)
            or (config.VALIDATION_UNCERTAIN_LOW <= score < config.VALIDATION_UNCERTAIN_HIGH)
        )
        return {
            "score": score,
            "passed": bool(data.get("passed", score >= config.VALIDATION_PASS_SCORE)),
            "uncertain": uncertain,
            "reasoning": str(data.get("reasoning", f"Validated via {tier_name}")),
            "checks": dict(data.get("checks", {})),
            "critical_failures": list(data.get("critical_failures", [])),
            "provider": getattr(llm, "provider_name", tier_name),
            "model": getattr(llm, "model_name", tier_name),
        }

    def _merge(
        self, llm_result: Dict[str, Any], tier_name: str,
        deterministic_reasons: List[str],
    ) -> QualityGateResult:
        det_critical = len(deterministic_reasons) > 0
        llm_critical_failures = list(llm_result.get("critical_failures", []))
        llm_critical = len(llm_critical_failures) > 0

        critical_failure = det_critical or llm_critical
        rejection_reasons = list(deterministic_reasons) + llm_critical_failures

        # FAIL-CLOSED: a critical failure -- deterministic OR LLM-flagged --
        # makes approval impossible no matter how high `score` is.
        approved = (not critical_failure) and bool(llm_result.get("passed")) and not llm_result.get("uncertain")

        if not approved and not critical_failure and not rejection_reasons:
            rejection_reasons = ["quality_score_below_threshold_or_unresolved_uncertainty"]

        return QualityGateResult(
            approved=approved,
            quality_score=llm_result.get("score", 0.0),
            critical_failure=critical_failure,
            rejection_reasons=rejection_reasons,
            check_results=llm_result.get("checks", {}),
            validation_tier=tier_name,
            validation_provider=llm_result.get("provider", tier_name),
            validation_model=llm_result.get("model", ""),
            reasoning=llm_result.get("reasoning", ""),
            uncertain=llm_result.get("uncertain", False),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def validate_question(
        self,
        candidate: QuestionObject,
        existing_bank: Optional[List[QuestionObject]] = None,
        expected_role: Optional[str] = None,
    ) -> QualityGateResult:
        """Run a candidate through the full quality gate. Always logs a
        structured result; the caller is responsible for persisting it to
        question_validation_attempts (see pipeline_runner.py)."""
        existing_bank = existing_bank or []

        det_reasons, rag_chunks = await self._run_deterministic_checks(candidate, existing_bank, expected_role)

        # Tier 1: primary (Gemini). Always runs -- it remains the primary,
        # cost-efficient validator regardless of what the deterministic
        # checks found, so the audit trail always has an LLM opinion attached.
        tier1_raw = await self._run_llm_tier(candidate, rag_chunks, self.primary_llm, "primary_gemini")
        result = self._merge(tier1_raw, "primary_gemini", det_reasons)

        if det_reasons:
            # Deterministic failures are not a matter of LLM opinion -- there
            # is nothing for a second opinion to resolve, so don't spend
            # additional tier-2/3 budget on an already-settled rejection.
            return result

        if not tier1_raw["uncertain"]:
            return result

        # Tier 2/3 escalation: only reached when tier 1 itself is uncertain
        # and no deterministic check already resolved the outcome.
        print(f"[QUALITY_GATE] Tier 1 uncertain (score {tier1_raw['score']:.2f}). Escalating to Tier 2 ({config.FALLBACK_LLM_PROVIDER})...")
        try:
            fallback_llm = get_llm_provider(config.FALLBACK_LLM_PROVIDER)
            tier2_raw = await self._run_llm_tier(candidate, rag_chunks, fallback_llm, "fallback_gpt")
            result2 = self._merge(tier2_raw, "fallback_gpt", det_reasons)

            if not tier2_raw["uncertain"]:
                return result2

            print(f"[QUALITY_GATE] Tier 2 still uncertain (score {tier2_raw['score']:.2f}). Escalating to Tier 3 ({config.ESCALATION_PROVIDER})...")
            escalation_llm = get_llm_provider(config.ESCALATION_PROVIDER)
            tier3_raw = await self._run_llm_tier(candidate, rag_chunks, escalation_llm, "escalation_claude")
            result3 = self._merge(tier3_raw, "escalation_claude", det_reasons)

            if tier3_raw["uncertain"]:
                # Even the top of the ladder couldn't resolve it. QUALITY MUST
                # NEVER BE COMPROMISED: an unresolved "maybe" is treated as a
                # rejection, not an approval on a shaky score.
                result3.approved = False
                if "unresolved_uncertainty_after_full_ladder" not in result3.rejection_reasons:
                    result3.rejection_reasons.append("unresolved_uncertainty_after_full_ladder")
            return result3

        except (QuotaExhaustedException, ProviderBlockedException) as e:
            # Escalation tiers unavailable (missing credentials or quota
            # exhausted). QUALITY MUST NEVER BE COMPROMISED: we do NOT fall
            # back to accepting tier 1's own uncertain verdict -- an
            # unconfirmed "maybe" is a rejection, not an approval. This is a
            # deliberate strengthening over the pre-Phase-2 behavior, which
            # used to accept the primary tier's shaky result in this case.
            result.approved = False
            if "escalation_unavailable_uncertain_result" not in result.rejection_reasons:
                result.rejection_reasons.append("escalation_unavailable_uncertain_result")
            result.reasoning += f" [Escalation unavailable: {e}]"
            return result


validator_service = ValidatorService()
