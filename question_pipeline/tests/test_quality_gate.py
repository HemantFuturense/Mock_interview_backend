"""
Phase 2 Quality Gate test suite.

Covers the required scenarios A-M from the quality-gate implementation
request: fail-closed behavior (a high score must never override a critical
failure), approval of genuinely good questions, role invariance across
experience bands, checkpointing under quota failure, and that every
validation attempt -- approved or rejected -- is logged with provider/model/
tier and never lets a rejected question reach the approved bank.

All providers are fakes/mocks -- no real network calls, no real API keys
required, and MOCK_MODE=true is set (matching tests/test_pipeline.py's
convention) before importing question_pipeline.config.
"""
import os
import asyncio
import shutil
import unittest
from pathlib import Path
from typing import Optional

os.environ["MOCK_MODE"] = "true"

from question_pipeline.models import QuestionObject, KnowledgeChunk
from question_pipeline.providers.base import LLMProvider, LLMResult
from question_pipeline.rag_service import RAGService
from question_pipeline.validator_service import ValidatorService
from question_pipeline.validation_store import (
    InMemoryValidationAttemptStore,
    FailClosedViolation,
)
from question_pipeline.governor import QuotaExhaustedException, ProviderBlockedException


def make_question(**overrides) -> QuestionObject:
    base = dict(
        question="How would you detect and respond to data drift in a production ML system serving live traffic?",
        role="Machine Learning Engineer",
        experience_band="3-5",
        difficulty=7,
        technical_depth=7,
        problem_complexity=7,
        architecture_complexity=6,
        troubleshooting=7,
        business_complexity=4,
        decision_making=5,
        leadership_ownership=3,
        question_type="system_design",
        paradigm="ARCHITECTURE",
        mandatory_skills=["Monitoring", "Statistics"],
        scope="UNIVERSAL",
        domains=["ML_PLATFORM"],
        applicable_companies=["Google", "Amazon", "Meta"],
        source_references=["RAG Chunk: chunk_1 | Topic: Drift Detection"],
    )
    base.update(overrides)
    return QuestionObject(**base)


class FakeLLM(LLMProvider):
    """Fully scriptable fake LLM for the validation ladder. One instance
    answers every call the same way unless `responses` (a list) is given, in
    which case each successive call pops the next scripted response."""

    def __init__(
        self,
        score: float = 0.9,
        passed: bool = True,
        uncertain: bool = False,
        critical_failures=None,
        checks=None,
        reasoning: str = "Looks good.",
        name: str = "fake_gemini",
        model_name: str = "fake-model-v1",
        raise_exc: Optional[Exception] = None,
        responses: Optional[list] = None,
    ):
        self._name = name
        self.model_name = model_name
        self._score = score
        self._passed = passed
        self._uncertain = uncertain
        self._critical_failures = critical_failures or []
        self._checks = checks or {}
        self._reasoning = reasoning
        self._raise_exc = raise_exc
        self._responses = responses
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None, task_name: str = "validation") -> LLMResult:
        self.call_count += 1
        if self._raise_exc is not None:
            raise self._raise_exc

        if self._responses is not None:
            data = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        else:
            data = {
                "score": self._score,
                "passed": self._passed,
                "uncertain": self._uncertain,
                "reasoning": self._reasoning,
                "checks": self._checks,
                "critical_failures": self._critical_failures,
            }
        return LLMResult(text=str(data), data=data, model=self.model_name, provider=self._name)

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, task_name: str = "text") -> LLMResult:
        return await self.generate_json(prompt, system_prompt, task_name)


class QualityGateTestBase(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "temp_quality_gate_data"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _grounded_rag(self) -> RAGService:
        """A RAGService instance pre-indexed with a chunk relevant to
        'Machine Learning Engineer', isolated to this test (not the shared
        production vector_store.json singleton)."""
        rag = RAGService(store_file=self.test_dir / "grounded_vector_store.json")

        async def _index():
            await rag.index_chunks([
                KnowledgeChunk(
                    chunk_id="chunk_drift_1",
                    role="Machine Learning Engineer",
                    topic="Drift Detection",
                    text="Production ML systems monitor for data drift and concept drift using statistical tests "
                         "(KS-test, PSI) on feature distributions, comparing live traffic against training baselines.",
                    knowledge_version="v1.0",
                )
            ])
        asyncio.run(_index())
        return rag

    def _empty_rag(self) -> RAGService:
        return RAGService(store_file=self.test_dir / "empty_vector_store.json")

    def _gate(self, llm: FakeLLM, rag: Optional[RAGService] = None) -> ValidatorService:
        gate = ValidatorService(rag_service_instance=rag or self._grounded_rag())
        gate.primary_llm = llm
        return gate


class TestFailClosedRejections(QualityGateTestBase):
    """A. High score + critical_failure=true -> REJECT
       C. High score + technically incorrect -> REJECT
       D. High score + unsupported/hallucinated claim -> REJECT
    """

    def test_A_high_score_with_critical_failure_is_rejected(self):
        llm = FakeLLM(score=0.97, passed=True, uncertain=False, critical_failures=["arbitrary_critical_issue"])
        gate = self._gate(llm)

        result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved, "A 0.97 score must not buy approval when a critical failure is flagged")
        self.assertTrue(result.critical_failure)
        self.assertIn("arbitrary_critical_issue", result.rejection_reasons)
        self.assertEqual(result.quality_score, 0.97)  # score is recorded truthfully, just doesn't win

    def test_C_high_score_but_technically_incorrect_is_rejected(self):
        llm = FakeLLM(score=0.95, passed=True, uncertain=False, critical_failures=["technically_incorrect"])
        gate = self._gate(llm)

        result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertIn("technically_incorrect", result.rejection_reasons)

    def test_D_high_score_but_hallucinated_claim_is_rejected(self):
        llm = FakeLLM(score=0.93, passed=True, uncertain=False, critical_failures=["hallucinated_technology"])
        gate = self._gate(llm)

        result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertIn("hallucinated_technology", result.rejection_reasons)


class TestDeterministicRejections(QualityGateTestBase):
    """B. High score + wrong role -> REJECT
       E. Duplicate question -> REJECT
       H. Missing research/RAG evidence -> REJECT
    """

    def test_B_high_score_but_wrong_role_is_rejected(self):
        # LLM is fully convinced -- but the candidate's role does not match
        # what the pipeline actually asked for.
        llm = FakeLLM(score=0.98, passed=True, uncertain=False, critical_failures=[])
        gate = self._gate(llm)

        candidate = make_question(role="Data Engineer")
        result = asyncio.run(gate.validate_question(candidate, [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertTrue(any("role_mismatch" in r for r in result.rejection_reasons))
        # The LLM's own opinion never even gets a chance to matter here --
        # this must hold even though the LLM said 0.98/passed.
        self.assertEqual(result.quality_score, 0.98)

    def test_E_duplicate_question_is_rejected(self):
        llm = FakeLLM(score=0.90, passed=True, uncertain=False, critical_failures=[])
        gate = self._gate(llm)

        existing = make_question()
        candidate = existing.model_copy(update={"question": existing.question.upper()})  # exact dup, case-insensitive

        result = asyncio.run(gate.validate_question(candidate, [existing], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertTrue(any(r.startswith("duplicate_question:") for r in result.rejection_reasons))

    def test_H_missing_rag_evidence_is_rejected(self):
        llm = FakeLLM(score=0.92, passed=True, uncertain=False, critical_failures=[])
        gate = self._gate(llm, rag=self._empty_rag())  # no chunks indexed at all

        result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertTrue(any("insufficient_rag_grounding" in r for r in result.rejection_reasons))


class TestApprovalAndInvariants(QualityGateTestBase):
    """F. Correct high-quality question -> APPROVE
       G. Experience changes difficulty/depth but preserves role
       L. Provider/model/tier are recorded
       M. Provenance is preserved
    """

    def test_F_correct_high_quality_question_is_approved(self):
        llm = FakeLLM(
            score=0.91, passed=True, uncertain=False, critical_failures=[],
            checks={"role_correctness": True, "technical_correctness": True, "no_hallucination": True},
        )
        gate = self._gate(llm)

        result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertTrue(result.approved)
        self.assertFalse(result.critical_failure)
        self.assertEqual(result.rejection_reasons, [])
        self.assertAlmostEqual(result.quality_score, 0.91)

    def test_G_experience_changes_difficulty_but_never_role(self):
        llm = FakeLLM(score=0.9, passed=True, uncertain=False, critical_failures=[])
        rag = self._grounded_rag()
        gate = self._gate(llm, rag=rag)

        junior = make_question(
            question="Write a script to compute basic descriptive statistics (mean, std) for a training dataset column.",
            experience_band="0-1", paradigm="EXECUTION", difficulty=4,
            technical_depth=4, problem_complexity=3, architecture_complexity=2,
            troubleshooting=3, business_complexity=1, decision_making=1, leadership_ownership=1,
        )
        senior = make_question(
            question="Define the multi-year strategy for how this org detects and remediates data drift across 50+ production models.",
            experience_band="8+", paradigm="DOMAIN_OWNERSHIP", difficulty=9,
            technical_depth=8, problem_complexity=9, architecture_complexity=8,
            troubleshooting=7, business_complexity=8, decision_making=9, leadership_ownership=9,
        )

        result_junior = asyncio.run(gate.validate_question(junior, [], expected_role="Machine Learning Engineer"))
        result_senior = asyncio.run(gate.validate_question(senior, [], expected_role="Machine Learning Engineer"))

        self.assertTrue(result_junior.approved, result_junior.rejection_reasons)
        self.assertTrue(result_senior.approved, result_senior.rejection_reasons)
        # Role is identical across both -- only difficulty/paradigm/dimensions changed.
        self.assertEqual(junior.role, "Machine Learning Engineer")
        self.assertEqual(senior.role, "Machine Learning Engineer")
        self.assertNotEqual(junior.difficulty, senior.difficulty)
        self.assertNotEqual(junior.paradigm, senior.paradigm)

    def test_L_provider_model_tier_are_recorded(self):
        llm = FakeLLM(score=0.9, passed=True, uncertain=False, name="fake_gemini", model_name="fake-gemini-flash")
        gate = self._gate(llm)

        approved = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))
        self.assertEqual(approved.validation_tier, "primary_gemini")
        self.assertEqual(approved.validation_provider, "fake_gemini")
        self.assertEqual(approved.validation_model, "fake-gemini-flash")

        rejecting_llm = FakeLLM(score=0.95, critical_failures=["technically_incorrect"], name="fake_gemini", model_name="fake-gemini-flash")
        gate2 = self._gate(rejecting_llm)
        rejected = asyncio.run(gate2.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))
        self.assertEqual(rejected.validation_tier, "primary_gemini")
        self.assertEqual(rejected.validation_provider, "fake_gemini")
        self.assertEqual(rejected.validation_model, "fake-gemini-flash")

    def test_M_provenance_is_preserved_through_the_gate(self):
        llm = FakeLLM(score=0.9, passed=True, uncertain=False)
        rag = self._grounded_rag()
        gate = self._gate(llm, rag=rag)

        candidate = make_question(source_references=["RAG Chunk: chunk_drift_1 | Topic: Drift Detection"])
        original_refs = list(candidate.source_references)

        result = asyncio.run(gate.validate_question(candidate, [], expected_role="Machine Learning Engineer"))

        self.assertTrue(result.approved)
        # The quality gate must not mutate/lose the candidate's own provenance trail.
        self.assertEqual(candidate.source_references, original_refs)
        # And the chunk it was actually grounded against is traceable in the
        # same RAGService store the generator would have used.
        chunk_ids_in_store = {c.chunk_id for c in rag.chunks}
        self.assertIn("chunk_drift_1", chunk_ids_in_store)


class TestCheckpointingAndLogging(QualityGateTestBase):
    """I. Validation quota failure -> checkpoint and resume
       J. Rejected question is never inserted into questions
       K. Every validation attempt is logged
    """

    def test_I_quota_failure_checkpoints_prior_attempts(self):
        store = InMemoryValidationAttemptStore()
        approving_llm = FakeLLM(score=0.9, passed=True, uncertain=False)
        gate = self._gate(approving_llm)

        candidates = [
            make_question(question="Question one about drift detection."),
            make_question(question="Question two about drift detection."),
            make_question(question="Question three about drift detection."),
        ]

        quota_hit_on_third = FakeLLM(
            responses=None, score=0.9, passed=True, uncertain=False,
        )
        # Make the third call raise QuotaExhaustedException by swapping the LLM
        # mid-loop, simulating quota exhaustion partway through a batch.
        processed = 0
        quota_error = QuotaExhaustedException("gemini", "fake-model", "simulated 429")
        try:
            for i, cand in enumerate(candidates):
                if i == 2:
                    gate.primary_llm = FakeLLM(raise_exc=quota_error)
                result = asyncio.run(gate.validate_question(cand, [], expected_role="Machine Learning Engineer"))
                attempt = result.to_attempt_record(
                    role=cand.role, candidate_question_text=cand.question,
                    question_id=(i + 1) if result.approved else None,
                )
                store.record(attempt)
                processed += 1
        except QuotaExhaustedException:
            pass

        # Only the first 2 candidates got far enough to be recorded; the
        # QuotaExhaustedException on candidate 3 propagated instead of being
        # silently swallowed or recorded as a fabricated result.
        self.assertEqual(processed, 2)
        self.assertEqual(len(store.list_for_role("Machine Learning Engineer")), 2)

    def test_J_rejected_question_never_enters_approved_bank(self):
        llm = FakeLLM(score=0.99, passed=True, uncertain=False, critical_failures=["wrong_role"])
        gate = self._gate(llm)
        approved_question_bank = []  # mirrors QuestionPipelineRunner.question_bank

        candidate = make_question()
        result = asyncio.run(gate.validate_question(candidate, [], expected_role="Machine Learning Engineer"))

        if result.approved:
            approved_question_bank.append(candidate)

        self.assertFalse(result.approved)
        self.assertEqual(approved_question_bank, [], "A rejected candidate must never be appended to the question bank")

    def test_J2_store_refuses_to_persist_an_invalid_approved_row(self):
        """Defense in depth: even if calling code has a bug and tries to mark
        something approved+critical_failure, the store itself refuses --
        mirroring the DB's chk_qva_fail_closed CHECK constraint."""
        from question_pipeline.models import ValidationAttemptRecord

        store = InMemoryValidationAttemptStore()
        bad_record = ValidationAttemptRecord(
            question_id=1, role="Machine Learning Engineer",
            candidate_question_text="x", approved=True, quality_score=0.99,
            critical_failure=True, validation_tier="primary_gemini",
            validation_provider="fake_gemini", validation_model="fake-model",
        )
        with self.assertRaises(FailClosedViolation):
            store.record(bad_record)

        # And an approved=True row with no question_id is equally refused.
        bad_record_2 = ValidationAttemptRecord(
            question_id=None, role="Machine Learning Engineer",
            candidate_question_text="x", approved=True, quality_score=0.99,
            critical_failure=False, validation_tier="primary_gemini",
            validation_provider="fake_gemini", validation_model="fake-model",
        )
        with self.assertRaises(FailClosedViolation):
            store.record(bad_record_2)

    def test_K_every_attempt_approved_and_rejected_is_logged(self):
        store = InMemoryValidationAttemptStore()

        approving_llm = FakeLLM(score=0.9, passed=True, uncertain=False)
        rejecting_llm = FakeLLM(score=0.95, critical_failures=["technically_incorrect"])

        approve_gate = self._gate(approving_llm)
        reject_gate = self._gate(rejecting_llm)

        outcomes = []
        for i in range(3):
            cand = make_question(question=f"Approved candidate number {i} about drift monitoring dashboards.")
            result = asyncio.run(approve_gate.validate_question(cand, [], expected_role="Machine Learning Engineer"))
            store.record(result.to_attempt_record(role=cand.role, candidate_question_text=cand.question, question_id=i + 1))
            outcomes.append(result.approved)

        for i in range(2):
            cand = make_question(question=f"Rejected candidate number {i} about drift monitoring dashboards.")
            result = asyncio.run(reject_gate.validate_question(cand, [], expected_role="Machine Learning Engineer"))
            store.record(result.to_attempt_record(role=cand.role, candidate_question_text=cand.question, question_id=None))
            outcomes.append(result.approved)

        all_records = store.list_for_role("Machine Learning Engineer")
        self.assertEqual(len(all_records), 5, "Every attempt -- approved AND rejected -- must be logged")
        self.assertEqual(outcomes, [True, True, True, False, False])
        approved_count = sum(1 for r in all_records if r.approved)
        rejected_count = sum(1 for r in all_records if not r.approved)
        self.assertEqual(approved_count, 3)
        self.assertEqual(rejected_count, 2)


if __name__ == "__main__":
    unittest.main()
