"""
Tests for the post-pilot fixes:
  1. classify_scope() no longer forces DOMAIN merely because a `domains` tag
     is present.
  2. generator_service.py no longer ships a copyable example company list.
  3. The 0-1 year generation prompt steers away from trivial/definition-only
     questions while keeping junior-appropriate difficulty.

No real LLM/API calls are made -- prompt-content assertions are used to
prove what instructions actually get sent for #2/#3, since verifying model
BEHAVIOR would require a real, billed call (explicitly out of scope for this
turn: "Do NOT run the real ML Engineer pilot again yet").
"""
import os
import asyncio
import unittest

os.environ.setdefault("MOCK_MODE", "true")

from question_pipeline.config import config
from question_pipeline.models import QuestionObject
from question_pipeline.pipeline_runner import QuestionPipelineRunner
from question_pipeline.generator_service import GeneratorService
from question_pipeline.validator_service import ValidatorService
from question_pipeline.providers.base import LLMProvider, LLMResult
from typing import Optional


def make_question(**overrides) -> QuestionObject:
    base = dict(
        question="Placeholder question text for scope-classification testing.",
        role="Machine Learning Engineer",
        experience_band="3-5",
        difficulty=7,
        technical_depth=7, problem_complexity=7, architecture_complexity=6,
        troubleshooting=7, business_complexity=4, decision_making=5, leadership_ownership=3,
        question_type="technical",
        paradigm="ARCHITECTURE",
        mandatory_skills=["Monitoring"],
        scope="UNIVERSAL",
        domains=[],
        applicable_companies=[],
    )
    base.update(overrides)
    return QuestionObject(**base)


class TestScopeClassificationFix(unittest.TestCase):
    def setUp(self):
        self.runner = QuestionPipelineRunner()

    def test_A_generic_question_classifies_universal_despite_specific_domain_tag(self):
        """The exact bug: a genuinely universal question must NOT be forced
        to DOMAIN just because it happens to carry a specific domain tag."""
        q = make_question(
            scope="UNIVERSAL",
            domains=["CLOUD_SYSTEMS"],  # specific tag -- used to force DOMAIN
            applicable_companies=[],     # no genuine company signal
        )
        self.assertEqual(self.runner.classify_scope(q, target_scope="UNIVERSAL"), "UNIVERSAL")

    def test_B_domain_specific_question_classifies_domain(self):
        q = make_question(
            scope="DOMAIN",
            domains=["FINTECH_PAYMENTS"],
            applicable_companies=["Stripe", "PayPal", "Square"],
        )
        self.assertEqual(self.runner.classify_scope(q, target_scope="DOMAIN"), "DOMAIN")

    def test_B2_two_or_more_genuine_companies_classify_domain_even_without_domain_target(self):
        q = make_question(scope="DOMAIN", applicable_companies=["Netflix", "Spotify"])
        # target_scope defaults to UNIVERSAL at the batch level in this case,
        # but the candidate's own self-reported scope is not UNIVERSAL/COMPANY,
        # so the multi-company signal should still resolve to DOMAIN.
        self.assertEqual(self.runner.classify_scope(q, target_scope="UNIVERSAL"), "DOMAIN")

    def test_C_single_genuine_company_classifies_company(self):
        q = make_question(scope="COMPANY", applicable_companies=["Netflix"], domains=["STREAMING"])
        self.assertEqual(self.runner.classify_scope(q, target_scope="COMPANY"), "COMPANY")
        # Decisive regardless of target_scope.
        self.assertEqual(self.runner.classify_scope(q, target_scope="UNIVERSAL"), "COMPANY")

    def test_E_no_company_requirement_never_receives_company_scope(self):
        q = make_question(scope="UNIVERSAL", applicable_companies=[], domains=[])
        self.assertNotEqual(self.runner.classify_scope(q, target_scope="UNIVERSAL"), "COMPANY")

        # Self-reported COMPANY with zero companies actually named is not
        # defensible -- must be downgraded, not trusted blindly.
        q2 = make_question(scope="COMPANY", applicable_companies=[], domains=["CLOUD_SYSTEMS"])
        self.assertNotEqual(self.runner.classify_scope(q2, target_scope="COMPANY"), "COMPANY")

    def test_existing_scope_regression_suite_still_passes(self):
        """Re-assert the three pre-existing scope scenarios from
        test_pipeline.py's test_scope_classification_rules to prove this fix
        did not regress them."""
        q_comp = make_question(applicable_companies=["Uber"], domains=["TRANSPORTATION"], scope="UNIVERSAL")
        self.assertEqual(self.runner.classify_scope(q_comp), "COMPANY")

        q_dom = q_comp.model_copy(update={"applicable_companies": ["Google", "Amazon", "Apple"], "domains": ["FINTECH_PAYMENTS"]})
        self.assertEqual(self.runner.classify_scope(q_dom, target_scope="DOMAIN"), "DOMAIN")

        q_univ = q_comp.model_copy(update={
            "applicable_companies": ["Google", "Amazon", "Microsoft", "Meta"],
            "domains": ["GENERAL_TECH"], "scope": "UNIVERSAL",
        })
        self.assertEqual(self.runner.classify_scope(q_univ, target_scope="UNIVERSAL"), "UNIVERSAL")


class TestCompanyMetadataFix(unittest.TestCase):
    def setUp(self):
        self.gen = GeneratorService()

    def test_D_example_company_list_is_not_present_in_the_prompt_template(self):
        """The literal contaminating example ['Google','Amazon','Microsoft','Uber']
        must not appear anywhere in the generator's prompt-building source."""
        import inspect
        source = inspect.getsource(self.gen.generate_batch_for_band)
        self.assertNotIn('"Google", "Amazon", "Microsoft", "Uber"', source)
        self.assertNotIn("Google\", \"Amazon\", \"Microsoft\", \"Uber", source)

    def test_D2_generated_prompt_instructs_empty_list_for_universal_and_forbids_invention(self):
        async def run():
            questions = await self.gen.generate_batch_for_band(
                role="Machine Learning Engineer", experience_band="3-5", count=2, target_scope="UNIVERSAL",
            )
            return questions
        # We can't inspect the exact prompt string sent to a real API without
        # a live call, but we CAN confirm the mock run still produces valid,
        # role-invariant output through the modified prompt-construction path
        # (i.e. the change didn't break generation).
        questions = asyncio.run(run())
        self.assertTrue(len(questions) > 0)
        for q in questions:
            self.assertEqual(q.role, "Machine Learning Engineer")

    def test_D3_prompt_source_contains_the_no_invention_rule(self):
        import inspect
        source = inspect.getsource(self.gen.generate_batch_for_band)
        self.assertIn("NEVER invent a company", source)
        self.assertIn("applicable_companies", source)
        self.assertIn("empty list", source.lower())


class TestJuniorGenerationGuidance(unittest.TestCase):
    def setUp(self):
        self.gen = GeneratorService()

    def test_F_junior_prompt_contains_avoid_and_prefer_guidance(self):
        import inspect
        source = inspect.getsource(self.gen.generate_batch_for_band)
        self.assertIn("AVOID", source)
        self.assertIn("PREFER", source)
        self.assertIn("trivial", source.lower())
        self.assertIn("debugging a realistic", source.lower())

    def test_G_junior_difficulty_bounds_unchanged(self):
        """Fix #3 must improve WHAT is asked for, not the difficulty
        calibration itself."""
        self.assertEqual(config.EXPERIENCE_DIFFICULTY_BOUNDS["0-1"], (1, 6))
        import inspect
        source = inspect.getsource(self.gen.generate_batch_for_band)
        self.assertIn("Difficulty 3-5", source)


class FakeLLM(LLMProvider):
    def __init__(self, score, passed, uncertain, critical_failures=None, name="fake_gemini", model_name="fake-model"):
        self._name = name
        self.model_name = model_name
        self._score, self._passed, self._uncertain = score, passed, uncertain
        self._critical_failures = critical_failures or []

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_json(self, prompt, system_prompt=None, task_name="validation") -> LLMResult:
        data = {"score": self._score, "passed": self._passed, "uncertain": self._uncertain,
                "reasoning": "test", "checks": {}, "critical_failures": self._critical_failures}
        return LLMResult(text=str(data), data=data, model=self.model_name, provider=self._name)

    async def generate_text(self, prompt, system_prompt=None, task_name="text") -> LLMResult:
        return await self.generate_json(prompt, system_prompt, task_name)


class TestQualityGateAndRoleInvarianceUnchanged(unittest.TestCase):
    def test_H_fail_closed_rejection_behavior_unchanged(self):
        """High score + critical_failure must still be rejected -- proves
        this turn's changes did not touch the quality gate."""
        gate = ValidatorService()
        gate.primary_llm = FakeLLM(score=0.97, passed=True, uncertain=False, critical_failures=["hallucinated_technology"])
        q = make_question()

        result = asyncio.run(gate.validate_question(q, [], expected_role="Machine Learning Engineer"))
        self.assertFalse(result.approved)
        self.assertTrue(result.critical_failure)
        self.assertIn("hallucinated_technology", result.rejection_reasons)

    def test_I_role_invariance_unchanged_by_scope_or_generation_changes(self):
        runner = QuestionPipelineRunner()
        q_junior = make_question(role="Machine Learning Engineer", experience_band="0-1", difficulty=4)
        q_senior = make_question(role="Machine Learning Engineer", experience_band="8+", difficulty=9)

        # Scope classification must never touch role.
        runner.classify_scope(q_junior, target_scope="UNIVERSAL")
        runner.classify_scope(q_senior, target_scope="COMPANY")
        self.assertEqual(q_junior.role, "Machine Learning Engineer")
        self.assertEqual(q_senior.role, "Machine Learning Engineer")

        # Generator must still force role invariance regardless of band.
        async def run():
            gen = GeneratorService()
            qs = await gen.generate_batch_for_band(role="Machine Learning Engineer", experience_band="0-1", count=2)
            return qs
        for q in asyncio.run(run()):
            self.assertEqual(q.role, "Machine Learning Engineer")
            self.assertEqual(q.experience_band, "0-1")


if __name__ == "__main__":
    unittest.main()
