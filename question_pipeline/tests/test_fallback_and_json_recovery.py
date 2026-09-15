"""
Tests for the post-second-pilot fixes:
  1. OpenAI (Tier 2 fallback) configuration: provider/model are genuinely
     environment-configurable, credential validation is consistent with the
     rest of the codebase (shared is_valid_key, not a narrower ad-hoc
     check), and fail-closed holds when the fallback is genuinely
     unavailable -- exactly the "escalation_unavailable_uncertain_result"
     behavior observed live in both real pilots.
  2. Malformed Gemini JSON recovery: robust extraction (handles the actual
     "Extra data" failure seen live), a single bounded retry in
     generator_service, and that a batch which still fails after the retry
     is treated as failed/partial without crashing the rest of the pipeline
     run or losing already-checkpointed progress.

No real network calls are made anywhere in this file.
"""
import os
import json
import asyncio
import unittest
from unittest.mock import patch
from typing import Optional

os.environ.setdefault("MOCK_MODE", "true")

from question_pipeline.config import config
from question_pipeline.models import QuestionObject
from question_pipeline.providers.base import LLMProvider, LLMResult, extract_json_object
from question_pipeline.providers.factory import get_llm_provider
from question_pipeline.providers.openai_provider import OpenAILLMProvider
from question_pipeline.governor import QuotaExhaustedException, ProviderBlockedException
from question_pipeline.validator_service import ValidatorService
from question_pipeline.generator_service import GeneratorService
from question_pipeline.pipeline_runner import QuestionPipelineRunner


def make_question(**overrides) -> QuestionObject:
    base = dict(
        question="Placeholder question for fallback/JSON-recovery testing.",
        role="Machine Learning Engineer", experience_band="3-5", difficulty=7,
        technical_depth=7, problem_complexity=7, architecture_complexity=6,
        troubleshooting=7, business_complexity=4, decision_making=5, leadership_ownership=3,
        question_type="technical", paradigm="ARCHITECTURE",
        mandatory_skills=["Monitoring"], scope="UNIVERSAL", domains=[], applicable_companies=[],
    )
    base.update(overrides)
    return QuestionObject(**base)


# ---------------------------------------------------------------------------
# 1. OpenAI fallback configuration
# ---------------------------------------------------------------------------

class TestOpenAIFallbackConfiguration(unittest.TestCase):
    def test_fallback_model_is_environment_configurable(self):
        original = config.FALLBACK_LLM_MODEL
        try:
            config.FALLBACK_LLM_MODEL = "gpt-test-model-xyz"
            provider = OpenAILLMProvider()
            self.assertEqual(provider.model_name, "gpt-test-model-xyz")
            # Explicit override still wins over config.
            provider2 = OpenAILLMProvider(model_name="explicit-override")
            self.assertEqual(provider2.model_name, "explicit-override")
        finally:
            config.FALLBACK_LLM_MODEL = original

    def test_fallback_provider_selection_is_environment_configurable(self):
        original_provider = config.FALLBACK_LLM_PROVIDER
        original_mock = config.MOCK_MODE
        try:
            config.MOCK_MODE = False
            config.FALLBACK_LLM_PROVIDER = "openai"
            self.assertIsInstance(get_llm_provider(config.FALLBACK_LLM_PROVIDER), OpenAILLMProvider)

            config.FALLBACK_LLM_PROVIDER = "anthropic"
            from question_pipeline.providers.claude_provider import ClaudeLLMProvider
            self.assertIsInstance(get_llm_provider(config.FALLBACK_LLM_PROVIDER), ClaudeLLMProvider)
        finally:
            config.FALLBACK_LLM_PROVIDER = original_provider
            config.MOCK_MODE = original_mock

    def test_credential_check_uses_shared_placeholder_detection(self):
        """The old check only caught a literal 'dummy' substring. A
        plausible-but-placeholder key like 'sk-your_key_here' must now be
        caught too, via the same is_valid_key used everywhere else."""
        original = config.OPENAI_API_KEY
        try:
            config.OPENAI_API_KEY = "sk-your_key_here_1234"
            provider = OpenAILLMProvider()
            with self.assertRaises(ProviderBlockedException):
                provider._ensure_api_key()
        finally:
            config.OPENAI_API_KEY = original

    def test_valid_looking_key_passes_the_precheck(self):
        original = config.OPENAI_API_KEY
        try:
            config.OPENAI_API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
            provider = OpenAILLMProvider()
            provider._ensure_api_key()  # must not raise
        finally:
            config.OPENAI_API_KEY = original


class FakeLLM(LLMProvider):
    def __init__(self, score=0.9, passed=True, uncertain=False, critical_failures=None,
                 name="fake", model_name="fake-model", raise_exc: Optional[Exception] = None):
        self._name, self.model_name = name, model_name
        self._score, self._passed, self._uncertain = score, passed, uncertain
        self._critical_failures = critical_failures or []
        self._raise_exc = raise_exc

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_json(self, prompt, system_prompt=None, task_name="validation") -> LLMResult:
        if self._raise_exc:
            raise self._raise_exc
        data = {"score": self._score, "passed": self._passed, "uncertain": self._uncertain,
                "reasoning": "test", "checks": {}, "critical_failures": self._critical_failures}
        return LLMResult(text=str(data), data=data, model=self.model_name, provider=self._name)

    async def generate_text(self, prompt, system_prompt=None, task_name="text") -> LLMResult:
        return await self.generate_json(prompt, system_prompt, task_name)


class TestFallbackUnavailableFailsClosed(unittest.TestCase):
    """Reproduces, as a fast unit test, exactly the
    'escalation_unavailable_uncertain_result' behavior observed live in both
    real pilots: Tier 1 uncertain -> Tier 2 genuinely unreachable -> reject,
    never approve on the unconfirmed Tier 1 score."""

    def test_uncertain_primary_with_blocked_fallback_is_rejected_not_approved(self):
        gate = ValidatorService()
        gate.primary_llm = FakeLLM(score=0.72, passed=True, uncertain=True)

        blocked_exc = ProviderBlockedException("openai", "gpt-4o-mini", "OPENAI_API_KEY is not configured.")
        # Patch the factory (not config/MOCK_MODE) so this exercises the real
        # escalation code path -- get_llm_provider() itself returns
        # MockLLMProvider whenever MOCK_MODE is true, which would silently
        # bypass the exact real-world failure being reproduced here.
        with patch("question_pipeline.validator_service.get_llm_provider") as mock_factory:
            mock_factory.return_value = FakeLLM(raise_exc=blocked_exc)
            result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertFalse(result.approved, "An uncertain Tier 1 score must never be approved when escalation is unreachable")
        self.assertIn("escalation_unavailable_uncertain_result", result.rejection_reasons)

    def test_uncertain_primary_with_working_fallback_still_escalates_and_can_approve(self):
        """Sanity check the other side: when Tier 2 IS reachable and confident, it should resolve normally."""
        gate = ValidatorService()
        gate.primary_llm = FakeLLM(score=0.72, passed=True, uncertain=True)

        with patch("question_pipeline.validator_service.get_llm_provider") as mock_factory:
            mock_factory.return_value = FakeLLM(score=0.9, passed=True, uncertain=False)
            result = asyncio.run(gate.validate_question(make_question(), [], expected_role="Machine Learning Engineer"))

        self.assertTrue(result.approved)
        self.assertEqual(result.validation_tier, "fallback_gpt")


# ---------------------------------------------------------------------------
# 2. Malformed JSON recovery
# ---------------------------------------------------------------------------

class TestRobustJsonExtraction(unittest.TestCase):
    def test_clean_json_parses_normally(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_markdown_fenced_json_is_stripped(self):
        text = "```json\n{\"a\": 1}\n```"
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_extra_trailing_data_is_recovered(self):
        """The EXACT failure class observed live: 'Extra data: line N column 1'."""
        text = '{"questions": [{"question": "real content"}]}\n\nSome trailing commentary the model appended.'
        result = extract_json_object(text)
        self.assertEqual(result, {"questions": [{"question": "real content"}]})

    def test_duplicated_json_object_uses_only_the_first(self):
        text = '{"a": 1}\n{"a": 2}'
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_genuinely_invalid_json_still_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            extract_json_object("this is not json at all { [ broken")

    def test_extraction_never_alters_field_content(self):
        """Must extract verbatim -- never repair/rewrite field values."""
        text = '{"question": "Explain gradient descent.", "difficulty": 5}\ntrailing junk'
        result = extract_json_object(text)
        self.assertEqual(result["question"], "Explain gradient descent.")
        self.assertEqual(result["difficulty"], 5)


class TestGeneratorBoundedRetry(unittest.TestCase):
    def setUp(self):
        self.gen = GeneratorService()

    def test_single_retry_recovers_from_one_malformed_response(self):
        calls = {"count": 0}

        class FlakyOnceLLM(LLMProvider):
            @property
            def provider_name(self):
                return "flaky"

            async def generate_json(self, prompt, system_prompt=None, task_name="generation"):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise json.JSONDecodeError("Extra data", "bad json", 10)
                return LLMResult(text="{}", data={"questions": []}, model="flaky-model", provider="flaky")

            async def generate_text(self, prompt, system_prompt=None, task_name="text"):
                return await self.generate_json(prompt, system_prompt, task_name)

        self.gen.llm = FlakyOnceLLM()
        result = asyncio.run(self.gen._generate_json_with_bounded_retry("prompt", "Machine Learning Engineer", "3-5"))
        self.assertEqual(calls["count"], 2, "Exactly one retry should have occurred")
        self.assertEqual(result.data, {"questions": []})

    def test_retry_is_bounded_to_exactly_one_attempt(self):
        calls = {"count": 0}

        class AlwaysFlakyLLM(LLMProvider):
            @property
            def provider_name(self):
                return "flaky"

            async def generate_json(self, prompt, system_prompt=None, task_name="generation"):
                calls["count"] += 1
                raise json.JSONDecodeError("Extra data", "bad json", 10)

            async def generate_text(self, prompt, system_prompt=None, task_name="text"):
                return await self.generate_json(prompt, system_prompt, task_name)

        self.gen.llm = AlwaysFlakyLLM()
        with self.assertRaises(json.JSONDecodeError):
            asyncio.run(self.gen._generate_json_with_bounded_retry("prompt", "Machine Learning Engineer", "3-5"))
        self.assertEqual(calls["count"], 2, "Must attempt exactly twice total (1 initial + 1 retry), never more")

    def test_quota_exception_during_retry_propagates_without_a_third_attempt(self):
        calls = {"count": 0}
        quota_exc = QuotaExhaustedException("gemini", "fake-model", "simulated 429")

        class FlakyThenQuotaLLM(LLMProvider):
            @property
            def provider_name(self):
                return "flaky"

            async def generate_json(self, prompt, system_prompt=None, task_name="generation"):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise json.JSONDecodeError("Extra data", "bad json", 10)
                raise quota_exc

            async def generate_text(self, prompt, system_prompt=None, task_name="text"):
                return await self.generate_json(prompt, system_prompt, task_name)

        self.gen.llm = FlakyThenQuotaLLM()
        with self.assertRaises(QuotaExhaustedException):
            asyncio.run(self.gen._generate_json_with_bounded_retry("prompt", "Machine Learning Engineer", "3-5"))
        self.assertEqual(calls["count"], 2)


class TestBatchFailurePartialContinuation(unittest.TestCase):
    """Confirms a batch that fails after the bounded retry is treated as
    failed/partial and does NOT block the rest of the role's pipeline run --
    other bands still get processed and checkpointed."""

    def test_one_failed_band_does_not_block_the_others(self):
        import question_pipeline.pipeline_runner as pr_module

        runner = QuestionPipelineRunner()
        role = "Batch Failure Test Role"

        real_generate = pr_module.generator_service.generate_batch_for_band

        async def flaky_for_one_band(role, experience_band, count=4, target_scope="UNIVERSAL", domain=None):
            if experience_band == "3-5":
                raise json.JSONDecodeError("Extra data", "bad json after retry exhausted", 10)
            return await real_generate(role, experience_band, count, target_scope, domain)

        pr_module.generator_service.generate_batch_for_band = flaky_for_one_band
        try:
            result = asyncio.run(runner.run_role_pipeline(role, target_questions=20, force=True))
        finally:
            pr_module.generator_service.generate_batch_for_band = real_generate

        # The run must complete (not crash / not return early on the first error)
        # and bands other than the failed one must have produced candidates.
        self.assertIn(result["status"], ("COMPLETED", "PARTIAL"))
        self.assertGreater(result["metrics"]["generated"], 0, "Other bands must still have generated candidates")
        self.assertTrue(
            any("Extra data" in e or "bad json" in e for e in result["metrics"]["errors"]),
            "The failed band's error must be recorded, not silently swallowed",
        )


if __name__ == "__main__":
    unittest.main()
