"""
Regression tests for the cost-tracker pricing gap found during the real
Phase 5A rollout: config.LLM_MODEL ("gemini-3.1-flash-lite") had no entry
in config.PRICING["gemini"], so every real generation/synthesis/Tier-1-
validation call silently recorded $0.00 instead of failing loudly or using
a verified rate.

Fix: config.get_pricing(provider, model) is now the single, centralized,
model-name-based pricing lookup (config.PRICING itself is unchanged in
shape). It returns a real $0.00 in MOCK_MODE (genuinely free -- no API is
ever reached), and raises PricingNotConfiguredError outside MOCK_MODE for
any provider/model without a verified entry, rather than fabricating a
number. cost_tracker.CostTracker.record_usage() now calls it instead of
doing its own silent dict-lookup-with-default.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MOCK_MODE", "true")

from question_pipeline.config import config, PricingNotConfiguredError
from question_pipeline.cost_tracker import CostTracker


class TestConfiguredModelHasRealPricing(unittest.TestCase):
    """The literal regression test: whatever model is ACTUALLY configured
    for real use must have a verified pricing entry. If someone changes
    LLM_MODEL/EMBEDDING_MODEL/FALLBACK_LLM_MODEL/ESCALATION_MODEL without
    updating PRICING, this test fails immediately instead of the gap being
    discovered only after a real, unaccounted-for production run."""

    def test_configured_llm_model_has_pricing(self):
        pricing = config.PRICING.get(config.LLM_PROVIDER, {}).get(config.LLM_MODEL)
        self.assertIsNotNone(
            pricing,
            f"config.LLM_MODEL={config.LLM_MODEL!r} (provider={config.LLM_PROVIDER!r}) has no "
            f"PRICING entry -- real generation/synthesis/validation calls would be recorded as $0.00.",
        )
        self.assertGreater(pricing["input"] + pricing["output"], 0.0, "a real LLM model must not be priced at exactly $0.00")

    def test_configured_embedding_model_has_pricing(self):
        if config.EMBEDDING_PROVIDER == "local":
            self.skipTest("local embedding provider is genuinely free -- no pricing entry needed")
        pricing = config.PRICING.get(config.EMBEDDING_PROVIDER, {}).get(config.EMBEDDING_MODEL)
        self.assertIsNotNone(
            pricing,
            f"config.EMBEDDING_MODEL={config.EMBEDDING_MODEL!r} (provider={config.EMBEDDING_PROVIDER!r}) "
            f"has no PRICING entry.",
        )

    def test_configured_fallback_llm_model_has_pricing(self):
        pricing = config.PRICING.get(config.FALLBACK_LLM_PROVIDER, {}).get(config.FALLBACK_LLM_MODEL)
        self.assertIsNotNone(
            pricing,
            f"config.FALLBACK_LLM_MODEL={config.FALLBACK_LLM_MODEL!r} (provider={config.FALLBACK_LLM_PROVIDER!r}) "
            f"has no PRICING entry.",
        )

    def test_configured_escalation_model_has_pricing(self):
        pricing = config.PRICING.get(config.ESCALATION_PROVIDER, {}).get(config.ESCALATION_MODEL)
        self.assertIsNotNone(
            pricing,
            f"config.ESCALATION_MODEL={config.ESCALATION_MODEL!r} (provider={config.ESCALATION_PROVIDER!r}) "
            f"has no PRICING entry.",
        )

    def test_gemini_3_1_flash_lite_specifically_has_a_nonzero_verified_rate(self):
        """The exact model that triggered this bug (see Phase 5A report):
        pinned so this can never silently regress back to missing/zero."""
        pricing = config.PRICING["gemini"]["gemini-3.1-flash-lite"]
        self.assertEqual(pricing, {"input": 0.25, "output": 1.50})

    def test_gemini_embedding_2_is_the_corrected_020_rate_not_the_old_002(self):
        """Phase 5B follow-up bug: the original $0.02 entry was wrong by
        10x. Corrected to $0.20/1M input tokens, verified against the
        official Gemini Developer API pricing page
        (ai.google.dev/gemini-api/docs/pricing), standard tier, text input.
        Pinned so this can never silently regress back to the wrong value."""
        pricing = config.PRICING["gemini"]["gemini-embedding-2"]
        self.assertEqual(pricing, {"input": 0.20, "output": 0.0})
        self.assertNotEqual(pricing["input"], 0.02, "must not regress to the old, incorrect 10x-too-low rate")


class TestGetPricingFailsClosed(unittest.TestCase):
    """config.get_pricing() must never fabricate a price for a real call.

    Deliberately mode-independent (does NOT branch on MOCK_MODE) -- pricing
    math must be verifiable regardless of which mode the test process
    happens to be running under, which is also exactly what broke the
    pre-existing test_pipeline.py cost-calculation tests when an earlier
    version of this fix short-circuited to $0.00 for ANY call made while
    the process-wide MOCK_MODE flag was true, not just for genuinely mock
    model names."""

    def test_known_model_returns_its_real_entry(self):
        pricing = config.get_pricing("gemini", "gemini-3.1-flash-lite")
        self.assertEqual(pricing, {"input": 0.25, "output": 1.50})

    def test_unknown_model_raises_instead_of_defaulting_to_zero(self):
        with self.assertRaises(PricingNotConfiguredError):
            config.get_pricing("gemini", "some-future-model-not-yet-priced")

    def test_unknown_provider_raises(self):
        with self.assertRaises(PricingNotConfiguredError):
            config.get_pricing("some-new-provider", "any-model")

    def test_mock_model_name_is_explicitly_priced_at_zero_not_a_fallback_default(self):
        """MockLLMProvider always reports model="mock-model" -- genuinely
        free (MOCK_MODE never reaches a real API), so this must resolve via
        a real, explicit PRICING entry, not a missing-entry fallback."""
        for provider in ("gemini", "openai", "anthropic"):
            with self.subTest(provider=provider):
                pricing = config.get_pricing(provider, "mock-model")
                self.assertEqual(pricing, {"input": 0.0, "output": 0.0})


class TestCostTrackerUsesCentralizedPricing(unittest.TestCase):
    def setUp(self):
        self.tracker = CostTracker()
        self._records_backup = list(self.tracker._records)
        self.tracker._records = []

    def tearDown(self):
        self.tracker._records = self._records_backup

    def test_record_usage_raises_for_unpriced_real_model_instead_of_zero_cost(self):
        with self.assertRaises(PricingNotConfiguredError):
            self.tracker.record_usage(
                provider="gemini", model="totally-unpriced-model",
                task="generation", input_tokens=1000, output_tokens=500,
            )
        self.assertEqual(len(self.tracker._records), 0, "an unpriced real call must not be recorded at all, not even at $0.00")

    def test_record_usage_correctly_prices_the_configured_llm_model(self):
        rec = self.tracker.record_usage(
            provider="gemini", model="gemini-3.1-flash-lite",
            task="generation", input_tokens=1_000_000, output_tokens=1_000_000,
        )
        self.assertAlmostEqual(rec.estimated_cost, 0.25 + 1.50, places=6)

    def test_get_summary_usage_by_model_reports_model_input_output_calls_cost(self):
        self.tracker.record_usage(provider="gemini", model="gemini-3.1-flash-lite",
                                   task="generation", input_tokens=100_000, output_tokens=50_000)
        self.tracker.record_usage(provider="gemini", model="gemini-3.1-flash-lite",
                                   task="synthesis", input_tokens=200_000, output_tokens=100_000)
        self.tracker.record_usage(provider="gemini", model="gemini-embedding-2",
                                   task="rag_embedding", input_tokens=5_000, output_tokens=0)

        summary = self.tracker.get_summary()
        by_model = {(e["provider"], e["model"]): e for e in summary["usage_by_model"]}

        flash_lite = by_model[("gemini", "gemini-3.1-flash-lite")]
        self.assertEqual(flash_lite["calls"], 2)
        self.assertEqual(flash_lite["input_tokens"], 300_000)
        self.assertEqual(flash_lite["output_tokens"], 150_000)
        expected_cost = round(300_000 / 1_000_000 * 0.25 + 150_000 / 1_000_000 * 1.50, 6)
        self.assertAlmostEqual(flash_lite["estimated_cost"], expected_cost, places=6)

        embedding = by_model[("gemini", "gemini-embedding-2")]
        self.assertEqual(embedding["calls"], 1)
        self.assertEqual(embedding["input_tokens"], 5_000)

    def test_get_summary_usage_by_task_distinguishes_categories(self):
        """Requirement: the cost report must distinguish research/synthesis/
        generation/validation model usage from RAG-embedding, question-
        bank-embedding, and deduplication-embedding usage. Each already has
        its own task_name at the call site (generator_service.py="generation",
        synthesis_service.py="synthesis", validator_service.py=
        "validation_primary_gemini" etc., rag_service.py="rag_embedding"/
        "query_embedding", db_loader.py="question_bank_embedding",
        deduplicator_service.py="dedup_embedding") -- this proves the
        summary actually separates them out with full token/call/cost detail,
        not just a combined per-provider or per-model total."""
        self.tracker.record_usage(provider="gemini", model="gemini-3.1-flash-lite",
                                   task="generation", input_tokens=10_000, output_tokens=5_000)
        self.tracker.record_usage(provider="gemini", model="gemini-embedding-2",
                                   task="rag_embedding", input_tokens=1_000, output_tokens=0)
        self.tracker.record_usage(provider="gemini", model="gemini-embedding-2",
                                   task="question_bank_embedding", input_tokens=2_000, output_tokens=0)
        self.tracker.record_usage(provider="gemini", model="gemini-embedding-2",
                                   task="dedup_embedding", input_tokens=3_000, output_tokens=0)

        summary = self.tracker.get_summary()
        by_task = {e["task"]: e for e in summary["usage_by_task"]}

        self.assertEqual(set(by_task.keys()), {"generation", "rag_embedding", "question_bank_embedding", "dedup_embedding"})
        self.assertEqual(by_task["generation"]["input_tokens"], 10_000)
        self.assertEqual(by_task["rag_embedding"]["input_tokens"], 1_000)
        self.assertEqual(by_task["question_bank_embedding"]["input_tokens"], 2_000)
        self.assertEqual(by_task["dedup_embedding"]["input_tokens"], 3_000)
        # RAG/question-bank/dedup embedding usage must be separately
        # identifiable, not merged into one "embedding" bucket.
        self.assertNotEqual(by_task["rag_embedding"], by_task["question_bank_embedding"])
        self.assertNotEqual(by_task["question_bank_embedding"], by_task["dedup_embedding"])

    def test_existing_summary_keys_are_preserved(self):
        """Backward compatibility: pipeline_runner._build_pilot_report()
        reads cost_by_provider/cost_by_task/usage_by_provider directly --
        the new usage_by_model key must be additive, not a replacement."""
        with patch.object(config, "MOCK_MODE", True):
            self.tracker.record_usage(provider="gemini", model="mock-model",
                                       task="generation", input_tokens=10, output_tokens=10)
        summary = self.tracker.get_summary()
        for key in ("total_estimated_cost_usd", "cost_by_provider", "cost_by_task",
                    "usage_by_provider", "usage_by_model", "usage_by_task", "total_calls"):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
