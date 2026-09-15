import asyncio
import os
import unittest
import json
import shutil
from pathlib import Path

# Ensure MOCK_MODE is enabled for isolated unit tests
os.environ["MOCK_MODE"] = "true"

from question_pipeline.config import config
from question_pipeline.models import (
    QuestionObject,
    ResearchOutput,
    KnowledgeChunk,
    ValidationResult,
    CostRecord,
    RolePipelineState
)
from question_pipeline.cost_tracker import CostTracker
from question_pipeline.governor import RateLimitGovernor, QuotaExhaustedException, ProviderBlockedException
from question_pipeline.state_manager import StateManager
from question_pipeline.deduplicator_service import DeduplicatorService, normalize_text, token_jaccard_similarity
from question_pipeline.rag_service import RAGService
from question_pipeline.providers.local_provider import LocalEmbeddingProvider
from question_pipeline.providers.factory import get_research_provider, get_llm_provider, get_embedding_provider
from question_pipeline.pipeline_runner import QuestionPipelineRunner

class TestQuestionPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "temp_test_data"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_question_object_schema_and_role_invariance(self):
        """Test QuestionObject conforms to user specification and enforces 7 depth dimensions."""
        q = QuestionObject(
            question="How would you design a distributed feature store for real-time inference?",
            role="Machine Learning Engineer",
            experience_band="3-5",
            difficulty=7,
            technical_depth=7,
            problem_complexity=7,
            architecture_complexity=8,
            troubleshooting=6,
            business_complexity=5,
            decision_making=7,
            leadership_ownership=4,
            question_type="system_design",
            paradigm="ARCHITECTURE",
            mandatory_skills=["Redis", "Feature Store", "Distributed Systems"],
            scope="UNIVERSAL",
            domains=["CLOUD_PLATFORMS"],
            applicable_companies=["Google", "Amazon", "Meta"]
        )
        self.assertEqual(q.role, "Machine Learning Engineer")
        self.assertEqual(q.paradigm, "ARCHITECTURE")
        self.assertEqual(q.scope, "UNIVERSAL")
        self.assertEqual(q.technical_depth, 7)
        self.assertEqual(q.difficulty, 7)

    def test_cost_tracker_calculation(self):
        """Test exact cost calculations across providers and tasks."""
        tracker = CostTracker()
        tracker.clear()

        # 1M tokens of Gemini Flash-Lite input = $0.075, output = $0.30
        rec1 = tracker.record_usage(
            provider="gemini",
            model="gemini-2.5-flash-lite",
            task="generation",
            input_tokens=100_000,
            output_tokens=50_000
        )
        expected_cost = ((100_000 / 1_000_000) * 0.075) + ((50_000 / 1_000_000) * 0.30)
        self.assertAlmostEqual(rec1.estimated_cost, expected_cost, places=5)

        summary = tracker.get_summary()
        self.assertIn("gemini", summary["cost_by_provider"])
        self.assertIn("generation", summary["cost_by_task"])

    def test_governor_quota_and_blocked_exceptions(self):
        """Test that rate limits and missing credentials trigger proper typed exceptions."""
        gov = RateLimitGovernor()

        # Test simulated 429
        async def failing_func():
            raise Exception("Resource_Exhausted: 429 Too Many Requests")

        async def run_gov():
            with self.assertRaises(QuotaExhaustedException):
                await gov.execute_with_retry("gemini", "gemini-2.5-flash-lite", failing_func, max_retries=1)

        asyncio.run(run_gov())

        # Test simulated missing API key
        async def auth_fail_func():
            raise Exception("API_KEY not set or invalid 401 Unauthorized")

        async def run_auth_fail():
            with self.assertRaises(ProviderBlockedException):
                await gov.execute_with_retry("gemini", "gemini-2.5-flash-lite", auth_fail_func, max_retries=1)

        asyncio.run(run_auth_fail())

    def test_state_manager_resumability_and_cadence(self):
        """Test state transitions, batch checkpointing, and research cadence."""
        state_file = self.test_dir / "test_state.json"
        sm = StateManager(state_file=state_file)

        role = "Data Engineer"
        # Initial state
        s0 = sm.get_role_state(role)
        self.assertEqual(s0.status, "PENDING")

        # Update batch progress
        sm.update_role_state(role, status="RUNNING", current_batch=2, total_batches=5)
        s1 = sm.get_role_state(role)
        self.assertEqual(s1.current_batch, 2)

        # Transition to WAITING_FOR_QUOTA
        sm.mark_waiting_for_quota(role, "Rate limit encountered on batch 3")
        s2 = sm.get_role_state(role)
        self.assertEqual(s2.status, "WAITING_FOR_QUOTA")
        self.assertIn("Rate limit", s2.reason)

        # Transition to BLOCKED
        sm.mark_blocked(role, "API key missing")
        s3 = sm.get_role_state(role)
        self.assertEqual(s3.status, "BLOCKED")

    def test_deduplicator_service_3_tiers(self):
        """Test 3-tier deduplication: exact, normalized, and semantic."""
        async def run_dedup():
            dedup = DeduplicatorService()

            q1 = QuestionObject(
                question="How would you design a distributed feature store for real-time inference?",
                role="Machine Learning Engineer",
                experience_band="3-5",
                difficulty=7,
                technical_depth=7,
                problem_complexity=7,
                architecture_complexity=8,
                troubleshooting=6,
                business_complexity=5,
                decision_making=7,
                leadership_ownership=4,
                question_type="system_design",
                paradigm="ARCHITECTURE",
                mandatory_skills=["Redis"],
                scope="UNIVERSAL",
                domains=["CLOUD"],
                applicable_companies=["Google"]
            )
            bank = [q1]

            # 1. Exact duplicate
            q_exact = q1.model_copy(update={"question": "  HOW would you design a distributed feature store for real-time inference?  "})
            is_dup, reason = await dedup.check_duplicate(q_exact, bank)
            self.assertTrue(is_dup)
            self.assertIn("EXACT_DUPLICATE", reason)

            # 2. Normalized duplicate
            q_norm = q1.model_copy(update={"question": "How do you design a distributed feature store for real time inference?"})
            is_dup, reason = await dedup.check_duplicate(q_norm, bank)
            self.assertTrue(is_dup)

            # 3. Novel distinct question
            q_distinct = q1.model_copy(update={"question": "Explain the difference between optimistic and pessimistic locking in PostgreSQL transactions."})
            is_dup, reason = await dedup.check_duplicate(q_distinct, bank)
            self.assertFalse(is_dup)

        asyncio.run(run_dedup())

    def test_rag_indexing_and_cosine_retrieval(self):
        """Test knowledge chunk vector indexing and cosine retrieval."""
        async def run_rag():
            store_file = self.test_dir / "test_vector_store.json"
            rag = RAGService(store_file=store_file)

            chunks = [
                KnowledgeChunk(
                    chunk_id="chunk_1",
                    role="Data Engineer",
                    topic="Stream Processing",
                    text="Apache Flink and Spark Streaming state management under checkpointing.",
                    knowledge_version="v1.0"
                ),
                KnowledgeChunk(
                    chunk_id="chunk_2",
                    role="Data Engineer",
                    topic="Data Warehousing",
                    text="Snowflake and BigQuery partitioning, clustering, and cost optimization.",
                    knowledge_version="v1.0"
                )
            ]

            await rag.index_chunks(chunks)
            self.assertEqual(len(rag.chunks), 2)

            # Retrieve
            retrieved = await rag.retrieve_context("Data Engineer", "3-5", "ARCHITECTURE", top_k=1)
            self.assertTrue(len(retrieved) > 0)
            self.assertEqual(retrieved[0].role, "Data Engineer")

        asyncio.run(run_rag())

    def test_embedding_model_invalidation_on_change(self):
        """Test vector store invalidates incompatible embeddings when embedding model changes."""
        store_file = self.test_dir / "test_versioned_vector_store.json"
        
        # 1. Store with model-v1
        store_data_v1 = {
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "chunks": [
                {
                    "chunk_id": "c1",
                    "role": "Data Engineer",
                    "topic": "Storage",
                    "text": "Parquet and Delta Lake storage.",
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_model": "gemini-embedding-2",
                    "knowledge_version": "v1.0"
                }
            ]
        }
        with open(store_file, "w", encoding="utf-8") as f:
            json.dump(store_data_v1, f, indent=2)

        # 2. Simulate switching embedding model to a new model
        original_model = config.EMBEDDING_MODEL
        try:
            config.EMBEDDING_MODEL = "new-experimental-embedding-model"
            rag = RAGService(store_file=store_file)
            
            # The RAG service must detect the model mismatch and invalidate/clear the store
            self.assertEqual(len(rag.chunks), 0)
        finally:
            config.EMBEDDING_MODEL = original_model

    def test_cost_tracker_model_prefix_pricing(self):
        """Verify model names with 'models/' prefix still match pricing table correctly."""
        tracker = CostTracker()
        tracker.clear()
        rec = tracker.record_usage(
            provider="gemini",
            model="models/gemini-2.5-flash-lite",
            task="generation",
            input_tokens=100_000,
            output_tokens=50_000
        )
        self.assertGreater(rec.estimated_cost, 0.0)
        expected = ((100_000 / 1_000_000) * 0.075) + ((50_000 / 1_000_000) * 0.30)
        self.assertAlmostEqual(rec.estimated_cost, expected, places=5)

    def test_validation_calibration_and_error_handling(self):
        """Verify uncalibrated difficulty is rejected (fail-closed) regardless of the LLM's own score."""
        from question_pipeline.validator_service import validator_service

        # 1. Uncalibrated question (0-1 yrs with difficulty 10)
        q_bad = QuestionObject(
            question="Extreme multi-datacenter consensus protocol failure recovery.",
            role="Machine Learning Engineer",
            experience_band="0-1",
            difficulty=10,
            technical_depth=10,
            problem_complexity=10,
            architecture_complexity=10,
            troubleshooting=10,
            business_complexity=10,
            decision_making=10,
            leadership_ownership=10,
            question_type="system_design",
            paradigm="EXECUTION",
            mandatory_skills=["Raft"],
            scope="UNIVERSAL",
            domains=["CLOUD"],
            applicable_companies=["Google"]
        )

        async def run_calib():
            res = await validator_service.validate_question(q_bad, [], expected_role="Machine Learning Engineer")
            # Phase 2 quality gate: this is a deterministic, fail-closed rejection
            # (difficulty_out_of_band) -- true regardless of what the mock LLM's
            # own canned score/passed verdict says.
            self.assertFalse(res.approved)
            self.assertTrue(res.critical_failure)
            self.assertTrue(any("difficulty_out_of_band" in r for r in res.rejection_reasons))

        asyncio.run(run_calib())

    def test_generator_service_rag_context_and_batch_generation(self):
        """Verify generator_service formats RAG context, constructs prompt, and outputs calibrated questions."""
        from question_pipeline.generator_service import generator_service

        async def run_gen():
            qs = await generator_service.generate_batch_for_band(
                role="Machine Learning Engineer",
                experience_band="3-5",
                count=4,
                target_scope="UNIVERSAL"
            )
            self.assertEqual(len(qs), 4)
            for q in qs:
                self.assertEqual(q.role, "Machine Learning Engineer")
                self.assertEqual(q.experience_band, "3-5")
                self.assertEqual(q.paradigm, "ARCHITECTURE")
                self.assertTrue(len(q.source_references) > 0)
                # Verify 7 depth dimensions are populated and valid
                self.assertTrue(1 <= q.technical_depth <= 10)
                self.assertTrue(1 <= q.problem_complexity <= 10)
                self.assertTrue(1 <= q.architecture_complexity <= 10)
                self.assertTrue(1 <= q.troubleshooting <= 10)
                self.assertTrue(1 <= q.business_complexity <= 10)
                self.assertTrue(1 <= q.decision_making <= 10)
                self.assertTrue(1 <= q.leadership_ownership <= 10)

        asyncio.run(run_gen())

    def test_scope_classification_rules(self):
        """Verify explicit scope classification for UNIVERSAL, DOMAIN, and COMPANY."""
        runner = QuestionPipelineRunner()

        # 1. Company specific question
        q_comp = QuestionObject(
            question="How would you optimize Uber's Michelangelo feature store?",
            role="Machine Learning Engineer",
            experience_band="3-5",
            difficulty=7,
            technical_depth=7,
            problem_complexity=7,
            architecture_complexity=7,
            troubleshooting=6,
            business_complexity=5,
            decision_making=6,
            leadership_ownership=4,
            question_type="system_design",
            paradigm="ARCHITECTURE",
            mandatory_skills=["Michelangelo"],
            scope="UNIVERSAL",
            domains=["TRANSPORTATION"],
            applicable_companies=["Uber"]
        )
        self.assertEqual(runner.classify_scope(q_comp), "COMPANY")

        # 2. Domain specific question
        q_dom = q_comp.model_copy(update={
            "applicable_companies": ["Google", "Amazon", "Apple"],
            "domains": ["FINTECH_PAYMENTS"]
        })
        self.assertEqual(runner.classify_scope(q_dom, target_scope="DOMAIN"), "DOMAIN")

        # 3. Universal question
        q_univ = q_comp.model_copy(update={
            "applicable_companies": ["Google", "Amazon", "Microsoft", "Meta"],
            "domains": ["GENERAL_TECH"],
            "scope": "UNIVERSAL"
        })
        self.assertEqual(runner.classify_scope(q_univ, target_scope="UNIVERSAL"), "UNIVERSAL")

    def test_provider_factory_blocked_when_not_mock_and_key_missing(self):
        """Verify factory does not silently return local mock embeddings when MOCK_MODE=False and Gemini is configured."""
        from question_pipeline.providers.factory import get_embedding_provider
        from question_pipeline.providers.gemini_provider import GeminiEmbeddingProvider

        original_mock = config.MOCK_MODE
        original_provider = config.EMBEDDING_PROVIDER
        original_key = config.GEMINI_API_KEY
        try:
            config.MOCK_MODE = False
            config.EMBEDDING_PROVIDER = "gemini"
            config.GEMINI_API_KEY = ""
            provider = get_embedding_provider()
            self.assertIsInstance(provider, GeminiEmbeddingProvider)

            # When invoked with missing/dummy key, it must raise ProviderBlockedException
            async def run_blocked():
                with self.assertRaises(ProviderBlockedException):
                    await provider.embed_texts(["Test text"])

            asyncio.run(run_blocked())
        finally:
            config.MOCK_MODE = original_mock
            config.EMBEDDING_PROVIDER = original_provider
            config.GEMINI_API_KEY = original_key

    def test_completed_role_skips_without_force(self):
        """Verify that a role marked COMPLETED is skipped without --force."""
        role = "Data Engineer"
        from question_pipeline.state_manager import state_manager
        from question_pipeline.pipeline_runner import pipeline_runner

        state_manager.update_role_state(role, status="COMPLETED", accepted_questions_count=20, target_questions=20)

        async def run():
            res = await pipeline_runner.run_role_pipeline(role, target_questions=20, force=False)
            self.assertEqual(res.get("status"), "ALREADY_COMPLETED")
            self.assertEqual(res.get("accepted"), 20)

        asyncio.run(run())

    def test_completed_role_reruns_with_force(self):
        """Verify that a COMPLETED role reruns from batch 0 when --force is True."""
        role = "Data Engineer"
        from question_pipeline.state_manager import state_manager
        from question_pipeline.pipeline_runner import pipeline_runner

        state_manager.update_role_state(role, status="COMPLETED", accepted_questions_count=20, target_questions=20)

        async def run():
            res = await pipeline_runner.run_role_pipeline(role, target_questions=20, force=True)
            self.assertNotEqual(res.get("status"), "ALREADY_COMPLETED")
            # In mock test mode, it successfully runs and completes
            self.assertEqual(res.get("status"), "COMPLETED")
            self.assertGreater(res.get("metrics", {}).get("accepted", 0), 0)

        asyncio.run(run())

    def test_force_affects_only_selected_role(self):
        """Verify that --force resets ONLY the targeted role and preserves all other roles."""
        role_a = "Machine Learning Engineer"
        role_b = "Product Manager (Tech)"
        from question_pipeline.state_manager import state_manager
        from question_pipeline.pipeline_runner import pipeline_runner

        # Set both as completed
        state_manager.update_role_state(role_a, status="COMPLETED", accepted_questions_count=20, target_questions=20)
        state_manager.update_role_state(role_b, status="COMPLETED", accepted_questions_count=20, target_questions=20)

        state_b_before = state_manager.get_role_state(role_b).model_dump()

        async def run():
            # Force rerun ONLY role_a
            await pipeline_runner.run_role_pipeline(role_a, target_questions=20, force=True)

        asyncio.run(run())

        # Verify role_b is completely intact
        state_b_after = state_manager.get_role_state(role_b).model_dump()
        self.assertEqual(state_b_after["status"], "COMPLETED")
        self.assertEqual(state_b_after["accepted_questions_count"], 20)
        self.assertEqual(state_b_after["last_researched_at"], state_b_before["last_researched_at"])

    def test_mock_mode_false_cannot_fallback_to_mock_providers(self):
        """Verify that when MOCK_MODE=false, factory strictly returns real providers and never mock/local fallbacks."""
        from question_pipeline.providers.factory import get_research_provider, get_llm_provider, get_embedding_provider
        from question_pipeline.providers.perplexity_provider import PerplexityResearchProvider
        from question_pipeline.providers.gemini_provider import GeminiLLMProvider, GeminiEmbeddingProvider

        original_mock = config.MOCK_MODE
        original_res = config.RESEARCH_PROVIDER
        original_llm = config.LLM_PROVIDER
        original_emb = config.EMBEDDING_PROVIDER

        try:
            config.MOCK_MODE = False
            config.RESEARCH_PROVIDER = "perplexity"
            config.LLM_PROVIDER = "gemini"
            config.EMBEDDING_PROVIDER = "gemini"

            res_p = get_research_provider()
            self.assertIsInstance(res_p, PerplexityResearchProvider)

            llm_p = get_llm_provider()
            self.assertIsInstance(llm_p, GeminiLLMProvider)

            emb_p = get_embedding_provider()
            self.assertIsInstance(emb_p, GeminiEmbeddingProvider)
        finally:
            config.MOCK_MODE = original_mock
            config.RESEARCH_PROVIDER = original_res
            config.LLM_PROVIDER = original_llm
            config.EMBEDDING_PROVIDER = original_emb

    def test_missing_credentials_fail_clearly_before_api_calls(self):
        """Verify that missing/placeholder credentials fail immediately in real mode with BLOCKED state."""
        from question_pipeline.pipeline_runner import pipeline_runner
        from question_pipeline.state_manager import state_manager

        original_mock = config.MOCK_MODE
        original_gemini_key = config.GEMINI_API_KEY
        original_perp_key = config.PERPLEXITY_API_KEY
        original_research_provider = config.RESEARCH_PROVIDER

        try:
            config.MOCK_MODE = False
            config.GEMINI_API_KEY = "dummy_placeholder_key"
            config.PERPLEXITY_API_KEY = ""
            # Explicit: this test exercises the Perplexity credential-check
            # branch specifically, independent of whichever provider .env
            # currently sets as the real default (tavily_firecrawl).
            config.RESEARCH_PROVIDER = "perplexity"

            test_role = "Senior RTL / Logic Design Engineer"

            # Must raise ProviderBlockedException before making any calls
            with self.assertRaises(ProviderBlockedException) as ctx:
                pipeline_runner.validate_prerequisites(test_role)

            err_text = str(ctx.exception).lower()
            self.assertIn("perplexity_api_key", err_text)
            self.assertIn("gemini_api_key", err_text)

            # Role state must be marked BLOCKED
            state = state_manager.get_role_state(test_role)
            self.assertEqual(state.status, "BLOCKED")
        finally:
            config.MOCK_MODE = original_mock
            config.GEMINI_API_KEY = original_gemini_key
            config.PERPLEXITY_API_KEY = original_perp_key
            config.RESEARCH_PROVIDER = original_research_provider

if __name__ == "__main__":
    unittest.main()


