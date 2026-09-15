"""
Proves the test-isolation fix: no test in this suite may read or write the
real question_pipeline/data/ directory. Every pipeline singleton (config,
state_manager, cost_tracker, rag_service, pipeline_runner) must resolve its
file paths to the throwaway directory that question_pipeline/tests/__init__.py
points QUESTION_PIPELINE_DATA_DIR at.

This directly re-exercises the two calls that were previously proven to
mutate real files (a full run_role_pipeline(..., force=True) call through the
module-level `pipeline_runner` singleton -- the same object
test_completed_role_reruns_with_force and test_force_affects_only_selected_role
use) and shows the real directory is untouched afterward.
"""
import os
import asyncio
import hashlib
import unittest
from pathlib import Path

# tests/__init__.py has already set QUESTION_PIPELINE_DATA_DIR and MOCK_MODE
# by the time this module is imported.
from question_pipeline.config import config

REAL_PRODUCTION_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _snapshot(directory: Path) -> dict:
    """relative_path -> sha256 of file contents, for every file under `directory`."""
    if not directory.exists():
        return {}
    snap = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(directory))
            snap[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snap


class TestDataIsolation(unittest.TestCase):
    def test_config_data_dir_is_not_the_real_production_directory(self):
        """The single most important assertion: if this ever starts pointing
        back at the real directory, every other guarantee in this file is void."""
        self.assertNotEqual(
            config.DATA_DIR.resolve(), REAL_PRODUCTION_DATA_DIR.resolve(),
            "config.DATA_DIR must be redirected to an isolated test directory, "
            "never the real question_pipeline/data/ folder."
        )
        self.assertIn(
            "question_pipeline_test_data_", str(config.DATA_DIR),
            "config.DATA_DIR should be the temp directory created by tests/__init__.py."
        )
        self.assertEqual(os.environ.get("QUESTION_PIPELINE_DATA_DIR"), str(config.DATA_DIR))

    def test_real_production_data_is_byte_identical_after_heavy_pipeline_use(self):
        """Re-runs the exact kind of calls that previously mutated real files
        (module-level singleton, force=True full role rerun) and proves the
        real directory does not change by a single byte."""
        before = _snapshot(REAL_PRODUCTION_DATA_DIR)

        # Exercise every singleton the old, unsafe tests exercised.
        from question_pipeline.state_manager import state_manager
        from question_pipeline.cost_tracker import cost_tracker
        from question_pipeline.rag_service import rag_service
        from question_pipeline.pipeline_runner import pipeline_runner
        from question_pipeline.models import KnowledgeChunk

        state_manager.update_role_state("Isolation Test Role", status="RUNNING", current_batch=1, total_batches=5)
        state_manager.mark_waiting_for_quota("Isolation Test Role", "simulated for isolation test")
        cost_tracker.record_usage(provider="gemini", model="gemini-2.5-flash-lite", task="generation", input_tokens=10, output_tokens=10)

        async def _index():
            await rag_service.index_chunks([
                KnowledgeChunk(chunk_id="isolation_test_chunk", role="Isolation Test Role",
                               topic="Isolation", text="This chunk must never reach the real vector store.")
            ])
        asyncio.run(_index())

        # The exact pattern from test_completed_role_reruns_with_force /
        # test_force_affects_only_selected_role: a full force=True role run
        # through the module-level pipeline_runner singleton.
        async def _run():
            await pipeline_runner.run_role_pipeline("Isolation Test Role", target_questions=4, force=True)
        asyncio.run(_run())

        after = _snapshot(REAL_PRODUCTION_DATA_DIR)

        self.assertEqual(
            before, after,
            "The real question_pipeline/data/ directory changed during a test run. "
            "Test isolation is broken."
        )

    def test_isolated_directory_actually_received_the_writes(self):
        """Not just 'nothing happened' -- proves the isolated temp directory
        genuinely receives the pipeline's writes, so isolation is real
        redirection, not accidental no-ops."""
        from question_pipeline.state_manager import state_manager

        state_manager.update_role_state("Isolation Proof Role", status="COMPLETED", accepted_questions_count=1)

        self.assertTrue(config.STATE_FILE.exists())
        self.assertTrue(str(config.STATE_FILE).startswith(str(config.DATA_DIR)))
        content = config.STATE_FILE.read_text(encoding="utf-8")
        self.assertIn("Isolation Proof Role", content)

        # And, once more, the real file was not touched by this.
        real_state_file = REAL_PRODUCTION_DATA_DIR / "state.json"
        if real_state_file.exists():
            real_content = real_state_file.read_text(encoding="utf-8")
            self.assertNotIn("Isolation Proof Role", real_content)


if __name__ == "__main__":
    unittest.main()
