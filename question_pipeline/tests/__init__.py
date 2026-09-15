"""
Test-suite isolation bootstrap.

This MUST set QUESTION_PIPELINE_DATA_DIR before any test module in this
package imports anything from question_pipeline (config, state_manager,
rag_service, cost_tracker, pipeline_runner, etc. all resolve their file paths
from config.DATA_DIR at import time). Because this is the tests package's
__init__.py, Python always executes it before test_pipeline.py or
test_quality_gate.py are imported -- regardless of which one a test runner
picks first -- so this is the one place that is guaranteed to run early
enough to redirect every pipeline singleton to a throwaway directory instead
of the real question_pipeline/data/.

Without this, tests that exercise the module-level singletons (state_manager,
rag_service, cost_tracker, pipeline_runner) would read and write the real
production/generated data files -- which is exactly the bug this module
fixes.
"""
import os
import atexit
import shutil
import tempfile

os.environ.setdefault("MOCK_MODE", "true")

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="question_pipeline_test_data_")
os.environ["QUESTION_PIPELINE_DATA_DIR"] = _TEST_DATA_DIR


def _cleanup_test_data_dir():
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


atexit.register(_cleanup_test_data_dir)
