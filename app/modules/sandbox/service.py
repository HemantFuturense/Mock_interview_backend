import asyncio
import json
import urllib.request
from typing import Any, Dict, List, Optional
from fastapi import HTTPException

from app.config.constants import DEFAULT_RUNTIMES
from app.config.settings import config
from app.core.logger import logger
from app.modules.sandbox.runners import (
    execute_via_piston,
    run_local_python_sync,
    run_local_sql,
    run_via_judge0,
)

PYTHON_VISUALIZATION_BOOTSTRAP = """
import io
import base64

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def __emit_all_figures_for_ai_mock_interview():
    if plt is None:
        return
    try:
        fig_nums = plt.get_fignums()
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            print("__IMAGE_PNG__:" + img_b64)
        plt.close("all")
    except Exception:
        pass

__emit_all_figures_for_ai_mock_interview()
"""


def inject_python_visualization_bootstrap(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inject matplotlib/seaborn figure capturing script into Python submissions."""
    language = str(payload.get("language", "")).lower()
    if language not in {"python", "py", "py3", "python3"}:
        return payload
    files = payload.get("files")
    if not isinstance(files, list):
        return payload
    main_index: Optional[int] = None
    for i, item in enumerate(files):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name.endswith(".py"):
            continue
        if name == "main.py" or main_index is None:
            main_index = i
    if main_index is None:
        if files:
            main_index = 0
        else:
            return payload
    content = files[main_index].get("content")
    if not isinstance(content, str):
        return payload
    if "__IMAGE_PNG__:" in content:
        return payload
    files[main_index]["content"] = content + "\n\n" + PYTHON_VISUALIZATION_BOOTSTRAP
    return payload


async def execute_hybrid_code(payload: Dict[str, Any]) -> Any:
    """Execute code across local, Judge0, and Piston tiers with automatic fallback."""
    payload = inject_python_visualization_bootstrap(payload)
    language = str(payload.get("language", "")).lower()
    files = payload.get("files") or []
    code = ""
    if isinstance(files, list) and files and isinstance(files[0], dict):
        code = str(files[0].get("content") or "")
    stdin_val = str(payload.get("stdin") or "")

    # Execute locally for Python and SQL for instant reliability & zero network dependencies
    if language in {"python", "py", "py3", "python3"}:
        return await asyncio.to_thread(run_local_python_sync, code, stdin_val)
    elif language in {"sql", "sqlite", "sqlite3", "mysql", "postgres", "postgresql"}:
        return await asyncio.to_thread(run_local_sql, code)

    # Try external Judge0 CE for other languages
    try:
        return await run_via_judge0(language, code, stdin_val)
    except Exception as judge0_err:
        logger.warning("Judge0 execution failed, attempting Piston fallback: %s", judge0_err)

    # Final fallback: proxy to Piston instance
    try:
        return await execute_via_piston(payload)
    except urllib.error.HTTPError as http_err:
        logger.error("Piston execute HTTP error: %s", http_err)
        raise HTTPException(status_code=http_err.code, detail="Failed to execute code in runner")
    except Exception as exc:
        logger.error("Piston execute error: %s", exc)
        raise HTTPException(status_code=502, detail="Error contacting code runner service")


async def fetch_runtimes() -> List[Dict[str, Any]]:
    """Fetch supported code execution runtimes with built-in default fallback."""
    base_url = config.PISTON_BASE_URL.rstrip("/")
    url = f"{base_url}/runtimes"

    def _fetch() -> Any:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    try:
        data = await asyncio.to_thread(_fetch)
        if isinstance(data, list) and data:
            return data
    except Exception as exc:
        logger.warning("Failed to fetch Piston runtimes (%s), using default built-in runtimes.", exc)
    return DEFAULT_RUNTIMES
