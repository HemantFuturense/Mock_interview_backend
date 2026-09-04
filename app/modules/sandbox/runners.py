import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Any, Dict

from app.config.constants import JUDGE0_LANG_MAP
from app.config.settings import config
from app.core.logger import logger


def run_local_python_sync(code: str, stdin_val: str) -> Dict[str, Any]:
    """Execute Python code locally in a subprocess with a 15-second timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        tf_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, tf_path],
            input=stdin_val.encode("utf-8") if stdin_val else b"",
            capture_output=True,
            timeout=15.0,
        )
        stdout_str = proc.stdout.decode("utf-8", errors="replace")
        stderr_str = proc.stderr.decode("utf-8", errors="replace")
        return {
            "run": {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "output": stdout_str + ("\n" + stderr_str if stderr_str else ""),
                "code": proc.returncode or 0,
            },
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    except subprocess.TimeoutExpired:
        return {
            "run": {
                "stdout": "",
                "stderr": "Execution timed out (exceeded 15 seconds limit).",
                "output": "Execution timed out (exceeded 15 seconds limit).",
                "code": 124,
            },
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    except Exception as exc:
        return {
            "run": {
                "stdout": "",
                "stderr": f"Error executing Python: {str(exc)}",
                "output": f"Error executing Python: {str(exc)}",
                "code": 1,
            },
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    finally:
        try:
            os.remove(tf_path)
        except Exception:
            pass


def run_local_sql(code: str) -> Dict[str, Any]:
    """Execute SQL queries locally using an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    output_lines = []
    try:
        statements = [s.strip() for s in code.split(";") if s.strip()]
        if not statements:
            statements = [code]
        for stmt in statements:
            cursor.execute(stmt)
            if cursor.description:
                headers = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                if headers and rows:
                    col_widths = [len(str(h)) for h in headers]
                    for row in rows:
                        for i, val in enumerate(row):
                            if i < len(col_widths):
                                col_widths[i] = max(col_widths[i], len(str(val if val is not None else "NULL")))
                    header_line = " | ".join(f"{str(headers[i]).ljust(col_widths[i])}" for i in range(len(headers)))
                    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
                    output_lines.append(header_line)
                    output_lines.append(sep_line)
                    for row in rows:
                        row_line = " | ".join(
                            f"{str(row[i] if row[i] is not None else 'NULL').ljust(col_widths[i])}"
                            for i in range(len(headers))
                        )
                        output_lines.append(row_line)
                    output_lines.append(f"({len(rows)} rows)\n")
                elif headers:
                    output_lines.append(" | ".join(headers))
                    output_lines.append("(0 rows)\n")
        stdout_str = "\n".join(output_lines)
        if not stdout_str and statements:
            stdout_str = "Query executed successfully."
        return {
            "run": {"stdout": stdout_str, "stderr": "", "output": stdout_str, "code": 0},
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    except Exception as e:
        err_str = f"SQL Error: {str(e)}"
        return {
            "run": {
                "stdout": "\n".join(output_lines),
                "stderr": err_str,
                "output": ("\n".join(output_lines) + "\n" + err_str if output_lines else err_str),
                "code": 1,
            },
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    finally:
        conn.close()


async def run_via_judge0(language: str, code: str, stdin_val: str) -> Dict[str, Any]:
    """Execute code using public Judge0 CE API when local runner is not available."""
    lang_id = JUDGE0_LANG_MAP.get(language, 71)

    def _do_judge0() -> Dict[str, Any]:
        body = json.dumps({"source_code": code, "language_id": lang_id, "stdin": stdin_val}).encode("utf-8")
        req = urllib.request.Request(
            "https://ce.judge0.com/submissions?base64_encoded=false&wait=true",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))

    res = await asyncio.to_thread(_do_judge0)
    stdout_str = res.get("stdout") or ""
    stderr_str = res.get("stderr") or ""
    compile_out = res.get("compile_output") or ""
    status = res.get("status") or {}
    status_id = status.get("id", 0)
    code_val = 0 if status_id == 3 else (1 if status_id != 6 else 0)
    compile_code = 1 if status_id == 6 else 0
    output_str = stdout_str
    if compile_out:
        output_str = compile_out + "\n" + output_str
    elif stderr_str:
        output_str = output_str + "\n" + stderr_str
    return {
        "run": {
            "stdout": stdout_str,
            "stderr": stderr_str,
            "output": output_str.strip(),
            "code": code_val,
        },
        "compile": {"output": compile_out, "stderr": compile_out, "code": compile_code},
    }


async def execute_via_piston(payload: Dict[str, Any]) -> Any:
    """Proxy execution to remote Piston instance."""
    base_url = config.PISTON_BASE_URL.rstrip("/")
    url = f"{base_url}/execute"

    def _post() -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    return await asyncio.to_thread(_post)
