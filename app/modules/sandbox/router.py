from typing import Any
from fastapi import APIRouter, HTTPException, Request
from app.modules.sandbox.service import execute_hybrid_code, fetch_runtimes

router = APIRouter(prefix="/piston", tags=["Code Sandbox"])


@router.post("/execute")
async def execute_code_endpoint(request: Request) -> Any:
    """Execute code reliably using local runtime, Judge0 CE, or Piston."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    return await execute_hybrid_code(payload)


@router.get("/runtimes")
async def get_runtimes_endpoint() -> Any:
    """Fetch runtimes or return robust default built-in runtimes fallback."""
    return await fetch_runtimes()
