from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class FileItem(BaseModel):
    name: Optional[str] = None
    content: str


class PistonExecuteRequest(BaseModel):
    language: str
    version: Optional[str] = "*"
    files: List[FileItem]
    stdin: Optional[str] = ""
    args: Optional[List[str]] = None
    compile_timeout: Optional[int] = 10000
    run_timeout: Optional[int] = 15000
    compile_memory_limit: Optional[int] = -1
    run_memory_limit: Optional[int] = -1


class ExecutionStageResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    code: int = 0


class PistonExecuteResponse(BaseModel):
    run: ExecutionStageResult
    compile: Optional[ExecutionStageResult] = None
