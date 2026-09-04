from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool
    cache_size: int


class MessageResponse(BaseModel):
    message: str
    success: bool = True
    data: Optional[Dict[str, Any]] = None
