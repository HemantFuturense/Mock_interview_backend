from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SourceItem(BaseModel):
    """Normalized source representation preserving provenance and relevance."""
    title: str
    url: str
    snippet: Optional[str] = None
    scraped_content: Optional[str] = None
    relevance_score: Optional[float] = None
    source_type: str = "web"

class StageResult(BaseModel):
    """Detailed metadata for a single pipeline stage."""
    stage: str
    status: str  # "SUCCESS", "FAILED", "FALLBACK", "SKIPPED"
    duration_seconds: float
    details: Optional[Dict[str, Any]] = None

class ResearchRequest(BaseModel):
    """Client request schema for web research."""
    question: str = Field(..., min_length=3, max_length=1000, description="The technical or interview topic to research")
    max_results: Optional[int] = Field(default=None, ge=1, le=10, description="Number of search results to fetch")
    max_scrape_pages: Optional[int] = Field(default=None, ge=1, le=5, description="Number of pages to scrape with Firecrawl")

class ResearchResponse(BaseModel):
    """Complete structured response including answer, verified sources, and stage telemetry."""
    question: str
    answer: str
    sources: List[SourceItem]
    duration_seconds: float
    stages: List[StageResult]
    status: str  # "SUCCESS", "PARTIAL", "FAILED"
    error_message: Optional[str] = None

class ServiceStatusResponse(BaseModel):
    """Health check response for configured web research providers."""
    tavily_configured: bool
    firecrawl_configured: bool
    gemini_configured: bool
    gemini_model: str
    status: str
