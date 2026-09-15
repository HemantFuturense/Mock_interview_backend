import logging
from fastapi import APIRouter, HTTPException, status
from .models import ResearchRequest, ResearchResponse, ServiceStatusResponse
from .orchestrator import web_research_orchestrator
from .config import config, is_valid_key

logger = logging.getLogger(__name__)

research_router = APIRouter(prefix="/api/research", tags=["Web Research Pipeline"])

@research_router.get("/status", response_model=ServiceStatusResponse)
async def get_research_status():
    """Health and configuration check for web research providers."""
    tavily_ok = is_valid_key(config.TAVILY_API_KEY)
    firecrawl_ok = is_valid_key(config.FIRECRAWL_API_KEY)
    gemini_ok = is_valid_key(config.GEMINI_API_KEY)

    overall = "READY" if (tavily_ok and gemini_ok) else "CONFIG_REQUIRED"

    return ServiceStatusResponse(
        tavily_configured=tavily_ok,
        firecrawl_configured=firecrawl_ok,
        gemini_configured=gemini_ok,
        gemini_model=config.GEMINI_RESEARCH_MODEL,
        status=overall
    )

@research_router.post("", response_model=ResearchResponse)
async def perform_web_research(request: ResearchRequest):
    """Execute end-to-end web research: Tavily Search -> Firecrawl Scrape -> Gemini Synthesis."""
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must be at least 3 characters long."
        )

    try:
        response = await web_research_orchestrator.execute_research(request)
        return response
    except Exception as e:
        logger.error(f"[API_RESEARCH_ERROR] Unhandled exception during research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Web research pipeline execution error: {str(e)}"
        )
