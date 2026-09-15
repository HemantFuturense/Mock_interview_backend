import time
import logging
from typing import Optional, List, Callable, Dict, Any
from .config import config, is_valid_key
from .models import SourceItem, StageResult, ResearchRequest, ResearchResponse
from .clients.tavily_client import TavilySearchClient
from .clients.firecrawl_client import FirecrawlScrapeClient
from .clients.gemini_client import GeminiResearchClient

logger = logging.getLogger(__name__)

class WebResearchOrchestrator:
    """Central orchestrator coordinating Tavily search, Firecrawl scraping, and Gemini synthesis."""

    def __init__(
        self,
        search_client: Optional[TavilySearchClient] = None,
        scraper_client: Optional[FirecrawlScrapeClient] = None,
        llm_client: Optional[GeminiResearchClient] = None
    ):
        self.search_client = search_client or TavilySearchClient()
        self.scraper_client = scraper_client or FirecrawlScrapeClient()
        self.llm_client = llm_client or GeminiResearchClient()

    async def execute_research(
        self,
        request: ResearchRequest,
        on_stage_update: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
    ) -> ResearchResponse:
        """Execute the multi-stage web research pipeline with timing telemetry and fallback tolerance."""
        start_time = time.time()
        question = request.question.strip()
        max_results = request.max_results or config.TAVILY_MAX_RESULTS
        max_scrape = request.max_scrape_pages or config.FIRECRAWL_MAX_PAGES

        logger.info(f"[RESEARCH_START] Question: '{question[:80]}' (max_results={max_results}, max_scrape={max_scrape})")
        stages: List[StageResult] = []

        def _notify_stage(stage_name: str, status: str, duration: float, details: Dict[str, Any] = None):
            res = StageResult(stage=stage_name, status=status, duration_seconds=round(duration, 3), details=details or {})
            stages.append(res)
            if on_stage_update:
                try:
                    on_stage_update(stage_name, status, details or {})
                except Exception as ex:
                    logger.warning(f"[STAGE_CALLBACK_ERROR] Error in stage callback: {ex}")

        # -------------------------------------------------------------
        # Stage 1: Tavily Search
        # -------------------------------------------------------------
        stage1_start = time.time()
        search_sources: List[SourceItem] = []
        try:
            search_sources = await self.search_client.search(query=question, max_results=max_results)
            stage1_dur = time.time() - stage1_start
            if search_sources:
                logger.info(f"[TAVILY_SEARCH_COMPLETE] Found {len(search_sources)} sources in {stage1_dur:.2f}s")
                _notify_stage("tavily_search", "SUCCESS", stage1_dur, {"sources_found": len(search_sources)})
            else:
                logger.warning(f"[TAVILY_SEARCH_EMPTY] No sources returned or Tavily key not configured.")
                _notify_stage("tavily_search", "FAILED", stage1_dur, {"reason": "No search results returned or invalid API key"})
        except Exception as e:
            stage1_dur = time.time() - stage1_start
            logger.error(f"[TAVILY_SEARCH_ERROR] Search failed: {e}")
            _notify_stage("tavily_search", "FAILED", stage1_dur, {"error": str(e)})

        # -------------------------------------------------------------
        # Stage 2: Source Selection & Ranking
        # -------------------------------------------------------------
        stage2_start = time.time()
        selected_sources: List[SourceItem] = []
        if search_sources:
            # Sort by relevance score descending if present
            sorted_sources = sorted(
                search_sources,
                key=lambda s: s.relevance_score if s.relevance_score is not None else 0.0,
                reverse=True
            )
            # Pick top candidates for scraping
            selected_sources = sorted_sources[:max_scrape]
            stage2_dur = time.time() - stage2_start
            logger.info(f"[SOURCE_RANKING_COMPLETE] Selected {len(selected_sources)} top sources for deep scraping")
            _notify_stage("source_ranking", "SUCCESS", stage2_dur, {
                "selected_count": len(selected_sources),
                "total_considered": len(search_sources)
            })
        else:
            stage2_dur = time.time() - stage2_start
            _notify_stage("source_ranking", "SKIPPED", stage2_dur, {"reason": "No search sources to rank"})

        # -------------------------------------------------------------
        # Stage 3: Firecrawl Webpage Scraping
        # -------------------------------------------------------------
        stage3_start = time.time()
        scrape_urls = [s.url for s in selected_sources]
        scraped_content_map: Dict[str, Optional[str]] = {}
        successful_scrapes = 0
        failed_scrapes = 0

        scraper_is_ready = getattr(self.scraper_client, "is_configured", lambda: True)()
        if scrape_urls and scraper_is_ready:
            try:
                scraped_content_map = await self.scraper_client.scrape_batch(scrape_urls, max_pages=max_scrape)
                for s in selected_sources:
                    content = scraped_content_map.get(s.url)
                    if content:
                        s.scraped_content = content
                        successful_scrapes += 1
                    else:
                        failed_scrapes += 1
                stage3_dur = time.time() - stage3_start
                status_str = "SUCCESS" if successful_scrapes > 0 else "FALLBACK"
                logger.info(f"[FIRECRAWL_EXTRACTION_COMPLETE] Scraped {successful_scrapes}/{len(scrape_urls)} pages successfully in {stage3_dur:.2f}s")
                _notify_stage("firecrawl_extraction", status_str, stage3_dur, {
                    "successful_pages": successful_scrapes,
                    "failed_pages": failed_scrapes,
                    "urls_attempted": len(scrape_urls)
                })
            except Exception as e:
                stage3_dur = time.time() - stage3_start
                logger.error(f"[FIRECRAWL_EXTRACTION_ERROR] Firecrawl scraping failed: {e}. Falling back to search snippets.")
                _notify_stage("firecrawl_extraction", "FALLBACK", stage3_dur, {"error": str(e), "fallback": "Using Tavily snippets"})
        else:
            stage3_dur = time.time() - stage3_start
            reason_str = "FIRECRAWL_API_KEY not configured. Using Tavily snippets as fallback." if not scraper_is_ready else "No URLs to scrape."
            logger.info(f"[FIRECRAWL_EXTRACTION_SKIPPED] {reason_str}")
            _notify_stage("firecrawl_extraction", "FALLBACK" if search_sources else "SKIPPED", stage3_dur, {
                "reason": reason_str,
                "fallback": "Using search snippets"
            })

        # -------------------------------------------------------------
        # Stage 4: Gemini Analysis & Synthesis
        # -------------------------------------------------------------
        stage4_start = time.time()
        answer = ""
        # Final sources passed to synthesis: prioritize those with scraped content or top snippets
        final_sources = selected_sources if selected_sources else search_sources[:max_scrape]

        try:
            logger.info(f"[GEMINI_SYNTHESIS_START] Initiating reasoning with {len(final_sources)} sources using model '{config.GEMINI_RESEARCH_MODEL}'...")
            answer = await self.llm_client.synthesize(question=question, sources=final_sources)
            stage4_dur = time.time() - stage4_start
            logger.info(f"[GEMINI_SYNTHESIS_COMPLETE] Generated {len(answer)} characters in {stage4_dur:.2f}s")
            _notify_stage("gemini_synthesis", "SUCCESS", stage4_dur, {
                "model": config.GEMINI_RESEARCH_MODEL,
                "answer_length": len(answer),
                "sources_provided": len(final_sources)
            })
        except Exception as e:
            stage4_dur = time.time() - stage4_start
            logger.error(f"[GEMINI_SYNTHESIS_ERROR] Synthesis failed: {e}")
            _notify_stage("gemini_synthesis", "FAILED", stage4_dur, {"error": str(e)})
            answer = (
                f"### Research Incomplete\n\n"
                f"An error occurred during Gemini reasoning: {e}\n\n"
                f"Please verify that `GEMINI_API_KEY` is configured with an active quota in `.env`."
            )

        total_duration = time.time() - start_time
        overall_status = "SUCCESS" if ("FAILED" not in [s.status for s in stages if s.stage in ("gemini_synthesis",)]) and answer else "PARTIAL"

        logger.info(f"[RESEARCH_COMPLETE] Total pipeline elapsed time: {total_duration:.2f}s | Status: {overall_status}")

        return ResearchResponse(
            question=question,
            answer=answer,
            sources=final_sources,
            duration_seconds=round(total_duration, 3),
            stages=stages,
            status=overall_status,
            error_message=None if overall_status == "SUCCESS" else "One or more stages operated in fallback or failed mode."
        )

web_research_orchestrator = WebResearchOrchestrator()
