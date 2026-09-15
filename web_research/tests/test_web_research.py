import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from web_research.models import (
    SourceItem,
    ResearchRequest,
    ResearchResponse,
    StageResult
)
from web_research.clients.tavily_client import TavilySearchClient
from web_research.clients.firecrawl_client import FirecrawlScrapeClient
from web_research.clients.gemini_client import GeminiResearchClient
from web_research.orchestrator import WebResearchOrchestrator
from web_research.config import config

class TestWebResearchPipeline(unittest.TestCase):

    def test_tavily_search_success(self):
        """Verify Tavily client parses valid response into structured SourceItems."""
        client = TavilySearchClient(api_key="tvly-mock-valid-key-12345")

        mock_resp_data = {
            "query": "vector databases",
            "results": [
                {
                    "title": "Introduction to Vector DBs",
                    "url": "https://example.com/vector-db",
                    "content": "Vector databases index high-dimensional embeddings.",
                    "score": 0.94
                },
                {
                    "title": "Milvus vs Pinecone",
                    "url": "https://example.com/milvus-pinecone",
                    "content": "Comparison of open-source vs managed vector search.",
                    "score": 0.88
                }
            ]
        }

        async def run():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_resp_data
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                sources = await client.search("vector databases", max_results=2)
                self.assertEqual(len(sources), 2)
                self.assertEqual(sources[0].title, "Introduction to Vector DBs")
                self.assertEqual(sources[0].url, "https://example.com/vector-db")
                self.assertEqual(sources[0].relevance_score, 0.94)
                self.assertEqual(sources[0].source_type, "tavily_search")

        asyncio.run(run())

    def test_tavily_search_failure_and_empty(self):
        """Verify Tavily client handles rate limits and network errors without crashing."""
        client = TavilySearchClient(api_key="tvly-mock-valid-key-12345")

        async def run_429():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 429
                mock_post.return_value = mock_resp

                sources = await client.search("rate limited query")
                self.assertEqual(sources, [])

        asyncio.run(run_429())

        async def run_unconfigured():
            unconf_client = TavilySearchClient(api_key="")
            sources = await unconf_client.search("any query")
            self.assertEqual(sources, [])

        asyncio.run(run_unconfigured())

    def test_firecrawl_scrape_success(self):
        """Verify Firecrawl client extracts clean markdown from webpage."""
        client = FirecrawlScrapeClient(api_key="fc-mock-valid-key-12345")

        mock_resp_data = {
            "success": True,
            "data": {
                "markdown": "# Vector DB Architecture\n\nHNSW algorithms provide sub-10ms approximate nearest neighbor queries.",
                "metadata": {
                    "title": "Vector DB Architecture",
                    "sourceURL": "https://example.com/vector-arch"
                }
            }
        }

        async def run():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_resp_data
                mock_resp.raise_for_status = MagicMock()
                mock_post.return_value = mock_resp

                md = await client.scrape_url("https://example.com/vector-arch")
                self.assertIsNotNone(md)
                self.assertIn("HNSW algorithms", md)

        asyncio.run(run())

    def test_firecrawl_scrape_partial_failure(self):
        """Verify batch scraping handles one failed page gracefully while retaining others."""
        client = FirecrawlScrapeClient(api_key="fc-mock-valid-key-12345")

        async def run():
            async def mock_post(url, **kwargs):
                req_url = kwargs.get("json", {}).get("url")
                mock_resp = MagicMock()
                if "fail" in req_url:
                    mock_resp.status_code = 404
                    mock_resp.json.return_value = {"success": False, "error": "Not found"}
                    mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
                else:
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = {
                        "success": True,
                        "data": {"markdown": f"Content for {req_url}"}
                    }
                    mock_resp.raise_for_status = MagicMock()
                return mock_resp

            with patch("httpx.AsyncClient.post", side_effect=mock_post):
                batch_urls = [
                    "https://example.com/page1",
                    "https://example.com/fail-page",
                    "https://example.com/page2"
                ]
                results = await client.scrape_batch(batch_urls, max_pages=3)
                self.assertEqual(len(results), 3)
                self.assertIsNotNone(results["https://example.com/page1"])
                self.assertIsNone(results["https://example.com/fail-page"])
                self.assertIsNotNone(results["https://example.com/page2"])

        asyncio.run(run())

    def test_gemini_synthesis_success(self):
        """Verify Gemini reasoning synthesizes grounded response with source citations."""
        client = GeminiResearchClient(api_key="gemini-mock-valid-key-12345")

        sources = [
            SourceItem(
                title="Consensus in Distributed Systems",
                url="https://example.com/consensus",
                snippet="Raft and Paxos guarantee linearizability.",
                scraped_content="Raft uses leader election and log replication [1]."
            ),
            SourceItem(
                title="Eventual Consistency",
                url="https://example.com/eventual",
                snippet="Dynamo style quorum writes R+W>N.",
                scraped_content="Quorum consistency trades latency for freshness [2]."
            )
        ]

        async def run():
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model_instance = MagicMock()
                mock_response = MagicMock()
                mock_response.text = (
                    "### Distributed Consensus Overview\n\n"
                    "Raft guarantees strong linearizability through leader election [1]. "
                    "In contrast, Dynamo architectures leverage quorum reads and writes for higher availability [2]."
                )
                mock_model_instance.generate_content.return_value = mock_response
                mock_model_cls.return_value = mock_model_instance

                answer = await client.synthesize("Compare Paxos and Dynamo consistency models", sources)
                self.assertIn("Raft guarantees strong linearizability", answer)
                self.assertIn("[1]", answer)
                self.assertIn("[2]", answer)

        asyncio.run(run())

    def test_gemini_synthesis_failure(self):
        """Verify Gemini synthesis failure raises RuntimeError with diagnostic details."""
        client = GeminiResearchClient(api_key="gemini-mock-valid-key-12345")

        async def run():
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model_instance = MagicMock()
                mock_model_instance.generate_content.side_effect = Exception("API Quota Exhausted")
                mock_model_cls.return_value = mock_model_instance

                with self.assertRaises(RuntimeError) as ctx:
                    await client.synthesize("Test Question", [])
                self.assertIn("Quota Exhausted", str(ctx.exception))

        asyncio.run(run())

    def test_end_to_end_orchestration(self):
        """Verify orchestrator runs Tavily -> Rank -> Firecrawl -> Gemini pipeline with stage telemetry."""
        mock_search = AsyncMock()
        mock_search.search.return_value = [
            SourceItem(title="Src 1", url="https://example.com/1", snippet="Snippet 1", relevance_score=0.9),
            SourceItem(title="Src 2", url="https://example.com/2", snippet="Snippet 2", relevance_score=0.7)
        ]

        mock_scraper = AsyncMock()
        mock_scraper.scrape_batch.return_value = {
            "https://example.com/1": "Full markdown content for Src 1"
        }

        mock_llm = AsyncMock()
        mock_llm.synthesize.return_value = "Synthesized answer with verified citations [1]."

        orchestrator = WebResearchOrchestrator(
            search_client=mock_search,
            scraper_client=mock_scraper,
            llm_client=mock_llm
        )

        async def run():
            req = ResearchRequest(question="Explain Kafka event streaming architecture", max_results=2, max_scrape_pages=1)
            resp = await orchestrator.execute_research(req)

            self.assertEqual(resp.status, "SUCCESS")
            self.assertEqual(resp.question, "Explain Kafka event streaming architecture")
            self.assertIn("Synthesized answer", resp.answer)
            self.assertTrue(len(resp.sources) > 0)
            self.assertEqual(resp.sources[0].scraped_content, "Full markdown content for Src 1")

            # Check real stage breakdown
            stage_names = [s.stage for s in resp.stages]
            self.assertIn("tavily_search", stage_names)
            self.assertIn("source_ranking", stage_names)
            self.assertIn("firecrawl_extraction", stage_names)
            self.assertIn("gemini_synthesis", stage_names)

            for s in resp.stages:
                self.assertGreaterEqual(s.duration_seconds, 0.0)

        asyncio.run(run())

    def test_api_route_integration(self):
        """Verify POST /api/research endpoint responds with valid schema and HTTP 200."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)

        mock_response = ResearchResponse(
            question="What is WebAssembly?",
            answer="WebAssembly (WASM) is a binary instruction format [1].",
            sources=[
                SourceItem(title="Wasm Docs", url="https://webassembly.org", snippet="Fast binary format", relevance_score=0.98)
            ],
            duration_seconds=1.23,
            stages=[
                StageResult(stage="tavily_search", status="SUCCESS", duration_seconds=0.3, details={"sources_found": 1}),
                StageResult(stage="source_ranking", status="SUCCESS", duration_seconds=0.01, details={"selected_count": 1}),
                StageResult(stage="firecrawl_extraction", status="SUCCESS", duration_seconds=0.4, details={"successful_pages": 1}),
                StageResult(stage="gemini_synthesis", status="SUCCESS", duration_seconds=0.52, details={"model": "gemini-2.5-flash-lite"})
            ],
            status="SUCCESS"
        )

        with patch("web_research.router.web_research_orchestrator.execute_research", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response

            res = client.post("/api/research", json={"question": "What is WebAssembly?"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["question"], "What is WebAssembly?")
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(len(data["sources"]), 1)
            self.assertEqual(len(data["stages"]), 4)

        # Test status endpoint
        res_status = client.get("/api/research/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertIn("gemini_model", res_status.json())

if __name__ == "__main__":
    unittest.main()
