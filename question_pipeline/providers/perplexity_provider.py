import httpx
import hashlib
import json
from typing import Optional, List, Dict, Any
from .base import ResearchProvider
from ..models import ResearchOutput
from ..config import config
from ..governor import governor, ProviderBlockedException, QuotaExhaustedException
from ..cost_tracker import cost_tracker

class PerplexityResearchProvider(ResearchProvider):
    """Web research provider leveraging Perplexity Sonar."""
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.RESEARCH_MODEL
        self.api_url = "https://api.perplexity.ai/chat/completions"

    @property
    def provider_name(self) -> str:
        return "perplexity"

    def _ensure_api_key(self):
        if not config.PERPLEXITY_API_KEY or "dummy" in config.PERPLEXITY_API_KEY.lower():
            raise ProviderBlockedException(
                provider="perplexity",
                model=self.model_name,
                message="PERPLEXITY_API_KEY is not configured in .env."
            )

    async def _query_sonar(self, prompt: str) -> Dict[str, Any]:
        self._ensure_api_key()

        headers = {
            "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an elite technical recruiter and principal engineering interviewer. "
                        "Return your findings as structured technical intelligence with concrete tools, "
                        "architecture patterns, common production failures, and current interview topics."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        async def _call():
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise QuotaExhaustedException("perplexity", self.model_name, "Perplexity rate limit or quota reached (HTTP 429)")
                elif resp.status_code in (401, 403):
                    raise ProviderBlockedException("perplexity", self.model_name, f"Perplexity authentication failed: {resp.text}")
                resp.raise_for_status()
                return resp.json()

        result = await governor.execute_with_retry("perplexity", self.model_name, _call)

        usage = result.get("usage", {})
        in_tokens = usage.get("prompt_tokens", len(prompt.split()) * 2)
        out_tokens = usage.get("completion_tokens", 400)

        cost_tracker.record_usage(
            provider="perplexity",
            model=self.model_name,
            task="research",
            input_tokens=in_tokens,
            output_tokens=out_tokens
        )

        content = result["choices"][0]["message"]["content"]
        citations = result.get("citations", [])

        return {
            "content": content,
            "citations": citations
        }

    async def research_role(self, role: str) -> ResearchOutput:
        prompt = (
            f"Provide an up-to-date comprehensive interview intelligence dossier for the role: '{role}'.\n"
            "Include:\n"
            "1. Current Core Technologies & Tool Stacks (2025/2026 standards)\n"
            "2. Engineering Best Practices & Architecture Patterns\n"
            "3. Day-to-Day Responsibilities & Systems Owned\n"
            "4. Top 10 High-Yield Technical Interview Topics & Production Scenarios\n"
            "5. Industry Trends & Evolving Competencies\n"
        )
        res = await self._query_sonar(prompt)
        text = res["content"]
        citations = res.get("citations", [])

        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return ResearchOutput(
            role=role,
            technologies=[w.strip() for w in text[:300].split(",") if w.strip()][:10],
            engineering_practices=["Production Reliability", "Scalable Architecture", "Automated Testing"],
            responsibilities=[f"Design and maintain core {role} systems"],
            interview_topics=["System Architecture", "Troubleshooting", "Trade-off Decisions"],
            trends=["Cloud-native adoption", "Cost optimization", "Modern observability"],
            raw_summary=text,
            source_references=citations if citations else ["Perplexity Sonar Web Research"],
            source_hash=source_hash,
            actual_provider="perplexity" # Recorded explicitly (Adjustment #2)
        )

    async def research_company(self, company: str) -> ResearchOutput:
        prompt = (
            f"Research the engineering technology stack, system architecture patterns, and technical interview "
            f"conventions for company: '{company}'. Highlight tech stack, open-source usage, and engineering values."
        )
        res = await self._query_sonar(prompt)
        text = res["content"]
        citations = res.get("citations", [])

        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return ResearchOutput(
            role=company,
            company=company,
            raw_summary=text,
            source_references=citations if citations else [f"Perplexity Sonar Research for {company}"],
            source_hash=source_hash,
            actual_provider="perplexity"
        )
