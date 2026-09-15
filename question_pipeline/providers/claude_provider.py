import httpx
import json
from typing import Optional, Dict, Any
from .base import LLMProvider, LLMResult, extract_json_object
from ..config import config, is_valid_key
from ..governor import governor, ProviderBlockedException, QuotaExhaustedException
from ..cost_tracker import cost_tracker

class ClaudeLLMProvider(LLMProvider):
    """Tier 3 escalation validation LLM provider. Provider/model are fully
    environment-configurable via ESCALATION_PROVIDER / ESCALATION_MODEL."""
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.ESCALATION_MODEL
        self.api_url = "https://api.anthropic.com/v1/messages"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _ensure_api_key(self):
        if not is_valid_key(config.ANTHROPIC_API_KEY):
            raise ProviderBlockedException(
                provider="anthropic",
                model=self.model_name,
                message="ANTHROPIC_API_KEY is not configured or is a placeholder in .env for escalation validation."
            )

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "validation_escalation"
    ) -> LLMResult:
        self._ensure_api_key()

        headers = {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": f"{prompt}\n\nRespond ONLY with valid JSON."}
            ]
        }
        if system_prompt:
            payload["system"] = system_prompt

        async def _call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise QuotaExhaustedException("anthropic", self.model_name, "Anthropic rate limit or quota exceeded (429)")
                elif resp.status_code in (401, 403):
                    raise ProviderBlockedException("anthropic", self.model_name, f"Anthropic authentication failed: {resp.text}")
                resp.raise_for_status()
                return resp.json()

        result = await governor.execute_with_retry("anthropic", self.model_name, _call)

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        parsed = extract_json_object(text)

        usage = result.get("usage", {})
        in_tokens = usage.get("input_tokens", len(prompt.split()))
        out_tokens = usage.get("output_tokens", len(text.split()))

        cost_tracker.record_usage(
            provider="anthropic",
            model=self.model_name,
            task=task_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens
        )

        return LLMResult(
            text=text,
            data=parsed,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=self.model_name,
            provider="anthropic"
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "text"
    ) -> LLMResult:
        res = await self.generate_json(prompt, system_prompt, task_name)
        return res
