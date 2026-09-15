import httpx
import json
from typing import Optional, Dict, Any
from .base import LLMProvider, LLMResult, extract_json_object
from ..config import config, is_valid_key
from ..governor import governor, ProviderBlockedException, QuotaExhaustedException
from ..cost_tracker import cost_tracker

class OpenAILLMProvider(LLMProvider):
    """Tier 2 fallback validation LLM provider. Provider/model are fully
    environment-configurable via FALLBACK_LLM_PROVIDER / FALLBACK_LLM_MODEL
    (see providers/factory.py and config.py) -- nothing here is hardcoded
    beyond the default model name used when FALLBACK_LLM_MODEL is unset."""
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.FALLBACK_LLM_MODEL
        self.api_url = "https://api.openai.com/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "openai"

    def _ensure_api_key(self):
        # Uses the same shared placeholder-detection as every other provider
        # (config.is_valid_key) rather than a narrower ad-hoc check, so a
        # value like "sk-your_key_here" is caught here too, not just a
        # literal "dummy" substring.
        if not is_valid_key(config.OPENAI_API_KEY):
            raise ProviderBlockedException(
                provider="openai",
                model=self.model_name,
                message="OPENAI_API_KEY is not configured or is a placeholder in .env for fallback validation."
            )

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "validation_fallback"
    ) -> LLMResult:
        self._ensure_api_key()

        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        async def _call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise QuotaExhaustedException("openai", self.model_name, "OpenAI rate limit or quota exceeded (429)")
                elif resp.status_code in (401, 403):
                    raise ProviderBlockedException("openai", self.model_name, f"OpenAI authentication failed: {resp.text}")
                resp.raise_for_status()
                return resp.json()

        result = await governor.execute_with_retry("openai", self.model_name, _call)

        text = result["choices"][0]["message"]["content"]
        try:
            parsed = extract_json_object(text)
        except json.JSONDecodeError:
            parsed = {}

        usage = result.get("usage", {})
        in_tokens = usage.get("prompt_tokens", len(prompt.split()))
        out_tokens = usage.get("completion_tokens", len(text.split()))

        cost_tracker.record_usage(
            provider="openai",
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
            provider="openai"
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "text"
    ) -> LLMResult:
        res = await self.generate_json(prompt, system_prompt, task_name)
        return res
