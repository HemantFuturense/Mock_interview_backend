import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .base import LLMProvider, EmbeddingProvider, LLMResult, extract_json_object
from ..config import config, is_valid_key
from ..governor import governor, ProviderBlockedException, QuotaExhaustedException
from ..cost_tracker import cost_tracker

is_valid_gemini_key = is_valid_key

class GeminiLLMProvider(LLMProvider):
    """Primary free-tier LLM provider powered by Gemini Flash-Lite."""
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.LLM_MODEL
        self._configured = False

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _ensure_configured(self):
        if not is_valid_gemini_key(config.GEMINI_API_KEY):
            raise ProviderBlockedException(
                provider="gemini",
                model=self.model_name,
                message="GEMINI_API_KEY is missing or contains dummy placeholder in .env. Please configure a valid API key."
            )
        if not self._configured:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._configured = True

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "generation"
    ) -> LLMResult:
        self._ensure_configured()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Request:\n{prompt}\n\nIMPORTANT: Return strictly valid JSON."

        async def _call():
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: model.generate_content(full_prompt))
            return response

        response = await governor.execute_with_retry("gemini", self.model_name, _call)

        text = response.text if response else "{}"
        parsed = extract_json_object(text)

        # Extract or estimate token usage
        in_tokens = int(len(full_prompt.split()) * 1.3)
        out_tokens = int(len(text.split()) * 1.3)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = response.usage_metadata.prompt_token_count or in_tokens
            out_tokens = response.usage_metadata.candidates_token_count or out_tokens

        cost_tracker.record_usage(
            provider="gemini",
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
            provider="gemini"
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "text"
    ) -> LLMResult:
        self._ensure_configured()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        async def _call():
            model = genai.GenerativeModel(self.model_name)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: model.generate_content(full_prompt))

        response = await governor.execute_with_retry("gemini", self.model_name, _call)
        text = response.text if response else ""

        in_tokens = int(len(full_prompt.split()) * 1.3)
        out_tokens = int(len(text.split()) * 1.3)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = response.usage_metadata.prompt_token_count or in_tokens
            out_tokens = response.usage_metadata.candidates_token_count or out_tokens

        cost_tracker.record_usage(
            provider="gemini",
            model=self.model_name,
            task=task_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens
        )

        return LLMResult(
            text=text,
            data=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=self.model_name,
            provider="gemini"
        )


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Configurable embedding provider via Gemini (User adjustment #1)."""
    def __init__(self, model_name: Optional[str] = None):
        # Configurable embedding model
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._configured = False
        self._verified = False

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _ensure_configured(self):
        if not is_valid_gemini_key(config.GEMINI_API_KEY):
            raise ProviderBlockedException(
                provider="gemini",
                model=self.model_name,
                message="GEMINI_API_KEY is not configured for GeminiEmbeddingProvider."
            )
        if not self._configured:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._configured = True

        # Verify embedding model availability dynamically (Adjustment #1)
        if not self._verified:
            self._verify_model()

    def _verify_model(self):
        try:
            target_name = self.model_name.replace("models/", "")
            models = [m.name.replace("models/", "") for m in genai.list_models() if "embedContent" in m.supported_generation_methods]
            if target_name in models:
                print(f"[EMBEDDING] Confirmed active Gemini embedding model: '{target_name}'")
            elif len(models) > 0:
                print(f"[EMBEDDING] Configured '{self.model_name}' not directly in active list {models}. Using active model: '{models[0]}'")
                self.model_name = models[0]
            self._verified = True
        except Exception as e:
            self._verified = True

    async def embed_texts(self, texts: List[str], task_name: str = "rag_embedding") -> List[List[float]]:
        if not texts:
            return []
        self._ensure_configured()

        clean_model = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"

        async def _call_batch():
            loop = asyncio.get_running_loop()
            results = []
            # Batch in chunks of 20 to avoid size limits
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                sub_batch = texts[i:i+batch_size]
                res = await loop.run_in_executor(
                    None,
                    lambda: genai.embed_content(
                        model=clean_model,
                        content=sub_batch,
                        task_type="retrieval_document"
                    )
                )
                embeddings = res.get("embedding", [])
                results.extend(embeddings)
            return results

        embeddings = await governor.execute_with_retry("gemini", self.model_name, _call_batch)

        total_tokens = sum(len(t.split()) for t in texts)
        cost_tracker.record_usage(
            provider="gemini",
            model=self.model_name,
            task=task_name,
            input_tokens=total_tokens,
            output_tokens=0
        )

        return embeddings

    async def embed_query(self, text: str, task_name: str = "query_embedding") -> List[float]:
        self._ensure_configured()
        clean_model = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"

        async def _call():
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=clean_model,
                    content=text,
                    task_type="retrieval_query"
                )
            )
            return res.get("embedding", [])

        embedding = await governor.execute_with_retry("gemini", self.model_name, _call)

        tokens = len(text.split())
        cost_tracker.record_usage(
            provider="gemini",
            model=self.model_name,
            task=task_name,
            input_tokens=tokens,
            output_tokens=0
        )
        return embedding
