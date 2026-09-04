import asyncio
import inspect
from typing import Any, Dict, List, Optional
import google.generativeai as genai

from app.config.constants import FALLBACK_GEMINI_MODELS
from app.config.settings import config
from app.core.logger import logger

# Configure Gemini globally
genai.configure(api_key=config.GEMINI_API_KEY)


async def execute_with_retries(
    func: Any,
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_label: str = "Gemini call",
    **kwargs: Any
) -> Any:
    """Execute a callable with retry support and exponential backoff."""
    for attempt in range(max_retries):
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            logger.warning(f"{retry_label} attempt {attempt + 1} failed: {exc}")
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (backoff_factor ** attempt))


async def generate_content_with_fallback(
    prompt: Any,
    request_options: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    retry_label: str = "Gemini call",
    models: Optional[List[str]] = None,
) -> Any:
    """Try primary and fallback Gemini models automatically when 429 quota or rate limit occurs.

    `models`, when given, is tried in order before falling back to FALLBACK_GEMINI_MODELS -
    lets a caller (e.g. video analysis) put its own preferred model first while still
    getting the shared retry/fallback protection.
    """
    if request_options is None:
        request_options = {"timeout": 120}
    last_exc = None
    for model_name in (models or FALLBACK_GEMINI_MODELS):
        try:
            return await execute_with_retries(
                lambda m=model_name: genai.GenerativeModel(m).generate_content(
                    prompt,
                    request_options=request_options,
                    generation_config=generation_config,
                ),
                retry_label=f"{retry_label} ({model_name})",
                max_retries=2,
                base_delay=0.5,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"Model {model_name} quota/rate limit exceeded or failed during {retry_label}: {exc}. Trying next fallback model..."
            )
            continue
    raise last_exc or RuntimeError("All fallback Gemini models failed")


# Alias for backward compatibility across modules
_generate_content_with_fallback = generate_content_with_fallback


async def embed_text(text: str, task_type: str = "retrieval_query") -> list[float]:
    """Create a Gemini embedding for RAG retrieval."""
    async def _call() -> Any:
        return await asyncio.to_thread(
            genai.embed_content,
            model=config.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
            output_dimensionality=768,
        )

    result = await execute_with_retries(
        _call,
        max_retries=6,
        base_delay=8.0,
        backoff_factor=2.0,
        retry_label="Gemini embed_content",
    )
    return result["embedding"]
