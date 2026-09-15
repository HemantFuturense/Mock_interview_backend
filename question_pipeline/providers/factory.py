import os
from .base import ResearchProvider, LLMProvider, EmbeddingProvider
from ..config import config
from ..governor import ProviderBlockedException

from .gemini_provider import GeminiLLMProvider, GeminiEmbeddingProvider, is_valid_gemini_key
from .perplexity_provider import PerplexityResearchProvider
from .tavily_firecrawl_provider import TavilyFirecrawlResearchProvider
from .openai_provider import OpenAILLMProvider
from .claude_provider import ClaudeLLMProvider
from .local_provider import LocalEmbeddingProvider
from .mock_provider import MockResearchProvider, MockLLMProvider, MockEmbeddingProvider

def get_research_provider() -> ResearchProvider:
    """Instantiate configured web research provider."""
    if config.MOCK_MODE:
        return MockResearchProvider()

    provider_name = config.RESEARCH_PROVIDER.lower()
    if provider_name == "perplexity":
        return PerplexityResearchProvider()
    elif provider_name == "tavily_firecrawl":
        return TavilyFirecrawlResearchProvider()
    elif provider_name == "gemini":
        # In case Gemini web search / grounding is configured in future
        raise ProviderBlockedException("gemini", config.RESEARCH_MODEL, "Gemini web grounding provider not yet enabled in config.")
    else:
        raise ProviderBlockedException(
            provider_name,
            config.RESEARCH_MODEL,
            f"Unsupported RESEARCH_PROVIDER: '{provider_name}'. Please configure 'perplexity' or 'tavily_firecrawl'."
        )

def get_llm_provider(provider_override: str = None) -> LLMProvider:
    """Instantiate configured LLM provider for question generation and synthesis."""
    if config.MOCK_MODE:
        return MockLLMProvider(provider_override or "gemini")

    provider_name = (provider_override or config.LLM_PROVIDER).lower()
    if provider_name == "gemini":
        return GeminiLLMProvider()
    elif provider_name == "openai":
        return OpenAILLMProvider()
    elif provider_name == "anthropic":
        return ClaudeLLMProvider()
    else:
        return GeminiLLMProvider()

def get_embedding_provider() -> EmbeddingProvider:
    """Instantiate configured embedding provider (Gemini or Local)."""
    if config.MOCK_MODE:
        return LocalEmbeddingProvider()

    provider_name = config.EMBEDDING_PROVIDER.lower()
    if provider_name == "gemini":
        return GeminiEmbeddingProvider(model_name=config.EMBEDDING_MODEL)
    elif provider_name == "local":
        return LocalEmbeddingProvider()
    else:
        raise ProviderBlockedException(
            provider=provider_name,
            model=config.EMBEDDING_MODEL,
            message=f"Unsupported EMBEDDING_PROVIDER: '{provider_name}'. Please configure 'gemini' or 'local'."
        )

def get_validation_provider():
    """Instantiate configured validation LLM provider."""
    return get_llm_provider(config.VALIDATION_PROVIDER)

