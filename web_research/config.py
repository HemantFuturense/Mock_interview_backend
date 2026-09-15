import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

def is_valid_key(key: Optional[str]) -> bool:
    """Validate that an API key is non-empty and not a dummy/placeholder string."""
    if not key or not isinstance(key, str):
        return False
    cleaned = key.strip().lower()
    if len(cleaned) < 8:
        return False
    placeholders = ("dummy", "replace_with_real_key", "placeholder", "your_", "todo", "none")
    return not any(p in cleaned for p in placeholders)

class WebResearchConfig:
    # API Keys
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Endpoints
    TAVILY_SEARCH_URL: str = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
    FIRECRAWL_SCRAPE_URL: str = os.getenv("FIRECRAWL_SCRAPE_URL", "https://api.firecrawl.dev/v1/scrape")

    # Limits & Defaults
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    FIRECRAWL_MAX_PAGES: int = int(os.getenv("FIRECRAWL_MAX_PAGES", "3"))
    GEMINI_RESEARCH_MODEL: str = os.getenv("GEMINI_RESEARCH_MODEL", "gemini-2.5-flash")
    RESEARCH_TIMEOUT_SECONDS: float = float(os.getenv("RESEARCH_TIMEOUT_SECONDS", "60.0"))
    MAX_CONTENT_CHARS_PER_PAGE: int = int(os.getenv("MAX_CONTENT_CHARS_PER_PAGE", "6000"))

    # Cache Settings
    CACHE_TTL_SECONDS: int = int(os.getenv("RESEARCH_CACHE_TTL_SECONDS", "300"))

config = WebResearchConfig()
