import time
import asyncio
import random
from typing import Dict, List, Optional
from .config import config

class QuotaExhaustedException(Exception):
    """Raised when an API provider's quota / daily limit is reached."""
    def __init__(self, provider: str, model: str, message: str):
        super().__init__(f"Quota exhausted for {provider}/{model}: {message}")
        self.provider = provider
        self.model = model
        self.message = message

class ProviderBlockedException(Exception):
    """Raised when an API provider cannot run (missing credentials, forbidden, invalid config)."""
    def __init__(self, provider: str, model: str, message: str):
        super().__init__(f"Provider {provider}/{model} is blocked: {message}")
        self.provider = provider
        self.model = model
        self.message = message

class RateLimitGovernor:
    """Regulates API request cadence and handles rate limit events with jittered backoff."""
    def __init__(self):
        self._last_request_time: Dict[str, float] = {}
        # RPM limits per provider (configurable via env)
        self._rpm_limits = {
            "gemini": config.GEMINI_RPM_LIMIT,
            "perplexity": config.PERPLEXITY_RPM_LIMIT,
            "openai": config.OPENAI_RPM_LIMIT,
            "anthropic": config.ANTHROPIC_RPM_LIMIT,
            "local": 1000,
        }

    def set_rpm_limit(self, provider: str, rpm: int) -> None:
        self._rpm_limits[provider.lower()] = max(rpm, 1)

    async def throttle(self, provider: str) -> None:
        """Enforce request spacing based on provider's configured RPM."""
        provider_key = provider.lower()
        rpm = self._rpm_limits.get(provider_key, 15)
        min_interval = 60.0 / float(rpm)

        now = time.time()
        last_time = self._last_request_time.get(provider_key, 0.0)
        elapsed = now - last_time

        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            await asyncio.sleep(wait_time)

        self._last_request_time[provider_key] = time.time()

    async def execute_with_retry(self, provider: str, model: str, func, *args, max_retries: int = 3, **kwargs):
        """Execute async API call with throttling and backoff on transient 429s."""
        await self.throttle(provider)
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()

                # Check for quota exhaustion vs transient rate limit
                if "resource_exhausted" in err_msg or "quota" in err_msg or "429" in err_msg:
                    if attempt == max_retries - 1:
                        # Reached max retries on rate limit -> Mark as quota exhausted
                        raise QuotaExhaustedException(provider, model, str(e))
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"[GOVERNOR] 429/Rate limit on {provider}/{model}. Backing off {delay:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                elif "api_key" in err_msg or "unauthorized" in err_msg or "401" in err_msg or "403" in err_msg or "not set" in err_msg:
                    raise ProviderBlockedException(provider, model, str(e))
                else:
                    # Generic error
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay + random.uniform(0.2, 1.0)
                    print(f"[GOVERNOR] Transient error on {provider}/{model}: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

governor = RateLimitGovernor()
