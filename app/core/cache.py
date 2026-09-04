import time
from typing import Any, Dict, Optional, Tuple
from app.config.settings import config


class InMemoryCache:
    """Thread-safe in-memory cache with TTL expiration support."""

    def __init__(self, ttl: int = config.CACHE_TTL):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def remove(self, key: str) -> None:
        """Remove specific key from cache."""
        self.cache.pop(key, None)


# Initialize global caches
session_cache = InMemoryCache()
questions_cache = InMemoryCache()


def cached(cache: Optional[InMemoryCache] = None, ttl: Optional[int] = None):
    """Decorator to cache function return values using InMemoryCache."""
    import asyncio
    from functools import wraps

    target_cache = cache or InMemoryCache(ttl=ttl or config.CACHE_TTL)
    hits = 0
    misses = 0

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cached_val = target_cache.get(key)
            if cached_val is not None:
                hits += 1
                return cached_val
            misses += 1
            result = await func(*args, **kwargs)
            target_cache.set(key, result)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cached_val = target_cache.get(key)
            if cached_val is not None:
                hits += 1
                return cached_val
            misses += 1
            result = func(*args, **kwargs)
            target_cache.set(key, result)
            return result

        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        def cache_info():
            return {
                "hits": hits,
                "misses": misses,
                "size": len(target_cache.cache),
                "ttl": target_cache.ttl,
            }

        def cache_clear():
            nonlocal hits, misses
            target_cache.clear()
            hits = 0
            misses = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator

