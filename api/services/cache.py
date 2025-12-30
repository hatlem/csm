"""
Redis-based caching service.

Features:
- Distributed cache for multi-instance deployments
- TTL-based expiration
- Cache invalidation
- Metrics tracking
"""

import json
import hashlib
from typing import Optional, Any, TypeVar, Callable
from functools import wraps

from api.core.logging import get_logger
from api.core.metrics import get_metrics

logger = get_logger(__name__)

T = TypeVar("T")


class CacheService:
    """
    Redis-based cache service.

    Usage:
        cache = CacheService(redis_url="redis://localhost:6379")
        await cache.initialize()

        # Simple get/set
        await cache.set("key", {"data": "value"}, ttl=3600)
        data = await cache.get("key")

        # With decorator
        @cache.cached(ttl=300)
        async def expensive_operation(param):
            ...
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        prefix: str = "csm:cache:",
    ):
        self.redis_url = redis_url
        self.default_ttl = ttl
        self.prefix = prefix
        self._redis = None
        self._local_cache: dict = {}  # Fallback
        self._metrics = get_metrics()

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Cache service connected to Redis")
        except ImportError:
            logger.warning("redis package not installed, using in-memory cache")
            self._redis = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for cache: {e}")
            self._redis = None

    async def shutdown(self) -> None:
        """Shutdown the cache service."""
        if self._redis:
            await self._redis.close()

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        full_key = self._key(key)

        if self._redis:
            try:
                value = await self._redis.get(full_key)
                if value:
                    self._metrics.record_cache_access("redis", hit=True)
                    return json.loads(value)
                self._metrics.record_cache_access("redis", hit=False)
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        else:
            if full_key in self._local_cache:
                self._metrics.record_cache_access("local", hit=True)
                return self._local_cache[full_key]
            self._metrics.record_cache_access("local", hit=False)

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        full_key = self._key(key)
        ttl = ttl or self.default_ttl

        if self._redis:
            try:
                await self._redis.set(
                    full_key,
                    json.dumps(value, default=str),
                    ex=ttl,
                )
            except Exception as e:
                logger.warning(f"Cache set error: {e}")
        else:
            self._local_cache[full_key] = value

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        full_key = self._key(key)

        if self._redis:
            try:
                await self._redis.delete(full_key)
            except Exception as e:
                logger.warning(f"Cache delete error: {e}")
        else:
            self._local_cache.pop(full_key, None)

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        full_pattern = self._key(pattern)
        count = 0

        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=full_pattern,
                        count=100,
                    )
                    if keys:
                        await self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Cache clear pattern error: {e}")
        else:
            to_delete = [k for k in self._local_cache if k.startswith(full_pattern.replace("*", ""))]
            for k in to_delete:
                del self._local_cache[k]
            count = len(to_delete)

        logger.info(f"Cleared {count} cache keys matching {pattern}")
        return count

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = self._key(key)

        if self._redis:
            try:
                return await self._redis.exists(full_key) > 0
            except Exception:
                return False
        else:
            return full_key in self._local_cache

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        if not keys:
            return {}

        result = {}
        full_keys = [self._key(k) for k in keys]

        if self._redis:
            try:
                values = await self._redis.mget(full_keys)
                for key, value in zip(keys, values):
                    if value:
                        result[key] = json.loads(value)
            except Exception as e:
                logger.warning(f"Cache get_many error: {e}")
        else:
            for key in keys:
                if self._key(key) in self._local_cache:
                    result[key] = self._local_cache[self._key(key)]

        return result

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Set multiple values in cache."""
        ttl = ttl or self.default_ttl

        if self._redis:
            try:
                pipe = self._redis.pipeline()
                for key, value in items.items():
                    pipe.set(
                        self._key(key),
                        json.dumps(value, default=str),
                        ex=ttl,
                    )
                await pipe.execute()
            except Exception as e:
                logger.warning(f"Cache set_many error: {e}")
        else:
            for key, value in items.items():
                self._local_cache[self._key(key)] = value

    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
    ) -> Callable:
        """
        Decorator to cache function results.

        Usage:
            @cache.cached(ttl=300)
            async def get_user(user_id: str):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Generate cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:16]

                # Check cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Call function
                result = await func(*args, **kwargs)

                # Cache result
                await self.set(cache_key, result, ttl=ttl)

                return result

            return wrapper

        return decorator


class AudioCache:
    """
    Specialized cache for audio data.

    Audio is stored separately due to size and different access patterns.
    """

    def __init__(
        self,
        cache: CacheService,
        max_size_mb: int = 500,
    ):
        self.cache = cache
        self.max_size_mb = max_size_mb
        self._size_bytes = 0

    def _audio_key(
        self,
        voice_profile_id: str,
        text: str,
        params: dict,
    ) -> str:
        """Generate cache key for audio."""
        key_data = f"{voice_profile_id}:{text}:{sorted(params.items())}"
        return f"audio:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"

    async def get(
        self,
        voice_profile_id: str,
        text: str,
        params: dict,
    ) -> Optional[bytes]:
        """Get cached audio."""
        import base64

        key = self._audio_key(voice_profile_id, text, params)
        data = await self.cache.get(key)

        if data and "audio_b64" in data:
            return base64.b64decode(data["audio_b64"])
        return None

    async def set(
        self,
        voice_profile_id: str,
        text: str,
        params: dict,
        audio_bytes: bytes,
        duration_ms: int,
        ttl: int = 3600,
    ) -> None:
        """Cache audio."""
        import base64

        key = self._audio_key(voice_profile_id, text, params)

        await self.cache.set(
            key,
            {
                "audio_b64": base64.b64encode(audio_bytes).decode(),
                "duration_ms": duration_ms,
                "voice_profile_id": voice_profile_id,
            },
            ttl=ttl,
        )
