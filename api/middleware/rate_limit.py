"""
Rate limiting middleware using sliding window algorithm.

Features:
- Per-client rate limiting
- Configurable limits per endpoint
- Redis-backed for distributed deployments
- Bypass for internal services
"""

import time
from typing import Optional, Callable, Dict
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.logging import get_logger
from api.core.errors import RateLimitError

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    burst: int = 0  # Extra burst allowance


class RateLimiter:
    """
    Sliding window rate limiter.

    Uses Redis for distributed rate limiting, falls back to in-memory.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        prefix: str = "csm:ratelimit:",
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._redis = None
        self._local_counts: Dict[str, list] = {}

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not self.redis_url:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Rate limiter Redis connection failed: {e}")
            self._redis = None

    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Returns:
            (allowed, remaining, reset_time)
        """
        now = time.time()
        window_start = now - config.window_seconds
        full_key = f"{self.prefix}{key}"

        if self._redis:
            return await self._check_redis(full_key, config, now, window_start)
        else:
            return self._check_local(full_key, config, now, window_start)

    async def _check_redis(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        window_start: float,
    ) -> tuple[bool, int, int]:
        """Check rate limit using Redis."""
        pipe = self._redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiry
        pipe.expire(key, config.window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        max_requests = config.requests + config.burst
        remaining = max(0, max_requests - current_count)
        reset_time = int(now + config.window_seconds)

        allowed = current_count < max_requests

        if not allowed:
            # Remove the request we just added
            await self._redis.zrem(key, str(now))

        return allowed, remaining, reset_time

    def _check_local(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        window_start: float,
    ) -> tuple[bool, int, int]:
        """Check rate limit using local memory."""
        if key not in self._local_counts:
            self._local_counts[key] = []

        # Remove old entries
        self._local_counts[key] = [
            t for t in self._local_counts[key] if t > window_start
        ]

        current_count = len(self._local_counts[key])
        max_requests = config.requests + config.burst
        remaining = max(0, max_requests - current_count)
        reset_time = int(now + config.window_seconds)

        if current_count < max_requests:
            self._local_counts[key].append(now)
            return True, remaining - 1, reset_time
        else:
            return False, 0, reset_time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Usage:
        app.add_middleware(
            RateLimitMiddleware,
            default_config=RateLimitConfig(requests=100, window_seconds=60),
            redis_url="redis://localhost:6379",
        )
    """

    def __init__(
        self,
        app,
        default_config: RateLimitConfig = RateLimitConfig(requests=100, window_seconds=60),
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None,
        redis_url: Optional[str] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.default_config = default_config
        self.endpoint_configs = endpoint_configs or {}
        self.limiter = RateLimiter(redis_url=redis_url)
        self.key_func = key_func or self._default_key_func
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self._initialized = False

    def _default_key_func(self, request: Request) -> str:
        """Default key function: use client IP."""
        # Get real IP if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"{ip}:{request.url.path}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Initialize limiter on first request
        if not self._initialized:
            await self.limiter.initialize()
            self._initialized = True

        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Get rate limit config for endpoint
        config = self.endpoint_configs.get(request.url.path, self.default_config)

        # Get rate limit key
        key = self.key_func(request)

        # Check rate limit
        allowed, remaining, reset_time = await self.limiter.check_rate_limit(key, config)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "path": request.url.path,
                    "key": key,
                    "limit": config.requests,
                },
            )
            raise RateLimitError(
                retry_after=reset_time - int(time.time()),
                limit=config.requests,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(config.requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response
