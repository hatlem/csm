"""
Circuit Breaker pattern for fault tolerance.

Prevents cascading failures by temporarily stopping requests to
failing services and allowing them to recover.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service failing, requests are blocked
- HALF_OPEN: Testing if service recovered
"""

import time
import asyncio
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from functools import wraps

from api.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation.

    Usage:
        cb = CircuitBreaker("external-service", failure_threshold=5)

        if cb.allow_request():
            try:
                result = await external_call()
                cb.record_success()
            except Exception:
                cb.record_failure()
    """

    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout: float = 30.0

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = field(default=0)
    _success_count: int = field(default=0)
    _last_failure_time: Optional[float] = field(default=None)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time is None:
                return False

            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True

            return False

        # HALF_OPEN: Allow limited requests
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1

            if self._success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        else:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}",
            extra={
                "circuit_breaker": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "failure_count": self._failure_count,
            },
        )

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


def circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Callable[..., Any]] = None,
):
    """
    Decorator to apply circuit breaker to a function.

    Usage:
        breaker = CircuitBreaker("external-api")

        @circuit_breaker(breaker, fallback=lambda: {"status": "unavailable"})
        async def call_external_api():
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not breaker.allow_request():
                logger.warning(
                    f"Circuit breaker '{breaker.name}' is open, blocking request"
                )
                if fallback:
                    return fallback(*args, **kwargs)
                from api.core.errors import CircuitBreakerOpenError

                raise CircuitBreakerOpenError(breaker.name)

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create a new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self._breakers[name]

    def get_all_states(self) -> dict[str, str]:
        """Get states of all circuit breakers."""
        return {name: cb.state for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            cb.reset()


# Global registry
_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
