"""
Comprehensive error handling with proper error hierarchy.

Production-grade error handling includes:
- Structured error responses
- Error tracking/reporting
- Correlation IDs
- Safe error messages (no internal details leaked)
"""

import traceback
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from api.core.logging import get_logger

logger = get_logger(__name__)


class CSMError(Exception):
    """Base exception for all CSM errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        internal_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        # Internal message for logging, not exposed to client
        self.internal_message = internal_message or message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(CSMError):
    """Request validation failed."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class NotFoundError(CSMError):
    """Resource not found."""

    def __init__(
        self,
        resource: str,
        resource_id: str,
    ):
        super().__init__(
            message=f"{resource} not found",
            code="NOT_FOUND",
            status_code=HTTP_404_NOT_FOUND,
            details={"resource": resource, "id": resource_id},
        )


class AuthenticationError(CSMError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=HTTP_401_UNAUTHORIZED,
        )


class AuthorizationError(CSMError):
    """Authorization failed."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=HTTP_403_FORBIDDEN,
        )


class RateLimitError(CSMError):
    """Rate limit exceeded."""

    def __init__(
        self,
        retry_after: int = 60,
        limit: int = 100,
    ):
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMIT_EXCEEDED",
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after_seconds": retry_after, "limit": limit},
        )
        self.retry_after = retry_after


class ModelError(CSMError):
    """Model inference error."""

    def __init__(
        self,
        message: str = "Model inference failed",
        internal_message: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="MODEL_ERROR",
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            internal_message=internal_message,
        )


class ModelNotLoadedError(CSMError):
    """Model not loaded."""

    def __init__(self):
        super().__init__(
            message="Model is not loaded. Please try again shortly.",
            code="MODEL_NOT_LOADED",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )


class ExternalServiceError(CSMError):
    """External service (RunPod, etc.) failed."""

    def __init__(
        self,
        service: str,
        internal_message: Optional[str] = None,
    ):
        super().__init__(
            message=f"External service unavailable: {service}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service},
            internal_message=internal_message,
        )


class CircuitBreakerOpenError(CSMError):
    """Circuit breaker is open."""

    def __init__(self, service: str):
        super().__init__(
            message=f"Service temporarily unavailable: {service}",
            code="CIRCUIT_BREAKER_OPEN",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service},
        )


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler for all exceptions.

    - Logs errors with correlation ID
    - Returns safe error messages
    - Tracks error metrics
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    # Handle our custom errors
    if isinstance(exc, CSMError):
        logger.warning(
            "Request failed",
            extra={
                "correlation_id": correlation_id,
                "error_code": exc.code,
                "error_message": exc.internal_message,
                "path": request.url.path,
                "method": request.method,
            },
        )

        response = JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

        # Add retry-after header for rate limit errors
        if isinstance(exc, RateLimitError):
            response.headers["Retry-After"] = str(exc.retry_after)

        response.headers["X-Correlation-ID"] = correlation_id
        return response

    # Handle FastAPI HTTPException
    if isinstance(exc, HTTPException):
        logger.warning(
            "HTTP exception",
            extra={
                "correlation_id": correlation_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
            },
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail,
                }
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    # Handle unexpected errors - don't leak internal details
    logger.error(
        "Unhandled exception",
        extra={
            "correlation_id": correlation_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "correlation_id": correlation_id,
            }
        },
        headers={"X-Correlation-ID": correlation_id},
    )
