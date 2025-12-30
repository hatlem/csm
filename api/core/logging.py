"""
Structured JSON logging for production observability.

Features:
- JSON formatted logs for log aggregation (ELK, Datadog, etc.)
- Correlation ID propagation
- Request/response logging
- Performance timing
"""

import logging
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps

# Context variable for correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(""),
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add location info
        log_data["location"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data, default=str)


class StructuredLogger(logging.Logger):
    """Logger with structured logging support."""

    def _log(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if extra is None:
            extra = {}

        # Create a new LogRecord with extra data
        extra_data = {"extra": extra}
        super()._log(level, msg, args, exc_info=exc_info, extra=extra_data, **kwargs)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "csm-voice-service",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (True for production)
        service_name: Service name for log aggregation
    """
    # Set the custom logger class
    logging.setLoggerClass(StructuredLogger)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.addHandler(handler)

    # Reduce noise from libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id_var.get("")


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"{func.__name__} completed",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time, 2),
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time, 2),
                        "error": str(e),
                    },
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"{func.__name__} completed",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time, 2),
                    },
                )
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time, 2),
                        "error": str(e),
                    },
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
