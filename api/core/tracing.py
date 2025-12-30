"""
OpenTelemetry distributed tracing for request tracking.

Features:
- Automatic span creation for requests
- Correlation ID propagation
- Trace context in logs
- External service tracing
"""

import uuid
from typing import Optional, Callable
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.logging import set_correlation_id, get_logger

logger = get_logger(__name__)

# Try to import OpenTelemetry (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import extract, inject
    from opentelemetry.trace import SpanKind, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Tracing disabled.")


# Current span context variable
current_span_var: ContextVar[Optional[object]] = ContextVar("current_span", default=None)


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    environment: str = "development",
) -> None:
    """
    Set up OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
        environment: Deployment environment
    """
    if not OTEL_AVAILABLE:
        logger.info("OpenTelemetry not installed. Skipping tracing setup.")
        return

    if not otlp_endpoint:
        logger.info("No OTLP endpoint configured. Tracing disabled.")
        return

    # Create resource with service info
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": environment,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Create OTLP exporter
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)

    # Add batch processor
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global provider
    trace.set_tracer_provider(provider)

    logger.info(f"OpenTelemetry tracing configured: {otlp_endpoint}")


def get_tracer(name: str = "csm-voice-service"):
    """Get a tracer instance."""
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return None


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracing and correlation ID management.

    - Generates or extracts correlation ID
    - Creates request spans
    - Propagates trace context
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        # Create span if tracing is available
        if OTEL_AVAILABLE:
            tracer = get_tracer()
            if tracer:
                # Extract trace context from headers
                context = extract(request.headers)

                with tracer.start_as_current_span(
                    f"{request.method} {request.url.path}",
                    context=context,
                    kind=SpanKind.SERVER,
                ) as span:
                    # Add request attributes
                    span.set_attribute("http.method", request.method)
                    span.set_attribute("http.url", str(request.url))
                    span.set_attribute("http.route", request.url.path)
                    span.set_attribute("correlation_id", correlation_id)

                    try:
                        response = await call_next(request)

                        # Add response attributes
                        span.set_attribute("http.status_code", response.status_code)

                        if response.status_code >= 400:
                            span.set_status(Status(StatusCode.ERROR))

                        # Add correlation ID to response
                        response.headers["X-Correlation-ID"] = correlation_id

                        return response

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

        # No tracing - just pass through with correlation ID
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        except Exception:
            raise


def trace_external_call(
    service_name: str,
    operation: str,
):
    """
    Decorator to trace external service calls.

    Usage:
        @trace_external_call("runpod", "provision_gpu")
        async def provision_gpu(...):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = get_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(
                f"{service_name}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("external.service", service_name)
                span.set_attribute("external.operation", operation)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def trace_model_inference(voice_profile: str):
    """
    Decorator to trace model inference.

    Usage:
        @trace_model_inference("default")
        async def synthesize(...):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = get_tracer()
            if not tracer:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(
                "model.inference",
                kind=SpanKind.INTERNAL,
            ) as span:
                span.set_attribute("model.name", "csm-1b")
                span.set_attribute("voice.profile", voice_profile)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))

                    # Add result attributes if available
                    if hasattr(result, "duration_ms"):
                        span.set_attribute("audio.duration_ms", result.duration_ms)
                    if hasattr(result, "generation_time_ms"):
                        span.set_attribute("generation.time_ms", result.generation_time_ms)

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator
