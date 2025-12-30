"""
Prometheus metrics for monitoring and alerting.

Metrics include:
- Request latency histograms
- Request counts by endpoint/status
- Model inference timing
- Queue depths
- GPU memory usage
- Cache hit rates
"""

import time
from typing import Callable, Optional
from functools import wraps
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Create a custom registry
REGISTRY = CollectorRegistry()

# =============================================================================
# Request Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "csm_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "csm_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

ACTIVE_REQUESTS = Gauge(
    "csm_active_requests",
    "Number of active requests",
    registry=REGISTRY,
)

# =============================================================================
# Model Metrics
# =============================================================================

MODEL_LOADED = Gauge(
    "csm_model_loaded",
    "Whether the model is loaded (1) or not (0)",
    registry=REGISTRY,
)

MODEL_LOAD_TIME = Gauge(
    "csm_model_load_time_seconds",
    "Time taken to load the model",
    registry=REGISTRY,
)

SYNTHESIS_LATENCY = Histogram(
    "csm_synthesis_duration_seconds",
    "Speech synthesis latency",
    ["voice_profile"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

SYNTHESIS_COUNT = Counter(
    "csm_synthesis_total",
    "Total synthesis requests",
    ["voice_profile", "status"],
    registry=REGISTRY,
)

AUDIO_DURATION = Histogram(
    "csm_audio_duration_seconds",
    "Generated audio duration",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

REAL_TIME_FACTOR = Histogram(
    "csm_real_time_factor",
    "Synthesis speed (generation_time / audio_duration)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY,
)

# =============================================================================
# GPU Metrics
# =============================================================================

GPU_MEMORY_USED = Gauge(
    "csm_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    registry=REGISTRY,
)

GPU_MEMORY_TOTAL = Gauge(
    "csm_gpu_memory_total_bytes",
    "GPU total memory in bytes",
    registry=REGISTRY,
)

GPU_UTILIZATION = Gauge(
    "csm_gpu_utilization_percent",
    "GPU utilization percentage",
    registry=REGISTRY,
)

# =============================================================================
# Cache Metrics
# =============================================================================

CACHE_HITS = Counter(
    "csm_cache_hits_total",
    "Cache hit count",
    ["cache_type"],
    registry=REGISTRY,
)

CACHE_MISSES = Counter(
    "csm_cache_misses_total",
    "Cache miss count",
    ["cache_type"],
    registry=REGISTRY,
)

CACHE_SIZE = Gauge(
    "csm_cache_size",
    "Number of items in cache",
    ["cache_type"],
    registry=REGISTRY,
)

# =============================================================================
# Queue Metrics
# =============================================================================

QUEUE_SIZE = Gauge(
    "csm_queue_size",
    "Number of items in queue",
    ["queue_name"],
    registry=REGISTRY,
)

QUEUE_PROCESSING_TIME = Histogram(
    "csm_queue_processing_seconds",
    "Time to process queue items",
    ["queue_name"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    registry=REGISTRY,
)

# =============================================================================
# Training Metrics
# =============================================================================

TRAINING_JOBS_TOTAL = Counter(
    "csm_training_jobs_total",
    "Total training jobs",
    ["status"],
    registry=REGISTRY,
)

TRAINING_JOBS_ACTIVE = Gauge(
    "csm_training_jobs_active",
    "Number of active training jobs",
    registry=REGISTRY,
)

# =============================================================================
# Service Info
# =============================================================================

SERVICE_INFO = Info(
    "csm_service",
    "Service information",
    registry=REGISTRY,
)


class MetricsCollector:
    """Centralized metrics collection."""

    def __init__(self):
        self._registry = REGISTRY

    def set_service_info(self, version: str, environment: str):
        """Set service information."""
        SERVICE_INFO.info({
            "version": version,
            "environment": environment,
        })

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
    ):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

    def record_synthesis(
        self,
        voice_profile: str,
        duration_seconds: float,
        audio_duration_seconds: float,
        success: bool,
    ):
        """Record synthesis metrics."""
        status = "success" if success else "error"
        SYNTHESIS_COUNT.labels(voice_profile=voice_profile, status=status).inc()

        if success:
            SYNTHESIS_LATENCY.labels(voice_profile=voice_profile).observe(duration_seconds)
            AUDIO_DURATION.observe(audio_duration_seconds)

            if audio_duration_seconds > 0:
                rtf = duration_seconds / audio_duration_seconds
                REAL_TIME_FACTOR.observe(rtf)

    def set_model_loaded(self, loaded: bool, load_time: Optional[float] = None):
        """Set model loaded status."""
        MODEL_LOADED.set(1 if loaded else 0)
        if load_time is not None:
            MODEL_LOAD_TIME.set(load_time)

    def set_gpu_metrics(
        self,
        memory_used: int,
        memory_total: int,
        utilization: float,
    ):
        """Set GPU metrics."""
        GPU_MEMORY_USED.set(memory_used)
        GPU_MEMORY_TOTAL.set(memory_total)
        GPU_UTILIZATION.set(utilization)

    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache hit/miss."""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def set_cache_size(self, cache_type: str, size: int):
        """Set cache size."""
        CACHE_SIZE.labels(cache_type=cache_type).set(size)

    def set_queue_size(self, queue_name: str, size: int):
        """Set queue size."""
        QUEUE_SIZE.labels(queue_name=queue_name).set(size)

    def record_queue_processing(self, queue_name: str, duration: float):
        """Record queue processing time."""
        QUEUE_PROCESSING_TIME.labels(queue_name=queue_name).observe(duration)

    def export(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest(self._registry)

    def content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics collector
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        ACTIVE_REQUESTS.inc()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            get_metrics().record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration,
            )

            return response

        except Exception as e:
            duration = time.perf_counter() - start_time
            get_metrics().record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                duration=duration,
            )
            raise

        finally:
            ACTIVE_REQUESTS.dec()
