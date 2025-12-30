"""
Health check endpoints.

Provides:
- Liveness probe
- Readiness probe
- Detailed health status
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.core.dependencies import get_container
from api.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "csm-voice-api",
        "version": "1.0.0",
        "docs": "/docs",
    }


@router.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns service health status for load balancers and k8s probes.
    """
    container = get_container()

    health_info = {
        "status": "healthy",
        "version": "1.0.0",
        "checks": {},
    }

    # Check voice engine
    try:
        voice_engine = await container.get("voice_engine")
        engine_health = voice_engine.get_health_info()
        health_info["checks"]["voice_engine"] = {
            "status": "healthy" if engine_health.get("model_loaded") else "degraded",
            "details": engine_health,
        }
    except Exception as e:
        health_info["checks"]["voice_engine"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_info["status"] = "degraded"

    # Check cache/redis
    try:
        cache = await container.get("cache")
        if cache._redis:
            await cache._redis.ping()
            health_info["checks"]["redis"] = {"status": "healthy"}
        else:
            health_info["checks"]["redis"] = {"status": "not_configured"}
    except Exception as e:
        health_info["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Determine overall status
    statuses = [c.get("status") for c in health_info["checks"].values()]
    if "unhealthy" in statuses:
        health_info["status"] = "unhealthy"
        status_code = 503
    elif "degraded" in statuses:
        health_info["status"] = "degraded"
        status_code = 200
    else:
        status_code = 200

    return JSONResponse(content=health_info, status_code=status_code)


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(request: Request):
    """Kubernetes readiness probe."""
    container = get_container()

    try:
        voice_engine = await container.get("voice_engine")
        if voice_engine.is_loaded:
            return {"status": "ready"}
        else:
            return JSONResponse(
                content={"status": "not_ready", "reason": "model_loading"},
                status_code=503,
            )
    except Exception as e:
        return JSONResponse(
            content={"status": "not_ready", "reason": str(e)},
            status_code=503,
        )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
