"""
Production-Grade CSM Voice API Application Factory.

Features:
- Dependency injection
- Middleware stack (auth, rate limiting, tracing, metrics)
- Graceful shutdown
- Health checks
- Error handling
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import Settings, get_settings
from api.core.logging import get_logger, setup_logging
from api.core.metrics import get_metrics, MetricsMiddleware
from api.core.tracing import TracingMiddleware, setup_tracing
from api.core.dependencies import Container, get_container, set_container
from api.core.errors import CSMError, error_handler

from api.middleware.auth import AuthMiddleware, APIKeyValidator, JWTValidator
from api.middleware.rate_limit import RateLimitMiddleware

logger = get_logger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    testing: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override
        testing: If True, skip heavy initialization

    Returns:
        Configured FastAPI application
    """
    settings = settings or get_settings()

    # Setup logging first
    setup_logging(
        level=settings.log_level,
        json_format=not settings.debug,
    )

    # Setup tracing if enabled
    if settings.otlp_endpoint and not testing:
        setup_tracing(
            service_name="csm-voice-api",
            otlp_endpoint=settings.otlp_endpoint,
        )

    # Create app with lifespan
    app = FastAPI(
        title="CSM Voice API",
        description="Production-grade conversational speech synthesis API",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Store settings and testing flag
    app.state.settings = settings
    app.state.testing = testing

    # Add middleware (order matters - last added = first executed)
    _add_middleware(app, settings, testing)

    # Add routes
    _add_routes(app)

    # Add error handlers
    _add_error_handlers(app)

    logger.info(
        "Application created",
        extra={
            "debug": settings.debug,
            "testing": testing,
        },
    )

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = app.state.settings
    testing = app.state.testing

    # Startup
    logger.info("Starting CSM Voice API")

    # Initialize container
    container = Container()
    set_container(container)
    app.state.container = container

    if not testing:
        # Initialize services
        try:
            # Voice engine
            voice_engine = await container.get("voice_engine")
            logger.info("Voice engine initialized")

            # Training service
            training_service = await container.get("training_service")
            await training_service.initialize()
            logger.info("Training service initialized")

            # Cache
            cache = await container.get("cache")
            logger.info("Cache initialized")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            if not settings.debug:
                raise

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown")
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_handler)

    logger.info("CSM Voice API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down CSM Voice API")

    # Graceful shutdown with timeout
    try:
        await asyncio.wait_for(
            container.shutdown(),
            timeout=30.0,
        )
        logger.info("Graceful shutdown completed")
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out, forcing exit")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def _add_middleware(app: FastAPI, settings: Settings, testing: bool) -> None:
    """Add middleware stack to application."""

    # CORS (last in chain = first to process)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics
    app.add_middleware(MetricsMiddleware)

    # Tracing
    if settings.otlp_endpoint and not testing:
        app.add_middleware(TracingMiddleware)

    # Rate limiting
    if not testing:
        app.add_middleware(
            RateLimitMiddleware,
            redis_url=settings.redis_url,
            default_limit=settings.rate_limit_default,
            default_window=settings.rate_limit_window,
        )

    # Authentication
    api_key_validator = APIKeyValidator()
    if settings.api_key:
        api_key_validator.set_simple_key(settings.api_key)

    jwt_validator = None
    if settings.jwt_secret:
        jwt_validator = JWTValidator(
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
        )

    app.add_middleware(
        AuthMiddleware,
        api_key_validator=api_key_validator,
        jwt_validator=jwt_validator,
        require_auth=not settings.debug,
    )


def _add_routes(app: FastAPI) -> None:
    """Add API routes to application."""
    from api.routes import health, synthesize, profiles, training, streaming

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(synthesize.router, prefix="/v1", tags=["Synthesis"])
    app.include_router(profiles.router, prefix="/v1", tags=["Profiles"])
    app.include_router(training.router, prefix="/v1", tags=["Training"])
    app.include_router(streaming.router, prefix="/v1", tags=["Streaming"])


def _add_error_handlers(app: FastAPI) -> None:
    """Add error handlers to application."""

    @app.exception_handler(CSMError)
    async def csm_error_handler(request: Request, exc: CSMError):
        return error_handler(request, exc)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
            },
        )


# Create default app instance
app = create_app()
