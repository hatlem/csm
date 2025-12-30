"""
Dependency Injection Container for clean architecture.

This provides:
- Centralized dependency management
- Easy testing with mocks
- Lazy initialization
- Proper lifecycle management
"""

from typing import Optional, Dict, Any, TypeVar, Type
from functools import lru_cache
from contextlib import asynccontextmanager
import asyncio

from api.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Container:
    """
    Dependency injection container.

    Usage:
        container = Container()
        container.register("voice_engine", VoiceEngine, model_path="./models")
        engine = await container.get("voice_engine")
    """

    def __init__(self):
        self._factories: Dict[str, tuple] = {}
        self._instances: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._initialized = False

    def register(
        self,
        name: str,
        factory: Type[T],
        singleton: bool = True,
        **kwargs,
    ) -> None:
        """
        Register a dependency.

        Args:
            name: Dependency name
            factory: Class or factory function
            singleton: If True, only one instance is created
            **kwargs: Arguments to pass to factory
        """
        self._factories[name] = (factory, singleton, kwargs)
        self._locks[name] = asyncio.Lock()

    async def get(self, name: str) -> Any:
        """
        Get a dependency instance.

        Args:
            name: Dependency name

        Returns:
            The dependency instance
        """
        if name not in self._factories:
            raise KeyError(f"Dependency not registered: {name}")

        factory, singleton, kwargs = self._factories[name]

        # Return cached instance for singletons
        if singleton and name in self._instances:
            return self._instances[name]

        # Thread-safe instance creation
        async with self._locks[name]:
            # Double-check after acquiring lock
            if singleton and name in self._instances:
                return self._instances[name]

            # Create instance
            logger.info(f"Creating dependency: {name}")
            instance = factory(**kwargs)

            # Handle async initialization
            if hasattr(instance, "initialize"):
                await instance.initialize()

            if singleton:
                self._instances[name] = instance

            return instance

    def get_sync(self, name: str) -> Any:
        """Get dependency synchronously (for non-async contexts)."""
        if name in self._instances:
            return self._instances[name]
        raise RuntimeError(f"Dependency not initialized: {name}. Call get() first.")

    async def initialize_all(self) -> None:
        """Initialize all registered dependencies."""
        if self._initialized:
            return

        logger.info("Initializing all dependencies...")

        for name in self._factories:
            await self.get(name)

        self._initialized = True
        logger.info("All dependencies initialized")

    async def shutdown(self) -> None:
        """Shutdown all dependencies."""
        logger.info("Shutting down dependencies...")

        for name, instance in self._instances.items():
            if hasattr(instance, "shutdown"):
                logger.info(f"Shutting down: {name}")
                await instance.shutdown()
            elif hasattr(instance, "close"):
                await instance.close()

        self._instances.clear()
        self._initialized = False
        logger.info("All dependencies shut down")

    def override(self, name: str, instance: Any) -> None:
        """
        Override a dependency with a specific instance (for testing).

        Args:
            name: Dependency name
            instance: Instance to use
        """
        self._instances[name] = instance

    def clear(self) -> None:
        """Clear all instances (for testing)."""
        self._instances.clear()
        self._initialized = False


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
        _setup_default_dependencies(_container)
    return _container


def set_container(container: Container) -> None:
    """Set the global container instance."""
    global _container
    _container = container


def _setup_default_dependencies(container: Container) -> None:
    """Setup default dependencies in the container."""
    from api.core.config import get_settings
    from api.services.voice_engine import VoiceEngineService
    from api.services.training import TrainingService
    from api.services.cache import CacheService

    settings = get_settings()

    # Voice engine - core inference
    container.register(
        "voice_engine",
        VoiceEngineService,
        model_path=settings.model_path,
        device=settings.device,
    )

    # Training service
    container.register(
        "training_service",
        TrainingService,
        storage_dir=settings.voice_profiles_dir,
        gpu_provider=settings.gpu_provider,
        runpod_api_key=settings.runpod_api_key,
        redis_url=settings.redis_url,
    )

    # Cache service
    container.register(
        "cache",
        CacheService,
        redis_url=settings.redis_url,
    )


def setup_container(settings) -> Container:
    """
    Set up the dependency container with all services.

    Args:
        settings: Application settings

    Returns:
        Configured container
    """
    from api.services.voice_engine import VoiceEngineService
    from api.services.training import TrainingService
    from api.services.queue import JobQueue
    from api.services.cache import CacheService

    container = get_container()

    # Voice engine - core inference
    container.register(
        "voice_engine",
        VoiceEngineService,
        model_path=settings.model_path,
        device=settings.model_device,
        dtype=settings.model_dtype,
    )

    # Training service
    container.register(
        "training_service",
        TrainingService,
        storage_dir=settings.voice_profiles_dir,
        gpu_provider=settings.training_gpu_provider,
    )

    # Job queue
    container.register(
        "job_queue",
        JobQueue,
        redis_url=settings.redis_url,
    )

    # Cache service
    container.register(
        "cache",
        CacheService,
        redis_url=settings.redis_url,
        ttl=settings.cache_ttl_seconds,
    )

    return container


@asynccontextmanager
async def lifespan_context(app):
    """
    Application lifespan context manager.

    Handles startup and shutdown of all dependencies.
    """
    from api.config import get_settings
    from api.core.logging import setup_logging
    from api.core.tracing import setup_tracing
    from api.core.metrics import get_metrics

    settings = get_settings()

    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_format=settings.environment == "production",
        service_name=settings.service_name,
    )

    logger.info("Starting CSM Voice Service...")

    # Setup tracing
    setup_tracing(
        service_name=settings.service_name,
        otlp_endpoint=settings.otel_endpoint,
        environment=settings.environment,
    )

    # Setup metrics
    get_metrics().set_service_info(
        version="1.0.0",
        environment=settings.environment,
    )

    # Initialize container
    container = setup_container(settings)

    # Pre-load critical dependencies
    if settings.environment == "production":
        await container.initialize_all()

    yield

    # Shutdown
    logger.info("Shutting down CSM Voice Service...")
    await container.shutdown()
