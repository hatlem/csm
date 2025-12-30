"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_voice_engine():
    """Mock voice engine for testing without GPU."""
    import torch

    engine = Mock()
    engine.is_loaded = True
    engine.sample_rate = 24000

    # Mock synthesis result
    mock_audio = torch.zeros(24000)  # 1 second of silence

    async def mock_synthesize(*args, **kwargs):
        from api.services.voice_engine import SynthesisResult

        return SynthesisResult(
            audio=mock_audio,
            sample_rate=24000,
            duration_ms=1000,
            generation_time_ms=100,
        )

    engine.synthesize = AsyncMock(side_effect=mock_synthesize)
    engine.get_health_info = Mock(return_value={
        "model_loaded": True,
        "device": "cpu",
        "circuit_breaker_state": "closed",
    })

    return engine


@pytest.fixture
def mock_training_service():
    """Mock training service."""
    from api.services.training import TrainingJob

    service = Mock()

    async def mock_start_training(*args, **kwargs):
        return TrainingJob(
            id="test-job-123",
            voice_profile_id=kwargs.get("voice_profile_id", "test-profile"),
            status="queued",
        )

    service.start_training_job = AsyncMock(side_effect=mock_start_training)
    service.get_job = Mock(return_value=None)
    service.get_training_stats = Mock(return_value={
        "sample_count": 100,
        "total_duration_hours": 1.5,
        "ready_for_training": True,
    })

    return service


@pytest.fixture
def mock_container(mock_voice_engine, mock_training_service):
    """Mock dependency container."""
    from api.core.dependencies import Container

    container = Container()
    container.override("voice_engine", mock_voice_engine)
    container.override("training_service", mock_training_service)

    return container


@pytest.fixture
def app(mock_container):
    """Create test FastAPI app with mocked dependencies."""
    from api.app import create_app
    from api.core.dependencies import get_container

    # Override global container
    with patch("api.core.dependencies.get_container", return_value=mock_container):
        app = create_app(testing=True)
        yield app


@pytest.fixture
def client(app) -> Generator:
    """Sync test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app) -> AsyncGenerator:
    """Async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def api_key():
    """Test API key."""
    return "test-api-key-12345"


@pytest.fixture
def auth_headers(api_key):
    """Headers with API key."""
    return {"X-API-Key": api_key}
