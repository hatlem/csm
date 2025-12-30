"""
CSM Voice API Service

Production-ready voice synthesis API for Human-Like integration.
Provides voice cloning, training, and real-time synthesis capabilities.

Modules:
- main: FastAPI application and endpoints
- voice_engine: Core inference engine
- training_service: Voice training orchestration
- humanlike_client: Human-Like integration client
- twilio_streaming: Twilio bidirectional streaming
- database: PostgreSQL models and repositories
- config: Configuration management
- models: Pydantic schemas

Usage:
    # Run the API server
    python -m api.main

    # Or with uvicorn directly
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

__version__ = "1.0.0"
__author__ = "Human-Like"

from api.config import get_settings, Settings
from api.models import (
    VoiceProfileCreate,
    VoiceProfileResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TrainingJobCreate,
    TrainingJobResponse,
)

__all__ = [
    "get_settings",
    "Settings",
    "VoiceProfileCreate",
    "VoiceProfileResponse",
    "SynthesizeRequest",
    "SynthesizeResponse",
    "TrainingJobCreate",
    "TrainingJobResponse",
]
