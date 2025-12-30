"""
Configuration management for CSM Voice Service.

Environment-based configuration with sensible defaults for development
and production-ready settings for deployment.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service Configuration
    service_name: str = "csm-voice-service"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_key: Optional[str] = None  # Set in production

    # Model Configuration
    model_path: str = "./models/csm-1b"
    model_device: str = "cuda"  # cuda, cpu, mps
    model_dtype: str = "bfloat16"  # bfloat16, float16, float32
    max_audio_length_ms: int = 30000
    default_temperature: float = 0.9
    default_topk: int = 50

    # Voice Profiles
    voice_profiles_dir: str = "./voice_profiles"
    max_context_segments: int = 5

    # Database Configuration
    # Can use Human-Like's existing database or a separate one
    # Set CSM_DATABASE_URL to your Railway PostgreSQL URL
    database_url: str = "postgresql://localhost:5432/csm_voices"

    # Redis Configuration
    # Can use Human-Like's existing Redis or a separate one
    # Set CSM_REDIS_URL to your Railway Redis URL
    redis_url: str = "redis://localhost:6379/0"

    # Integration mode: "standalone" or "humanlike"
    # In humanlike mode, uses shared database schema
    integration_mode: str = "standalone"

    # Storage Configuration (for voice model files)
    storage_backend: str = "local"  # local, s3, gcs
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    gcs_bucket: Optional[str] = None

    # Training Configuration
    training_gpu_provider: str = "runpod"  # runpod, vast, modal, local
    runpod_api_key: Optional[str] = None
    vast_api_key: Optional[str] = None
    modal_token_id: Optional[str] = None

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Caching
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 100

    # Audio Configuration
    sample_rate: int = 24000
    output_format: str = "wav"  # wav, mp3, ogg

    # Observability
    otel_endpoint: Optional[str] = None
    otel_service_name: str = "csm-voice-service"
    metrics_enabled: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "CSM_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Ensure directories exist
def init_directories():
    """Initialize required directories."""
    settings = get_settings()
    Path(settings.voice_profiles_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
