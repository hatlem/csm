"""
Application configuration with Pydantic settings.

Supports:
- Environment variables
- .env files
- Validation
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # General
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # Authentication
    api_key: Optional[str] = Field(
        default=None,
        description="Simple API key for authentication",
    )
    jwt_secret: Optional[str] = Field(
        default=None,
        description="JWT signing secret",
    )
    jwt_issuer: Optional[str] = Field(
        default=None,
        description="JWT issuer",
    )

    # Rate Limiting
    rate_limit_default: int = Field(
        default=100,
        description="Default rate limit (requests per window)",
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds",
    )

    # Redis
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for caching and queues",
    )

    # Database
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL database URL",
    )

    # Model
    model_path: str = Field(
        default="./models/csm-1b",
        description="Path to CSM model",
    )
    device: str = Field(
        default="cuda",
        description="Device to run model on (cuda, cpu, mps)",
    )
    max_concurrent_inferences: int = Field(
        default=4,
        description="Maximum concurrent model inferences",
    )

    # Voice profiles
    voice_profiles_dir: str = Field(
        default="./voice_profiles",
        description="Directory for voice profile storage",
    )

    # Training
    gpu_provider: str = Field(
        default="runpod",
        description="Cloud GPU provider (runpod, modal, vast)",
    )
    runpod_api_key: Optional[str] = Field(
        default=None,
        description="RunPod API key",
    )
    modal_token_id: Optional[str] = Field(
        default=None,
        description="Modal token ID",
    )
    modal_token_secret: Optional[str] = Field(
        default=None,
        description="Modal token secret",
    )

    # Observability
    otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint",
    )
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking",
    )

    # Human-Like Integration
    humanlike_api_url: Optional[str] = Field(
        default=None,
        description="Human-Like API base URL",
    )
    humanlike_api_key: Optional[str] = Field(
        default=None,
        description="Human-Like API key",
    )

    # Twilio
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio account SID",
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio auth token",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
