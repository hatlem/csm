"""
Database models and schema for voice profile management.

Uses SQLAlchemy with async support for PostgreSQL.
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Any, Dict
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SQLEnum, create_engine, Index,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from api.models import VoiceProfileStatus, TrainingStatus

Base = declarative_base()


# =============================================================================
# Models
# =============================================================================

class VoiceProfile(Base):
    """Voice profile database model."""

    __tablename__ = "voice_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4())[:12])
    name = Column(String(255), nullable=False)
    agent_id = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Status
    status = Column(
        SQLEnum(VoiceProfileStatus),
        default=VoiceProfileStatus.PENDING,
        nullable=False,
    )

    # Training data stats
    sample_count = Column(Integer, default=0)
    total_duration_ms = Column(Integer, default=0)

    # Model artifacts
    model_path = Column(String(500), nullable=True)
    lora_path = Column(String(500), nullable=True)
    prompt_audio_path = Column(String(500), nullable=True)
    prompt_text = Column(Text, nullable=True)

    # Configuration
    config = Column(JSONB, default=dict)
    metadata_ = Column("metadata", JSONB, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_samples = relationship("TrainingSample", back_populates="voice_profile")
    training_jobs = relationship("TrainingJob", back_populates="voice_profile")

    # Indexes
    __table_args__ = (
        Index("ix_voice_profiles_agent_status", "agent_id", "status"),
    )


class TrainingSample(Base):
    """Training sample database model."""

    __tablename__ = "training_samples"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    voice_profile_id = Column(
        String(36),
        ForeignKey("voice_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Audio data
    audio_path = Column(String(500), nullable=False)
    transcript = Column(Text, nullable=False)
    duration_ms = Column(Integer, nullable=False)

    # Quality metrics
    snr_db = Column(Float, nullable=True)  # Signal-to-noise ratio
    clipping_detected = Column(Boolean, default=False)

    # Status
    processed = Column(Boolean, default=False)
    validation_errors = Column(JSONB, default=list)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    voice_profile = relationship("VoiceProfile", back_populates="training_samples")


class TrainingJob(Base):
    """Training job database model."""

    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4())[:12])
    voice_profile_id = Column(
        String(36),
        ForeignKey("voice_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Status
    status = Column(
        SQLEnum(TrainingStatus),
        default=TrainingStatus.QUEUED,
        nullable=False,
    )
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # Configuration
    config = Column(JSONB, default=dict)
    gpu_type = Column(String(50), default="a100")

    # Cloud instance info
    cloud_provider = Column(String(50), nullable=True)
    cloud_instance_id = Column(String(255), nullable=True)
    cloud_instance_ip = Column(String(50), nullable=True)

    # Costs
    estimated_cost_usd = Column(Float, nullable=True)
    actual_cost_usd = Column(Float, nullable=True)

    # Results
    output_path = Column(String(500), nullable=True)
    metrics = Column(JSONB, default=dict)
    logs_url = Column(String(500), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    voice_profile = relationship("VoiceProfile", back_populates="training_jobs")

    # Indexes
    __table_args__ = (
        Index("ix_training_jobs_status", "status"),
        Index("ix_training_jobs_profile_status", "voice_profile_id", "status"),
    )


class WebhookSubscription(Base):
    """Webhook subscription for event notifications."""

    __tablename__ = "webhook_subscriptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=False)
    events = Column(JSONB, default=list)  # List of event types
    enabled = Column(Boolean, default=True)

    # Filtering
    agent_id = Column(String(255), nullable=True)  # Filter by agent

    # Status
    last_triggered_at = Column(DateTime, nullable=True)
    failure_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hash
    name = Column(String(255), nullable=False)

    # Permissions
    scopes = Column(JSONB, default=["synthesize", "profiles:read"])
    rate_limit = Column(Integer, default=100)  # Requests per minute

    # Status
    active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


class UsageLog(Base):
    """Usage tracking for billing and analytics."""

    __tablename__ = "usage_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=True)
    voice_profile_id = Column(String(36), nullable=True)

    # Request details
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)

    # Synthesis metrics
    input_chars = Column(Integer, nullable=True)
    output_duration_ms = Column(Integer, nullable=True)
    generation_time_ms = Column(Integer, nullable=True)

    # Performance
    cached = Column(Boolean, default=False)
    latency_ms = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("ix_usage_logs_created", "created_at"),
        Index("ix_usage_logs_profile", "voice_profile_id", "created_at"),
    )


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """
    Async database manager for the voice service.
    """

    def __init__(self, database_url: str):
        # Convert sync URL to async
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self):
        """Get a database session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# =============================================================================
# Repository Classes
# =============================================================================

class VoiceProfileRepository:
    """Repository for voice profile operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create(
        self,
        name: str,
        agent_id: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VoiceProfile:
        """Create a new voice profile."""
        async with self.db.session() as session:
            profile = VoiceProfile(
                name=name,
                agent_id=agent_id,
                description=description,
                metadata_=metadata or {},
            )
            session.add(profile)
            await session.flush()
            return profile

    async def get(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a voice profile by ID."""
        async with self.db.session() as session:
            return await session.get(VoiceProfile, profile_id)

    async def get_by_agent(self, agent_id: str) -> List[VoiceProfile]:
        """Get all voice profiles for an agent."""
        from sqlalchemy import select

        async with self.db.session() as session:
            result = await session.execute(
                select(VoiceProfile).where(VoiceProfile.agent_id == agent_id)
            )
            return list(result.scalars().all())

    async def update_status(
        self,
        profile_id: str,
        status: VoiceProfileStatus,
        model_path: Optional[str] = None,
    ):
        """Update profile status."""
        async with self.db.session() as session:
            profile = await session.get(VoiceProfile, profile_id)
            if profile:
                profile.status = status
                if model_path:
                    profile.model_path = model_path

    async def update_training_stats(
        self,
        profile_id: str,
        sample_count: int,
        total_duration_ms: int,
    ):
        """Update training data statistics."""
        async with self.db.session() as session:
            profile = await session.get(VoiceProfile, profile_id)
            if profile:
                profile.sample_count = sample_count
                profile.total_duration_ms = total_duration_ms

    async def list_all(
        self,
        status: Optional[VoiceProfileStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VoiceProfile]:
        """List voice profiles with optional filtering."""
        from sqlalchemy import select

        async with self.db.session() as session:
            query = select(VoiceProfile)
            if status:
                query = query.where(VoiceProfile.status == status)
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return list(result.scalars().all())


class TrainingJobRepository:
    """Repository for training job operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create(
        self,
        voice_profile_id: str,
        config: Dict[str, Any],
        gpu_type: str = "a100",
    ) -> TrainingJob:
        """Create a new training job."""
        async with self.db.session() as session:
            job = TrainingJob(
                voice_profile_id=voice_profile_id,
                config=config,
                gpu_type=gpu_type,
            )
            session.add(job)
            await session.flush()
            return job

    async def get(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        async with self.db.session() as session:
            return await session.get(TrainingJob, job_id)

    async def update_status(
        self,
        job_id: str,
        status: TrainingStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """Update job status."""
        async with self.db.session() as session:
            job = await session.get(TrainingJob, job_id)
            if job:
                job.status = status
                if progress is not None:
                    job.progress = progress
                if error_message:
                    job.error_message = error_message
                if status == TrainingStatus.TRAINING and not job.started_at:
                    job.started_at = datetime.utcnow()
                if status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
                    job.completed_at = datetime.utcnow()


# =============================================================================
# Global Database Instance
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager."""
    global _db_manager
    if _db_manager is None:
        from api.config import get_settings
        settings = get_settings()
        _db_manager = DatabaseManager(settings.database_url)
    return _db_manager


async def init_database():
    """Initialize the database."""
    db = get_database()
    await db.create_tables()
