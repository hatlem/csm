"""
Pydantic models for API request/response schemas.

These models define the contract between Human-Like and the CSM Voice Service.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import base64


class VoiceProfileStatus(str, Enum):
    """Status of a voice profile."""
    PENDING = "pending"          # Awaiting training data
    TRAINING = "training"        # Currently training
    READY = "ready"              # Ready for inference
    FAILED = "failed"            # Training failed
    ARCHIVED = "archived"        # No longer active


class TrainingStatus(str, Enum):
    """Status of a training job."""
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, Enum):
    """Supported audio output formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    PCM = "pcm"


# ============================================================================
# Voice Profile Management
# ============================================================================

class VoiceProfileCreate(BaseModel):
    """Request to create a new voice profile."""
    name: str = Field(..., description="Display name for the voice profile")
    agent_id: str = Field(..., description="Human-Like agent/CSM ID")
    description: Optional[str] = Field(None, description="Description of the voice")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class VoiceProfileResponse(BaseModel):
    """Voice profile details."""
    id: str
    name: str
    agent_id: str
    status: VoiceProfileStatus
    description: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    training_hours: Optional[float] = None
    sample_count: Optional[int] = None
    model_path: Optional[str] = None


class VoiceProfileList(BaseModel):
    """List of voice profiles."""
    profiles: List[VoiceProfileResponse]
    total: int


# ============================================================================
# Training Data Upload
# ============================================================================

class TrainingSample(BaseModel):
    """A single training sample with audio and transcript."""
    audio_base64: str = Field(..., description="Base64 encoded audio file")
    transcript: str = Field(..., description="Text transcript of the audio")
    duration_ms: Optional[int] = Field(None, description="Audio duration in milliseconds")

    @validator('audio_base64')
    def validate_audio(cls, v):
        try:
            decoded = base64.b64decode(v)
            if len(decoded) < 1000:
                raise ValueError("Audio file too small")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio: {e}")


class TrainingDataUpload(BaseModel):
    """Batch upload of training samples."""
    voice_profile_id: str
    samples: List[TrainingSample]


class TrainingDataResponse(BaseModel):
    """Response after uploading training data."""
    voice_profile_id: str
    samples_uploaded: int
    total_samples: int
    total_duration_hours: float
    ready_for_training: bool
    minimum_duration_hours: float = 0.5


# ============================================================================
# Training Jobs
# ============================================================================

class TrainingJobCreate(BaseModel):
    """Request to start a training job."""
    voice_profile_id: str
    use_lora: bool = Field(True, description="Use LoRA for efficient fine-tuning")
    lora_rank: int = Field(8, ge=4, le=64)
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-2)
    num_epochs: int = Field(10, ge=1, le=100)
    gpu_type: str = Field("a100", description="GPU type: a100, rtx4090, h100")


class TrainingJobResponse(BaseModel):
    """Training job details."""
    id: str
    voice_profile_id: str
    status: TrainingStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    gpu_type: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    estimated_cost_usd: Optional[float]
    logs_url: Optional[str]


class TrainingJobList(BaseModel):
    """List of training jobs."""
    jobs: List[TrainingJobResponse]
    total: int


# ============================================================================
# Voice Synthesis (Inference)
# ============================================================================

class ContextSegment(BaseModel):
    """A context segment for multi-turn conversation."""
    speaker_id: int = Field(..., ge=0, le=10)
    text: str
    audio_base64: Optional[str] = Field(None, description="Optional audio for better voice matching")


class SynthesizeRequest(BaseModel):
    """Request to synthesize speech."""
    voice_profile_id: str = Field(..., description="Voice profile to use")
    text: str = Field(..., min_length=1, max_length=5000)
    speaker_id: int = Field(0, ge=0, le=10, description="Speaker ID for multi-speaker")
    context: List[ContextSegment] = Field(default_factory=list, max_items=10)
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    topk: int = Field(50, ge=1, le=200)
    max_duration_ms: int = Field(30000, ge=1000, le=90000)
    output_format: AudioFormat = Field(AudioFormat.WAV)
    stream: bool = Field(False, description="Enable streaming response")


class SynthesizeResponse(BaseModel):
    """Response with synthesized audio."""
    audio_base64: str
    duration_ms: int
    sample_rate: int
    format: AudioFormat
    voice_profile_id: str
    generation_time_ms: int
    cached: bool = False


class StreamChunk(BaseModel):
    """A chunk of streamed audio."""
    chunk_index: int
    audio_base64: str
    is_final: bool


# ============================================================================
# Health & Metrics
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_mb: Optional[int]
    gpu_memory_total_mb: Optional[int]
    active_requests: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Service metrics."""
    total_requests: int
    total_audio_generated_seconds: float
    average_latency_ms: float
    cache_hit_rate: float
    active_voice_profiles: int
    training_jobs_completed: int
    errors_last_hour: int


# ============================================================================
# Webhooks (for Human-Like integration)
# ============================================================================

class WebhookEvent(str, Enum):
    """Webhook event types."""
    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    VOICE_PROFILE_READY = "voice_profile.ready"


class WebhookPayload(BaseModel):
    """Webhook payload sent to Human-Like."""
    event: WebhookEvent
    timestamp: datetime
    voice_profile_id: str
    data: Dict[str, Any]


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    url: str
    secret: str
    events: List[WebhookEvent]
    enabled: bool = True
