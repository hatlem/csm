"""
Voice profile management endpoints.

Provides:
- Profile CRUD operations
- Sample upload
- Profile activation
"""

import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from api.core.dependencies import get_container
from api.core.logging import get_logger
from api.core.errors import NotFoundError, ValidationError
from api.middleware.auth import get_auth_context

logger = get_logger(__name__)
router = APIRouter()


class VoiceProfile(BaseModel):
    """Voice profile model."""

    id: str
    name: str
    agent_id: Optional[str] = None
    description: Optional[str] = None
    status: str = "pending"  # pending, training, ready, failed
    sample_count: int = 0
    total_duration_hours: float = 0.0
    created_at: str
    updated_at: str
    model_version: Optional[str] = None


class CreateProfileRequest(BaseModel):
    """Create profile request."""

    name: str = Field(..., min_length=1, max_length=100)
    agent_id: Optional[str] = None
    description: Optional[str] = Field(default=None, max_length=500)


class CreateProfileResponse(BaseModel):
    """Create profile response."""

    id: str
    name: str
    status: str


class ListProfilesResponse(BaseModel):
    """List profiles response."""

    profiles: List[VoiceProfile]
    total: int
    page: int
    page_size: int


# In-memory profile storage (replace with database in production)
_profiles: dict[str, dict] = {}


@router.post("/profiles", response_model=CreateProfileResponse)
async def create_profile(request: Request, body: CreateProfileRequest):
    """Create a new voice profile."""
    auth = get_auth_context(request)

    profile_id = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()

    profile = {
        "id": profile_id,
        "name": body.name,
        "agent_id": body.agent_id,
        "description": body.description,
        "status": "pending",
        "sample_count": 0,
        "total_duration_hours": 0.0,
        "created_at": now,
        "updated_at": now,
        "model_version": None,
        "owner_id": auth.api_key_id or auth.user_id,
    }

    _profiles[profile_id] = profile

    logger.info(
        f"Created voice profile {profile_id}",
        extra={"profile_id": profile_id, "name": body.name},
    )

    return CreateProfileResponse(
        id=profile_id,
        name=body.name,
        status="pending",
    )


@router.get("/profiles", response_model=ListProfilesResponse)
async def list_profiles(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    agent_id: Optional[str] = None,
):
    """List voice profiles."""
    auth = get_auth_context(request)

    # Filter profiles
    profiles = list(_profiles.values())

    if agent_id:
        profiles = [p for p in profiles if p.get("agent_id") == agent_id]

    # Pagination
    total = len(profiles)
    start = (page - 1) * page_size
    end = start + page_size
    profiles = profiles[start:end]

    return ListProfilesResponse(
        profiles=[VoiceProfile(**p) for p in profiles],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/profiles/{profile_id}", response_model=VoiceProfile)
async def get_profile(request: Request, profile_id: str):
    """Get a voice profile by ID."""
    if profile_id not in _profiles:
        raise NotFoundError("voice_profile", profile_id)

    return VoiceProfile(**_profiles[profile_id])


@router.delete("/profiles/{profile_id}")
async def delete_profile(request: Request, profile_id: str):
    """Delete a voice profile."""
    if profile_id not in _profiles:
        raise NotFoundError("voice_profile", profile_id)

    del _profiles[profile_id]

    logger.info(f"Deleted voice profile {profile_id}")

    return {"deleted": True, "id": profile_id}


class UploadSampleResponse(BaseModel):
    """Upload sample response."""

    sample_id: str
    duration_ms: int
    profile_sample_count: int
    profile_total_duration_hours: float


@router.post("/profiles/{profile_id}/samples", response_model=UploadSampleResponse)
async def upload_sample(
    request: Request,
    profile_id: str,
    audio: UploadFile = File(...),
    transcript: str = Form(...),
):
    """
    Upload a training sample to a voice profile.

    Audio should be WAV format, 16-48kHz, mono.
    """
    if profile_id not in _profiles:
        raise NotFoundError("voice_profile", profile_id)

    # Validate file type
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise ValidationError(
            message="Invalid file type",
            details={"expected": "audio/*", "received": audio.content_type},
        )

    # Read audio bytes
    audio_bytes = await audio.read()

    # Get training service
    container = get_container()
    training_service = await container.get("training_service")

    # Add sample
    result = await training_service.add_training_sample(
        voice_profile_id=profile_id,
        audio_bytes=audio_bytes,
        transcript=transcript,
    )

    # Update profile stats
    stats = training_service.get_training_stats(profile_id)
    profile = _profiles[profile_id]
    profile["sample_count"] = stats["sample_count"]
    profile["total_duration_hours"] = stats["total_duration_hours"]
    profile["updated_at"] = datetime.utcnow().isoformat()

    return UploadSampleResponse(
        sample_id=result["sample_id"],
        duration_ms=result["duration_ms"],
        profile_sample_count=stats["sample_count"],
        profile_total_duration_hours=stats["total_duration_hours"],
    )


@router.get("/profiles/{profile_id}/stats")
async def get_profile_stats(request: Request, profile_id: str):
    """Get training statistics for a voice profile."""
    if profile_id not in _profiles:
        raise NotFoundError("voice_profile", profile_id)

    container = get_container()
    training_service = await container.get("training_service")

    stats = training_service.get_training_stats(profile_id)

    return stats
