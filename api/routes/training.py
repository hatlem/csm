"""
Training job management endpoints.

Provides:
- Start training jobs
- Monitor training progress
- Cancel training
"""

from typing import Optional, List

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from api.core.dependencies import get_container
from api.core.logging import get_logger
from api.core.errors import NotFoundError, ValidationError
from api.middleware.auth import get_auth_context

logger = get_logger(__name__)
router = APIRouter()


class StartTrainingRequest(BaseModel):
    """Start training request."""

    voice_profile_id: str = Field(..., description="Voice profile to train")
    use_lora: bool = Field(default=True, description="Use LoRA for fine-tuning")
    lora_rank: int = Field(default=8, ge=4, le=64, description="LoRA rank")
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    num_epochs: int = Field(default=10, ge=1, le=100)
    gpu_type: str = Field(
        default="a100",
        description="GPU type: a100, rtx4090, h100",
    )


class TrainingJobResponse(BaseModel):
    """Training job response."""

    id: str
    voice_profile_id: str
    status: str
    progress: float
    gpu_type: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ListTrainingJobsResponse(BaseModel):
    """List training jobs response."""

    jobs: List[TrainingJobResponse]
    total: int


@router.post("/training", response_model=TrainingJobResponse)
async def start_training(request: Request, body: StartTrainingRequest):
    """
    Start a voice training job.

    Requires sufficient training data (minimum 30 minutes of audio).
    """
    auth = get_auth_context(request)
    container = get_container()

    training_service = await container.get("training_service")

    # Validate training data
    stats = training_service.get_training_stats(body.voice_profile_id)
    if not stats["ready_for_training"]:
        raise ValidationError(
            message="Insufficient training data",
            details={
                "current_hours": stats["total_duration_hours"],
                "required_hours": stats["minimum_duration_hours"],
                "sample_count": stats["sample_count"],
            },
        )

    # Start training job
    job = await training_service.start_training_job(
        voice_profile_id=body.voice_profile_id,
        use_lora=body.use_lora,
        lora_rank=body.lora_rank,
        learning_rate=body.learning_rate,
        num_epochs=body.num_epochs,
        gpu_type=body.gpu_type,
    )

    logger.info(
        f"Started training job {job.id}",
        extra={
            "job_id": job.id,
            "profile_id": body.voice_profile_id,
            "gpu_type": body.gpu_type,
        },
    )

    return TrainingJobResponse(
        id=job.id,
        voice_profile_id=job.voice_profile_id,
        status=job.status,
        progress=job.progress,
        gpu_type=job.gpu_type,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get("/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(request: Request, job_id: str):
    """Get training job status."""
    container = get_container()
    training_service = await container.get("training_service")

    job = training_service.get_job(job_id)
    if not job:
        raise NotFoundError("training_job", job_id)

    return TrainingJobResponse(
        id=job.id,
        voice_profile_id=job.voice_profile_id,
        status=job.status,
        progress=job.progress,
        gpu_type=job.gpu_type,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get("/training", response_model=ListTrainingJobsResponse)
async def list_training_jobs(
    request: Request,
    voice_profile_id: Optional[str] = None,
):
    """List training jobs."""
    container = get_container()
    training_service = await container.get("training_service")

    jobs = training_service.list_jobs(voice_profile_id=voice_profile_id)

    return ListTrainingJobsResponse(
        jobs=[
            TrainingJobResponse(
                id=j.id,
                voice_profile_id=j.voice_profile_id,
                status=j.status,
                progress=j.progress,
                gpu_type=j.gpu_type,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                error_message=j.error_message,
            )
            for j in jobs
        ],
        total=len(jobs),
    )


@router.post("/training/{job_id}/cancel")
async def cancel_training_job(request: Request, job_id: str):
    """Cancel a training job."""
    container = get_container()
    training_service = await container.get("training_service")

    success = await training_service.cancel_job(job_id)
    if not success:
        raise NotFoundError("training_job", job_id)

    logger.info(f"Cancelled training job {job_id}")

    return {"cancelled": True, "job_id": job_id}


@router.get("/training/{job_id}/logs")
async def get_training_logs(
    request: Request,
    job_id: str,
    tail: int = 100,
):
    """Get training job logs."""
    container = get_container()
    training_service = await container.get("training_service")

    job = training_service.get_job(job_id)
    if not job:
        raise NotFoundError("training_job", job_id)

    return {
        "job_id": job_id,
        "logs": job.logs[-tail:] if job.logs else [],
        "total_lines": len(job.logs) if job.logs else 0,
    }
