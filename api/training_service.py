"""
Training Service - Manages voice profile training jobs.

Handles:
- Training data storage and validation
- Cloud GPU provisioning (RunPod, Vast.ai, Modal)
- Training job orchestration
- Model artifact management
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import httpx

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """Training job state."""
    id: str
    voice_profile_id: str
    status: str = "queued"
    progress: float = 0.0
    gpu_type: str = "a100"
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    cloud_instance_id: Optional[str] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class TrainingSample:
    """A training sample."""
    id: str
    voice_profile_id: str
    audio_path: str
    transcript: str
    duration_ms: int


class TrainingService:
    """
    Manages the full lifecycle of voice training jobs.

    Supports multiple cloud GPU providers for flexible deployment.
    """

    def __init__(
        self,
        storage_dir: str = "./voice_profiles",
        gpu_provider: str = "runpod",
        runpod_api_key: Optional[str] = None,
        vast_api_key: Optional[str] = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_provider = gpu_provider
        self.runpod_api_key = runpod_api_key
        self.vast_api_key = vast_api_key

        # In-memory job tracking (use Redis in production)
        self._jobs: Dict[str, TrainingJob] = {}
        self._samples: Dict[str, List[TrainingSample]] = {}

    def get_profile_dir(self, profile_id: str) -> Path:
        """Get directory for a voice profile."""
        profile_dir = self.storage_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    async def add_training_sample(
        self,
        voice_profile_id: str,
        audio_base64: str,
        transcript: str,
    ) -> TrainingSample:
        """Add a training sample to a voice profile."""
        import torchaudio

        profile_dir = self.get_profile_dir(voice_profile_id)
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        sample_id = str(uuid.uuid4())[:8]
        audio_path = samples_dir / f"{sample_id}.wav"

        # Save audio file
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Get duration
        try:
            info = torchaudio.info(str(audio_path))
            duration_ms = int(info.num_frames / info.sample_rate * 1000)
        except Exception:
            duration_ms = 0

        sample = TrainingSample(
            id=sample_id,
            voice_profile_id=voice_profile_id,
            audio_path=str(audio_path),
            transcript=transcript,
            duration_ms=duration_ms,
        )

        # Store sample metadata
        if voice_profile_id not in self._samples:
            self._samples[voice_profile_id] = []
        self._samples[voice_profile_id].append(sample)

        # Save transcripts file
        self._save_transcripts(voice_profile_id)

        logger.info(f"Added sample {sample_id} to profile {voice_profile_id} ({duration_ms}ms)")
        return sample

    def _save_transcripts(self, voice_profile_id: str) -> None:
        """Save transcripts JSON file for training."""
        profile_dir = self.get_profile_dir(voice_profile_id)
        samples = self._samples.get(voice_profile_id, [])

        transcripts = [
            {
                "audio": Path(s.audio_path).name,
                "text": s.transcript,
            }
            for s in samples
        ]

        with open(profile_dir / "transcripts.json", "w") as f:
            json.dump(transcripts, f, indent=2)

    def get_training_stats(self, voice_profile_id: str) -> Dict[str, Any]:
        """Get training data statistics."""
        samples = self._samples.get(voice_profile_id, [])
        total_duration_ms = sum(s.duration_ms for s in samples)

        return {
            "sample_count": len(samples),
            "total_duration_hours": total_duration_ms / 1000 / 3600,
            "ready_for_training": total_duration_ms >= 30 * 60 * 1000,  # 30 min minimum
            "minimum_duration_hours": 0.5,
        }

    async def start_training_job(
        self,
        voice_profile_id: str,
        use_lora: bool = True,
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        gpu_type: str = "a100",
    ) -> TrainingJob:
        """Start a training job on cloud GPU."""

        stats = self.get_training_stats(voice_profile_id)
        if not stats["ready_for_training"]:
            raise ValueError(
                f"Not enough training data. Need {stats['minimum_duration_hours']} hours, "
                f"have {stats['total_duration_hours']:.2f} hours"
            )

        job_id = str(uuid.uuid4())[:12]
        job = TrainingJob(
            id=job_id,
            voice_profile_id=voice_profile_id,
            status="queued",
            gpu_type=gpu_type,
            config={
                "use_lora": use_lora,
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
        )

        self._jobs[job_id] = job

        # Start training in background
        asyncio.create_task(self._run_training_job(job))

        return job

    async def _run_training_job(self, job: TrainingJob) -> None:
        """Execute the training job."""
        try:
            job.status = "provisioning"
            job.started_at = datetime.utcnow()

            # Provision cloud GPU
            if self.gpu_provider == "runpod":
                instance_id = await self._provision_runpod(job)
            elif self.gpu_provider == "vast":
                instance_id = await self._provision_vast(job)
            else:
                # Local training
                instance_id = "local"

            job.cloud_instance_id = instance_id
            job.status = "training"

            # For local training, run directly
            if instance_id == "local":
                await self._train_local(job)
            else:
                # Monitor remote training
                await self._monitor_training(job)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = datetime.utcnow()

            logger.info(f"Training job {job.id} completed successfully")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Training job {job.id} failed: {e}")

    async def _train_local(self, job: TrainingJob) -> None:
        """Run training locally (for development/testing)."""
        import subprocess

        profile_dir = self.get_profile_dir(job.voice_profile_id)

        cmd = [
            "python", "finetune/train.py",
            "--data_path", str(profile_dir / "train_data.pt"),
            "--output_dir", str(profile_dir / "checkpoints"),
            "--num_epochs", str(job.config["num_epochs"]),
            "--learning_rate", str(job.config["learning_rate"]),
        ]

        if job.config.get("use_lora"):
            cmd.extend(["--use_lora", "--lora_rank", str(job.config["lora_rank"])])

        # First prepare the data
        prep_cmd = [
            "python", "finetune/prepare_data.py",
            "--audio_dir", str(profile_dir / "samples"),
            "--transcripts", str(profile_dir / "transcripts.json"),
            "--output_dir", str(profile_dir),
        ]

        logger.info(f"Preparing training data: {' '.join(prep_cmd)}")
        subprocess.run(prep_cmd, check=True)

        logger.info(f"Starting local training: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                job.logs.append(line.strip())
                # Parse progress from logs
                if "Epoch" in line:
                    # Simple progress parsing
                    job.progress = min(0.9, job.progress + 0.1)

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with code {process.returncode}")

    async def _provision_runpod(self, job: TrainingJob) -> str:
        """Provision a RunPod GPU instance."""
        if not self.runpod_api_key:
            raise ValueError("RunPod API key not configured")

        # RunPod GPU type mapping
        gpu_map = {
            "a100": "NVIDIA A100 80GB PCIe",
            "rtx4090": "NVIDIA GeForce RTX 4090",
            "h100": "NVIDIA H100 PCIe",
        }

        async with httpx.AsyncClient() as client:
            # Create pod
            response = await client.post(
                "https://api.runpod.io/graphql",
                headers={"Authorization": f"Bearer {self.runpod_api_key}"},
                json={
                    "query": """
                        mutation {
                            podRentInterruptable(
                                input: {
                                    cloudType: SECURE
                                    gpuCount: 1
                                    volumeInGb: 50
                                    containerDiskInGb: 20
                                    minVcpuCount: 4
                                    minMemoryInGb: 16
                                    gpuTypeId: "%s"
                                    imageName: "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
                                    dockerArgs: ""
                                    ports: "8888/http,22/tcp"
                                    volumeMountPath: "/workspace"
                                }
                            ) {
                                id
                                imageName
                                gpuCount
                            }
                        }
                    """ % gpu_map.get(job.gpu_type, gpu_map["a100"])
                },
            )

            data = response.json()
            if "errors" in data:
                raise RuntimeError(f"RunPod error: {data['errors']}")

            pod_id = data["data"]["podRentInterruptable"]["id"]
            logger.info(f"Provisioned RunPod instance: {pod_id}")

            return pod_id

    async def _provision_vast(self, job: TrainingJob) -> str:
        """Provision a Vast.ai GPU instance."""
        if not self.vast_api_key:
            raise ValueError("Vast.ai API key not configured")

        # Vast.ai provisioning would go here
        # For now, return placeholder
        logger.info("Vast.ai provisioning not yet implemented")
        return "vast-placeholder"

    async def _monitor_training(self, job: TrainingJob) -> None:
        """Monitor remote training progress."""
        # Poll for training status
        while job.status == "training":
            await asyncio.sleep(30)
            # Check remote instance status
            # Update job.progress based on logs
            # This would integrate with the cloud provider's API

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, voice_profile_id: Optional[str] = None) -> List[TrainingJob]:
        """List training jobs."""
        jobs = list(self._jobs.values())
        if voice_profile_id:
            jobs = [j for j in jobs if j.voice_profile_id == voice_profile_id]
        return jobs

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in ("completed", "failed", "cancelled"):
            return False

        job.status = "cancelled"
        job.completed_at = datetime.utcnow()

        # Terminate cloud instance if running
        if job.cloud_instance_id and job.cloud_instance_id != "local":
            await self._terminate_instance(job.cloud_instance_id)

        return True

    async def _terminate_instance(self, instance_id: str) -> None:
        """Terminate a cloud GPU instance."""
        # Implementation depends on provider
        logger.info(f"Terminating instance: {instance_id}")


# Global service instance
_training_service: Optional[TrainingService] = None


def get_training_service() -> TrainingService:
    """Get or create the global training service instance."""
    global _training_service
    if _training_service is None:
        from api.config import get_settings
        settings = get_settings()
        _training_service = TrainingService(
            storage_dir=settings.voice_profiles_dir,
            gpu_provider=settings.training_gpu_provider,
            runpod_api_key=settings.runpod_api_key,
            vast_api_key=settings.vast_api_key,
        )
    return _training_service
