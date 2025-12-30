"""
Production-grade Voice Engine Service.

Key improvements over the basic implementation:
- True async inference using thread pool
- Proper resource management
- Circuit breaker for fault tolerance
- Metrics collection
- Graceful shutdown
"""

import asyncio
import base64
import io
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import torch
import torchaudio
from cachetools import TTLCache

from api.core.logging import get_logger, log_execution_time
from api.core.metrics import get_metrics
from api.core.errors import ModelError, ModelNotLoadedError
from api.services.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


@dataclass
class VoiceProfile:
    """Loaded voice profile."""
    id: str
    name: str
    checkpoint_path: str
    prompt_audio: Optional[torch.Tensor] = None
    prompt_text: Optional[str] = None
    lora_weights: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""
    audio: torch.Tensor
    sample_rate: int
    duration_ms: int
    generation_time_ms: int


class VoiceEngineService:
    """
    Production-grade voice synthesis engine.

    Features:
    - Non-blocking async inference via thread pool
    - GPU memory management
    - Voice profile caching
    - Circuit breaker for fault tolerance
    - Comprehensive metrics
    """

    def __init__(
        self,
        model_path: str = "./models/csm-1b",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_size: int = 10,
        cache_ttl: int = 3600,
        max_workers: int = 2,
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Model state
        self._generator = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Thread pool for blocking inference
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="csm-inference",
        )

        # Caches
        self._profile_cache: TTLCache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._audio_cache: TTLCache = TTLCache(maxsize=100, ttl=300)

        # Circuit breaker for inference
        self._circuit_breaker = CircuitBreaker(
            name="voice_engine",
            failure_threshold=3,
            recovery_timeout=30,
        )

        # Metrics
        self._metrics = get_metrics()

    def _resolve_device(self, device: str) -> str:
        """Resolve the best available device."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            if device not in ("cpu",):
                logger.warning(f"Device '{device}' not available, using CPU")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def sample_rate(self) -> int:
        """Get model sample rate."""
        if self._generator:
            return self._generator.sample_rate
        return 24000

    async def initialize(self) -> None:
        """Initialize the service (called by DI container)."""
        logger.info("Initializing VoiceEngineService...")
        await self.load_model()

    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        logger.info("Shutting down VoiceEngineService...")
        self._shutdown_event.set()

        # Wait for pending tasks
        self._executor.shutdown(wait=True)

        # Unload model
        await self.unload_model()

    async def load_model(self) -> None:
        """Load the CSM model."""
        async with self._load_lock:
            if self._model_loaded:
                return

            logger.info(f"Loading CSM model from {self.model_path}")
            start_time = time.time()

            try:
                # Run model loading in thread pool to not block
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    self._load_model_sync,
                )

                load_time = time.time() - start_time
                self._model_loaded = True

                # Update metrics
                self._metrics.set_model_loaded(True, load_time)

                logger.info(
                    f"Model loaded in {load_time:.2f}s",
                    extra={
                        "device": self.device,
                        "load_time_seconds": load_time,
                    },
                )

                # Log GPU info
                if self.device == "cuda":
                    self._update_gpu_metrics()

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._metrics.set_model_loaded(False)
                raise ModelError(
                    message="Failed to load model",
                    internal_message=str(e),
                )

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from generator import Generator
        from models import Model

        model = Model.from_pretrained(self.model_path)
        model.to(device=self.device, dtype=self.dtype)
        self._generator = Generator(model)

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        async with self._load_lock:
            if self._generator is not None:
                del self._generator
                self._generator = None
                self._model_loaded = False

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                self._metrics.set_model_loaded(False)
                logger.info("Model unloaded")

    def _update_gpu_metrics(self) -> None:
        """Update GPU metrics."""
        if self.device == "cuda" and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory

            self._metrics.set_gpu_metrics(
                memory_used=memory_used,
                memory_total=memory_total,
                utilization=0,  # Would need nvidia-smi for real utilization
            )

    def _get_cache_key(self, voice_profile_id: str, text: str, **kwargs) -> str:
        """Generate cache key for synthesis request."""
        key_data = f"{voice_profile_id}:{text}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def synthesize(
        self,
        text: str,
        voice_profile_id: str,
        speaker_id: int = 0,
        context: Optional[List[Tuple[int, str, Optional[torch.Tensor]]]] = None,
        temperature: float = 0.9,
        topk: int = 50,
        max_duration_ms: int = 30000,
        use_cache: bool = True,
    ) -> SynthesisResult:
        """
        Synthesize speech from text (non-blocking).

        This runs the actual model inference in a thread pool to avoid
        blocking the async event loop.
        """
        if not self._model_loaded:
            await self.load_model()

        if not self._model_loaded:
            raise ModelNotLoadedError()

        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            raise ModelError(
                message="Service temporarily unavailable",
                internal_message="Circuit breaker is open",
            )

        # Check cache
        cache_key = self._get_cache_key(
            voice_profile_id, text,
            speaker_id=speaker_id,
            temp=temperature,
            topk=topk,
        )

        if use_cache and cache_key in self._audio_cache:
            self._metrics.record_cache_access("audio", hit=True)
            cached = self._audio_cache[cache_key]
            return SynthesisResult(
                audio=cached["audio"],
                sample_rate=cached["sample_rate"],
                duration_ms=cached["duration_ms"],
                generation_time_ms=0,
            )

        self._metrics.record_cache_access("audio", hit=False)

        # Run inference in thread pool
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._synthesize_sync,
                text,
                voice_profile_id,
                speaker_id,
                context,
                temperature,
                topk,
                max_duration_ms,
            )

            # Record success
            self._circuit_breaker.record_success()

            generation_time_ms = int((time.time() - start_time) * 1000)
            duration_ms = int(result.shape[0] / self.sample_rate * 1000)

            # Record metrics
            self._metrics.record_synthesis(
                voice_profile=voice_profile_id,
                duration_seconds=generation_time_ms / 1000,
                audio_duration_seconds=duration_ms / 1000,
                success=True,
            )

            # Cache result
            if use_cache:
                self._audio_cache[cache_key] = {
                    "audio": result,
                    "sample_rate": self.sample_rate,
                    "duration_ms": duration_ms,
                }
                self._metrics.set_cache_size("audio", len(self._audio_cache))

            return SynthesisResult(
                audio=result,
                sample_rate=self.sample_rate,
                duration_ms=duration_ms,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            self._metrics.record_synthesis(
                voice_profile=voice_profile_id,
                duration_seconds=time.time() - start_time,
                audio_duration_seconds=0,
                success=False,
            )

            logger.error(
                "Synthesis failed",
                extra={
                    "voice_profile": voice_profile_id,
                    "text_length": len(text),
                    "error": str(e),
                },
            )

            raise ModelError(
                message="Speech synthesis failed",
                internal_message=str(e),
            )

    def _synthesize_sync(
        self,
        text: str,
        voice_profile_id: str,
        speaker_id: int,
        context: Optional[List],
        temperature: float,
        topk: int,
        max_duration_ms: int,
    ) -> torch.Tensor:
        """Synchronous synthesis (runs in thread pool)."""
        from generator import Segment

        segments = []

        # Add context if provided
        if context:
            for ctx_speaker, ctx_text, ctx_audio in context:
                if ctx_audio is not None:
                    segments.append(Segment(
                        speaker=ctx_speaker,
                        text=ctx_text,
                        audio=ctx_audio,
                    ))

        # Generate
        audio = self._generator.generate(
            text=text,
            speaker=speaker_id,
            context=segments,
            max_audio_length_ms=max_duration_ms,
            temperature=temperature,
            topk=topk,
        )

        return audio

    def audio_to_bytes(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        format: str = "wav",
    ) -> bytes:
        """Convert audio tensor to bytes."""
        buffer = io.BytesIO()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        torchaudio.save(buffer, audio.cpu(), sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()

    def audio_to_base64(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        format: str = "wav",
    ) -> str:
        """Convert audio tensor to base64."""
        audio_bytes = self.audio_to_bytes(audio, sample_rate, format)
        return base64.b64encode(audio_bytes).decode("utf-8")

    def get_health_info(self) -> Dict[str, Any]:
        """Get health information for health check endpoint."""
        info = {
            "model_loaded": self._model_loaded,
            "device": self.device,
            "circuit_breaker_state": self._circuit_breaker.state,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
            }

        return info
