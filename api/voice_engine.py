"""
Voice Engine - Core inference engine for CSM voice synthesis.

This is the heart of the voice service. It manages:
- Model loading and GPU memory optimization
- Voice profile loading and caching
- Real-time speech synthesis
- Streaming audio generation
"""

import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from functools import lru_cache
import hashlib

import torch
import torchaudio
from cachetools import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Loaded voice profile with model weights."""
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


class VoiceEngine:
    """
    Production-grade voice synthesis engine.

    Features:
    - Lazy model loading
    - Voice profile caching with LRU eviction
    - GPU memory management
    - Batch inference support
    - Streaming generation
    """

    def __init__(
        self,
        model_path: str = "./models/csm-1b",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_size: int = 10,
        cache_ttl: int = 3600,
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.dtype = getattr(torch, dtype)

        # Model state
        self._generator = None
        self._model_loaded = False
        self._load_lock = asyncio.Lock()

        # Voice profile cache
        self._profile_cache: TTLCache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._audio_cache: TTLCache = TTLCache(maxsize=100, ttl=300)  # 5 min cache for repeated requests

        # Metrics
        self.total_generations = 0
        self.total_audio_seconds = 0.0
        self.total_generation_time_ms = 0

    def _resolve_device(self, device: str) -> str:
        """Resolve the best available device."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning(f"Requested device '{device}' not available, falling back to CPU")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def gpu_info(self) -> Optional[Dict]:
        """Get GPU memory info."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "memory_used_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                "device_name": torch.cuda.get_device_name(0),
            }
        return None

    async def load_model(self) -> None:
        """Load the base CSM model."""
        async with self._load_lock:
            if self._model_loaded:
                return

            logger.info(f"Loading CSM model from {self.model_path} on {self.device}")
            start_time = time.time()

            try:
                # Import here to avoid loading at module level
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from generator import Generator, load_llama3_tokenizer
                from models import Model

                # Load model
                model = Model.from_pretrained(self.model_path)
                model.to(device=self.device, dtype=self.dtype)

                # Create generator
                self._generator = Generator(model)
                self._model_loaded = True

                load_time = time.time() - start_time
                logger.info(f"Model loaded in {load_time:.2f}s on {self.device}")

                if self.device == "cuda":
                    gpu_info = self.gpu_info
                    logger.info(f"GPU memory: {gpu_info['memory_used_mb']}MB / {gpu_info['memory_total_mb']}MB")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    async def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        async with self._load_lock:
            if self._generator is not None:
                del self._generator
                self._generator = None
                self._model_loaded = False

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                logger.info("Model unloaded")

    def _get_cache_key(self, voice_profile_id: str, text: str, **kwargs) -> str:
        """Generate cache key for synthesis request."""
        key_data = f"{voice_profile_id}:{text}:{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def load_voice_profile(
        self,
        profile_id: str,
        checkpoint_path: Optional[str] = None,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> VoiceProfile:
        """Load a voice profile with optional LoRA weights."""

        # Check cache
        if profile_id in self._profile_cache:
            return self._profile_cache[profile_id]

        logger.info(f"Loading voice profile: {profile_id}")

        profile = VoiceProfile(
            id=profile_id,
            name=profile_id,
            checkpoint_path=checkpoint_path or self.model_path,
        )

        # Load prompt audio if provided (for voice cloning context)
        if prompt_audio_path and Path(prompt_audio_path).exists():
            audio, sr = torchaudio.load(prompt_audio_path)
            if sr != self._generator.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=self._generator.sample_rate
                )
            profile.prompt_audio = audio.squeeze(0)
            profile.prompt_text = prompt_text

        # Load LoRA weights if checkpoint differs from base
        if checkpoint_path and checkpoint_path != self.model_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "lora_state_dict" in checkpoint:
                profile.lora_weights = checkpoint["lora_state_dict"]
                logger.info(f"Loaded LoRA weights for profile {profile_id}")

        self._profile_cache[profile_id] = profile
        return profile

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
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_profile_id: Voice profile to use
            speaker_id: Speaker ID for multi-speaker
            context: List of (speaker_id, text, optional_audio) tuples for context
            temperature: Sampling temperature
            topk: Top-k sampling
            max_duration_ms: Maximum audio duration
            use_cache: Whether to use result caching

        Returns:
            SynthesisResult with audio tensor and metadata
        """
        if not self._model_loaded:
            await self.load_model()

        # Check cache
        cache_key = self._get_cache_key(
            voice_profile_id, text, speaker_id=speaker_id, temp=temperature, topk=topk
        )
        if use_cache and cache_key in self._audio_cache:
            logger.debug(f"Cache hit for: {text[:50]}...")
            cached = self._audio_cache[cache_key]
            return SynthesisResult(
                audio=cached["audio"],
                sample_rate=cached["sample_rate"],
                duration_ms=cached["duration_ms"],
                generation_time_ms=0,  # Cached
            )

        start_time = time.time()

        # Load voice profile
        profile = await self.load_voice_profile(voice_profile_id)

        # Build context segments
        from generator import Segment
        segments = []

        # Add voice prompt as first context if available
        if profile.prompt_audio is not None and profile.prompt_text:
            segments.append(Segment(
                speaker=speaker_id,
                text=profile.prompt_text,
                audio=profile.prompt_audio,
            ))

        # Add provided context
        if context:
            for ctx_speaker, ctx_text, ctx_audio in context:
                if ctx_audio is not None:
                    segments.append(Segment(
                        speaker=ctx_speaker,
                        text=ctx_text,
                        audio=ctx_audio,
                    ))

        # Generate audio
        try:
            audio = self._generator.generate(
                text=text,
                speaker=speaker_id,
                context=segments,
                max_audio_length_ms=max_duration_ms,
                temperature=temperature,
                topk=topk,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        generation_time_ms = int((time.time() - start_time) * 1000)
        duration_ms = int(audio.shape[0] / self._generator.sample_rate * 1000)

        # Update metrics
        self.total_generations += 1
        self.total_audio_seconds += duration_ms / 1000
        self.total_generation_time_ms += generation_time_ms

        result = SynthesisResult(
            audio=audio,
            sample_rate=self._generator.sample_rate,
            duration_ms=duration_ms,
            generation_time_ms=generation_time_ms,
        )

        # Cache result
        if use_cache:
            self._audio_cache[cache_key] = {
                "audio": audio,
                "sample_rate": self._generator.sample_rate,
                "duration_ms": duration_ms,
            }

        logger.info(
            f"Generated {duration_ms}ms audio in {generation_time_ms}ms "
            f"(RTF: {generation_time_ms/duration_ms:.2f}x)"
        )

        return result

    async def synthesize_streaming(
        self,
        text: str,
        voice_profile_id: str,
        chunk_size_ms: int = 500,
        **kwargs,
    ):
        """
        Stream audio generation chunk by chunk.

        Note: CSM doesn't natively support streaming, so we generate
        full audio and chunk it. For true streaming, consider using
        a different architecture.
        """
        result = await self.synthesize(text, voice_profile_id, **kwargs)

        samples_per_chunk = int(chunk_size_ms * result.sample_rate / 1000)
        total_samples = result.audio.shape[0]

        for i, start in enumerate(range(0, total_samples, samples_per_chunk)):
            end = min(start + samples_per_chunk, total_samples)
            chunk = result.audio[start:end]

            yield {
                "chunk_index": i,
                "audio": chunk,
                "is_final": end >= total_samples,
            }

    def audio_to_bytes(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        format: str = "wav",
    ) -> bytes:
        """Convert audio tensor to bytes in specified format."""
        buffer = io.BytesIO()

        # Ensure 2D tensor (channels, samples)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if format == "wav":
            torchaudio.save(buffer, audio.cpu(), sample_rate, format="wav")
        elif format == "mp3":
            # MP3 requires additional backend
            torchaudio.save(buffer, audio.cpu(), sample_rate, format="mp3")
        elif format == "ogg":
            torchaudio.save(buffer, audio.cpu(), sample_rate, format="ogg")
        else:
            # Raw PCM
            buffer.write(audio.cpu().numpy().tobytes())

        buffer.seek(0)
        return buffer.read()

    def audio_to_base64(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        format: str = "wav",
    ) -> str:
        """Convert audio tensor to base64 string."""
        audio_bytes = self.audio_to_bytes(audio, sample_rate, format)
        return base64.b64encode(audio_bytes).decode("utf-8")


# Global engine instance (singleton pattern)
_engine: Optional[VoiceEngine] = None


def get_engine() -> VoiceEngine:
    """Get or create the global voice engine instance."""
    global _engine
    if _engine is None:
        from api.config import get_settings
        settings = get_settings()
        _engine = VoiceEngine(
            model_path=settings.model_path,
            device=settings.model_device,
            dtype=settings.model_dtype,
        )
    return _engine
