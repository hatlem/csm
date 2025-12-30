"""
Speech synthesis endpoints.

Provides:
- Text-to-speech synthesis
- Streaming synthesis
- Voice customization
"""

import base64
import time
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, validator

from api.core.dependencies import get_container
from api.core.logging import get_logger
from api.core.metrics import get_metrics
from api.core.errors import ValidationError
from api.middleware.auth import get_auth_context

logger = get_logger(__name__)
router = APIRouter()


class SynthesizeRequest(BaseModel):
    """Speech synthesis request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to synthesize",
    )
    voice_profile_id: str = Field(
        default="default",
        description="Voice profile ID",
    )
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Top-k sampling parameter",
    )
    max_audio_length_ms: Optional[int] = Field(
        default=None,
        ge=1000,
        le=300000,
        description="Maximum audio length in milliseconds",
    )
    context: Optional[List[dict]] = Field(
        default=None,
        description="Conversation context for better prosody",
    )

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class SynthesizeResponse(BaseModel):
    """Speech synthesis response."""

    audio_base64: str = Field(description="Base64 encoded audio (WAV)")
    sample_rate: int = Field(description="Audio sample rate")
    duration_ms: int = Field(description="Audio duration in milliseconds")
    generation_time_ms: int = Field(description="Time to generate audio")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: Request, body: SynthesizeRequest):
    """
    Synthesize speech from text.

    Returns base64 encoded WAV audio.
    """
    auth = get_auth_context(request)
    metrics = get_metrics()
    container = get_container()

    start_time = time.time()

    logger.info(
        "Synthesis request",
        extra={
            "text_length": len(body.text),
            "voice_profile_id": body.voice_profile_id,
            "api_key_id": auth.api_key_id,
        },
    )

    # Get voice engine
    voice_engine = await container.get("voice_engine")

    # Build context if provided
    context = None
    if body.context:
        context = [
            (item.get("text", ""), item.get("speaker", 0))
            for item in body.context
        ]

    # Synthesize
    try:
        result = await voice_engine.synthesize(
            text=body.text,
            voice_profile_id=body.voice_profile_id,
            temperature=body.temperature,
            top_k=body.top_k,
            max_audio_length_ms=body.max_audio_length_ms,
            context=context,
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        metrics.SYNTHESIS_ERRORS.labels(error_type=type(e).__name__).inc()
        raise

    # Encode audio to base64
    import io
    import torchaudio

    buffer = io.BytesIO()
    torchaudio.save(
        buffer,
        result.audio.unsqueeze(0),
        result.sample_rate,
        format="wav",
    )
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()

    generation_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "Synthesis completed",
        extra={
            "duration_ms": result.duration_ms,
            "generation_time_ms": generation_time_ms,
            "voice_profile_id": body.voice_profile_id,
        },
    )

    return SynthesizeResponse(
        audio_base64=audio_base64,
        sample_rate=result.sample_rate,
        duration_ms=result.duration_ms,
        generation_time_ms=generation_time_ms,
    )


class BatchSynthesizeRequest(BaseModel):
    """Batch synthesis request."""

    items: List[SynthesizeRequest] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Items to synthesize",
    )


class BatchSynthesizeResponse(BaseModel):
    """Batch synthesis response."""

    results: List[SynthesizeResponse]
    total_duration_ms: int
    total_generation_time_ms: int


@router.post("/synthesize/batch", response_model=BatchSynthesizeResponse)
async def synthesize_batch(request: Request, body: BatchSynthesizeRequest):
    """
    Batch synthesize multiple texts.

    Processes items sequentially to manage GPU memory.
    """
    start_time = time.time()
    results = []

    for item in body.items:
        # Create a mock request for individual synthesis
        result = await synthesize(request, item)
        results.append(result)

    total_generation_time_ms = int((time.time() - start_time) * 1000)
    total_duration_ms = sum(r.duration_ms for r in results)

    return BatchSynthesizeResponse(
        results=results,
        total_duration_ms=total_duration_ms,
        total_generation_time_ms=total_generation_time_ms,
    )
