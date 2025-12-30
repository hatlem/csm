"""
CSM Voice API Service - Main FastAPI Application

Production-ready voice synthesis API that integrates with Human-Like
for native voice agent capabilities.

Features:
- Voice profile management (clone any voice)
- Real-time speech synthesis
- Twilio integration for phone calls
- WebSocket streaming for low-latency
- Native agent support for Human-Like
"""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from api.config import get_settings, Settings
from api.models import (
    VoiceProfileCreate, VoiceProfileResponse, VoiceProfileList,
    TrainingDataUpload, TrainingDataResponse,
    TrainingJobCreate, TrainingJobResponse, TrainingJobList,
    SynthesizeRequest, SynthesizeResponse,
    HealthResponse, AudioFormat, VoiceProfileStatus,
)
from api.voice_engine import get_engine, VoiceEngine
from api.training_service import get_training_service, TrainingService
from api.database import get_database, VoiceProfileRepository, DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Startup time tracking
_start_time = time.time()

# Active request tracking
_active_requests = 0
_active_requests_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting CSM Voice Service...")

    # Pre-load model on startup (optional, can be lazy)
    settings = get_settings()
    if settings.environment == "production":
        engine = get_engine()
        await engine.load_model()

    yield

    # Cleanup
    logger.info("Shutting down CSM Voice Service...")
    engine = get_engine()
    await engine.unload_model()


# Create FastAPI app
app = FastAPI(
    title="CSM Voice Service",
    description="Voice cloning and synthesis API for Human-Like",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependencies
# ============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verify API key for protected endpoints."""
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_voice_engine() -> VoiceEngine:
    """Dependency to get voice engine."""
    return get_engine()


def get_training_svc() -> TrainingService:
    """Dependency to get training service."""
    return get_training_service()


def get_db() -> DatabaseManager:
    """Dependency to get database manager."""
    return get_database()


async def track_request():
    """Middleware dependency to track active requests."""
    global _active_requests
    async with _active_requests_lock:
        _active_requests += 1
    try:
        yield
    finally:
        async with _active_requests_lock:
            _active_requests -= 1


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(engine: VoiceEngine = Depends(get_voice_engine)):
    """Health check endpoint."""
    global _active_requests
    gpu_info = engine.gpu_info

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=engine.is_loaded,
        gpu_available=gpu_info is not None,
        gpu_memory_used_mb=gpu_info["memory_used_mb"] if gpu_info else None,
        gpu_memory_total_mb=gpu_info["memory_total_mb"] if gpu_info else None,
        active_requests=_active_requests,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "CSM Voice Service",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================================================
# Voice Synthesis
# ============================================================================

@app.post("/v1/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(
    request: SynthesizeRequest,
    engine: VoiceEngine = Depends(get_voice_engine),
    _: bool = Depends(verify_api_key),
    __: None = Depends(track_request),
):
    """
    Synthesize speech from text using a voice profile.

    This is the main endpoint for text-to-speech generation.
    """
    start_time = time.time()

    # Ensure model is loaded
    if not engine.is_loaded:
        await engine.load_model()

    # Build context from request
    context = None
    if request.context:
        context = []
        for seg in request.context:
            audio = None
            if seg.audio_base64:
                import torchaudio
                import io
                audio_bytes = base64.b64decode(seg.audio_base64)
                audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
                if sr != engine._generator.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio, orig_freq=sr, new_freq=engine._generator.sample_rate
                    )
                audio = audio.squeeze(0)
            context.append((seg.speaker_id, seg.text, audio))

    # Generate speech
    result = await engine.synthesize(
        text=request.text,
        voice_profile_id=request.voice_profile_id,
        speaker_id=request.speaker_id,
        context=context,
        temperature=request.temperature,
        topk=request.topk,
        max_duration_ms=request.max_duration_ms,
    )

    # Convert to base64
    audio_base64 = engine.audio_to_base64(
        result.audio,
        result.sample_rate,
        request.output_format.value,
    )

    return SynthesizeResponse(
        audio_base64=audio_base64,
        duration_ms=result.duration_ms,
        sample_rate=result.sample_rate,
        format=request.output_format,
        voice_profile_id=request.voice_profile_id,
        generation_time_ms=result.generation_time_ms,
        cached=False,
    )


@app.post("/v1/synthesize/stream")
async def synthesize_stream(
    request: SynthesizeRequest,
    engine: VoiceEngine = Depends(get_voice_engine),
    _: bool = Depends(verify_api_key),
    __: None = Depends(track_request),
):
    """
    Stream synthesized speech chunk by chunk.

    Returns Server-Sent Events with audio chunks.
    """
    if not engine.is_loaded:
        await engine.load_model()

    async def generate():
        async for chunk in engine.synthesize_streaming(
            text=request.text,
            voice_profile_id=request.voice_profile_id,
            speaker_id=request.speaker_id,
            temperature=request.temperature,
            topk=request.topk,
        ):
            audio_b64 = engine.audio_to_base64(
                chunk["audio"],
                engine._generator.sample_rate,
                request.output_format.value,
            )
            data = {
                "chunk_index": chunk["chunk_index"],
                "audio_base64": audio_b64,
                "is_final": chunk["is_final"],
            }
            yield f"data: {JSONResponse(content=data).body.decode()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# ============================================================================
# Voice Profiles
# ============================================================================

@app.post("/v1/profiles", response_model=VoiceProfileResponse)
async def create_voice_profile(
    request: VoiceProfileCreate,
    db: DatabaseManager = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Create a new voice profile."""
    # Create voice profile in database
    repo = VoiceProfileRepository(db)
    profile = await repo.create(
        name=request.name,
        agent_id=request.agent_id,
        description=request.description,
        metadata=request.metadata,
    )

    # Create profile directory
    training_svc = get_training_service()
    training_svc.get_profile_dir(profile.id)

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        agent_id=profile.agent_id,
        status=profile.status,
        description=profile.description,
        metadata=profile.metadata_,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        sample_count=profile.sample_count,
        training_hours=profile.total_duration_ms / 3600000 if profile.total_duration_ms else None,
        model_path=profile.model_path,
    )


@app.get("/v1/profiles", response_model=VoiceProfileList)
async def list_voice_profiles(
    status: Optional[VoiceProfileStatus] = None,
    limit: int = 100,
    offset: int = 0,
    db: DatabaseManager = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """List all voice profiles."""
    repo = VoiceProfileRepository(db)
    profiles = await repo.list_all(status=status, limit=limit, offset=offset)

    profile_responses = [
        VoiceProfileResponse(
            id=profile.id,
            name=profile.name,
            agent_id=profile.agent_id,
            status=profile.status,
            description=profile.description,
            metadata=profile.metadata_,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            sample_count=profile.sample_count,
            training_hours=profile.total_duration_ms / 3600000 if profile.total_duration_ms else None,
            model_path=profile.model_path,
        )
        for profile in profiles
    ]

    return VoiceProfileList(profiles=profile_responses, total=len(profile_responses))


@app.get("/v1/profiles/{profile_id}", response_model=VoiceProfileResponse)
async def get_voice_profile(
    profile_id: str,
    db: DatabaseManager = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Get a voice profile by ID."""
    repo = VoiceProfileRepository(db)
    profile = await repo.get(profile_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        agent_id=profile.agent_id,
        status=profile.status,
        description=profile.description,
        metadata=profile.metadata_,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        sample_count=profile.sample_count,
        training_hours=profile.total_duration_ms / 3600000 if profile.total_duration_ms else None,
        model_path=profile.model_path,
    )


# ============================================================================
# Training Data
# ============================================================================

@app.post("/v1/profiles/{profile_id}/samples", response_model=TrainingDataResponse)
async def upload_training_samples(
    profile_id: str,
    request: TrainingDataUpload,
    training_svc: TrainingService = Depends(get_training_svc),
    _: bool = Depends(verify_api_key),
):
    """Upload training samples for a voice profile."""
    for sample in request.samples:
        await training_svc.add_training_sample(
            voice_profile_id=profile_id,
            audio_base64=sample.audio_base64,
            transcript=sample.transcript,
        )

    stats = training_svc.get_training_stats(profile_id)

    return TrainingDataResponse(
        voice_profile_id=profile_id,
        samples_uploaded=len(request.samples),
        total_samples=stats["sample_count"],
        total_duration_hours=stats["total_duration_hours"],
        ready_for_training=stats["ready_for_training"],
    )


# ============================================================================
# Training Jobs
# ============================================================================

@app.post("/v1/training", response_model=TrainingJobResponse)
async def start_training(
    request: TrainingJobCreate,
    training_svc: TrainingService = Depends(get_training_svc),
    _: bool = Depends(verify_api_key),
):
    """Start a training job for a voice profile."""
    try:
        job = await training_svc.start_training_job(
            voice_profile_id=request.voice_profile_id,
            use_lora=request.use_lora,
            lora_rank=request.lora_rank,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            gpu_type=request.gpu_type,
        )

        return TrainingJobResponse(
            id=job.id,
            voice_profile_id=job.voice_profile_id,
            status=job.status,
            progress=job.progress,
            gpu_type=job.gpu_type,
            started_at=job.started_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    training_svc: TrainingService = Depends(get_training_svc),
    _: bool = Depends(verify_api_key),
):
    """Get training job status."""
    job = training_svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return TrainingJobResponse(
        id=job.id,
        voice_profile_id=job.voice_profile_id,
        status=job.status,
        progress=job.progress,
        gpu_type=job.gpu_type,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


# ============================================================================
# Twilio Integration
# ============================================================================

@app.post("/v1/twilio/voice")
async def twilio_voice_webhook(
    request: Request,
    engine: VoiceEngine = Depends(get_voice_engine),
):
    """
    Twilio voice webhook for incoming calls.

    Returns TwiML with synthesized speech.
    """
    from xml.etree.ElementTree import Element, SubElement, tostring

    form_data = await request.form()
    call_sid = form_data.get("CallSid", "")
    caller = form_data.get("From", "")

    logger.info(f"Incoming Twilio call: {call_sid} from {caller}")

    # Generate TwiML response
    response = Element("Response")

    # For demo, use a simple greeting
    # In production, integrate with Human-Like agent flow
    say = SubElement(response, "Say")
    say.text = "Hello! This is a test of the native voice system. Goodbye!"

    # Alternative: Use <Play> with synthesized audio URL
    # play = SubElement(response, "Play")
    # play.text = f"https://your-domain/v1/twilio/audio/{call_sid}"

    twiml = tostring(response, encoding="unicode")
    return Response(content=twiml, media_type="application/xml")


@app.get("/v1/twilio/audio/{call_sid}")
async def twilio_audio(
    call_sid: str,
    text: str,
    voice_profile_id: str = "default",
    engine: VoiceEngine = Depends(get_voice_engine),
):
    """
    Generate audio for Twilio <Play> verb.

    Returns raw audio file for Twilio to play.
    """
    if not engine.is_loaded:
        await engine.load_model()

    result = await engine.synthesize(
        text=text,
        voice_profile_id=voice_profile_id,
        max_duration_ms=30000,
    )

    audio_bytes = engine.audio_to_bytes(
        result.audio,
        result.sample_rate,
        "wav",
    )

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={call_sid}.wav"
        }
    )


# ============================================================================
# Native Agent WebSocket (for Human-Like integration)
# ============================================================================

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/v1/native/ws/{session_id}")
async def native_agent_websocket(
    websocket: WebSocket,
    session_id: str,
    engine: VoiceEngine = Depends(get_voice_engine),
):
    """
    WebSocket endpoint for native voice agent.

    Protocol:
    - Client sends: {"type": "synthesize", "text": "...", "voice_profile_id": "..."}
    - Server sends: {"type": "audio", "audio_base64": "...", "duration_ms": ...}

    This enables real-time bidirectional voice communication.
    """
    await websocket.accept()
    logger.info(f"Native agent WebSocket connected: {session_id}")

    if not engine.is_loaded:
        await engine.load_model()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "synthesize":
                result = await engine.synthesize(
                    text=data["text"],
                    voice_profile_id=data.get("voice_profile_id", "default"),
                    speaker_id=data.get("speaker_id", 0),
                    temperature=data.get("temperature", 0.9),
                    topk=data.get("topk", 50),
                    max_duration_ms=data.get("max_duration_ms", 30000),
                )

                audio_b64 = engine.audio_to_base64(
                    result.audio,
                    result.sample_rate,
                    "wav",
                )

                await websocket.send_json({
                    "type": "audio",
                    "audio_base64": audio_b64,
                    "duration_ms": result.duration_ms,
                    "generation_time_ms": result.generation_time_ms,
                    "sample_rate": result.sample_rate,
                })

            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Native agent WebSocket disconnected: {session_id}")


# ============================================================================
# Twilio Bidirectional Streaming
# ============================================================================

from api.twilio_streaming import TwilioMediaStreamHandler, generate_stream_twiml

# Global streaming handler
_twilio_handler: Optional[TwilioMediaStreamHandler] = None


def get_twilio_handler() -> TwilioMediaStreamHandler:
    """Get or create Twilio streaming handler."""
    global _twilio_handler
    if _twilio_handler is None:
        _twilio_handler = TwilioMediaStreamHandler(get_engine())
    return _twilio_handler


@app.post("/v1/twilio/connect")
async def twilio_connect_webhook(
    request: Request,
    voice_profile_id: str = "default",
):
    """
    Twilio voice webhook that initiates bidirectional streaming.

    Returns TwiML that connects the call to our WebSocket endpoint.
    Configure this URL in your Twilio phone number settings.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")

    # Get the host from the request to build WebSocket URL
    host = request.headers.get("host", "localhost:8000")
    protocol = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{protocol}://{host}"

    twiml = generate_stream_twiml(
        call_sid=call_sid,
        stream_url=stream_url,
        voice_profile_id=voice_profile_id,
    )

    logger.info(f"Twilio call connected: {call_sid}, streaming to {stream_url}")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/v1/twilio/stream/{call_sid}")
async def twilio_media_stream(
    websocket: WebSocket,
    call_sid: str,
):
    """
    Twilio Media Stream WebSocket endpoint.

    This receives real-time audio from Twilio and can send
    synthesized audio back.
    """
    handler = get_twilio_handler()
    await handler.handle_connection(websocket, call_sid)


# ============================================================================
# Run Server
# ============================================================================

def run():
    """Run the API server."""
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
