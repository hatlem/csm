"""
Full CSM Voice API with HTTP and WebSocket endpoints.

HTTP Endpoints:
- GET /health - Health check
- POST /synthesize - Batch synthesis

WebSocket Endpoints:
- /v1/native/ws/{session_id} - Real-time streaming synthesis for Human-Like

Audio Format:
- WebSocket returns mulaw 8kHz for Twilio Media Streams
- HTTP returns WAV base64
"""
import sys
sys.path.insert(0, "/workspace/csm")

import asyncio
import base64
import io
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import torch
import torchaudio
import torchaudio.functional as F

from generator import load_csm_1b

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CSM Voice API",
    description="Real-time voice synthesis using Sesame's CSM model",
    version="1.0.0",
)

# Global model
generator = None


# ============================================================================
# Pydantic Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    text: str
    speaker: int = 0
    max_audio_length_ms: int = 30000
    output_format: str = "wav"  # "wav" or "mulaw"


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_ms: int
    generation_time_ms: int
    format: str


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def load_model():
    global generator
    print("Loading CSM model...")
    start = time.time()
    generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded in {time.time() - start:.2f}s")
    print(f"Sample rate: {generator.sample_rate}")


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Run synthesis in thread pool
    loop = asyncio.get_event_loop()
    start = time.time()

    audio = await loop.run_in_executor(
        None,
        lambda: generator.generate(
            text=request.text,
            speaker=request.speaker,
            context=[],
            max_audio_length_ms=request.max_audio_length_ms,
        ),
    )

    generation_time = time.time() - start
    duration_ms = int(len(audio) / generator.sample_rate * 1000)

    # Convert to requested format
    if request.output_format == "mulaw":
        audio_bytes = convert_to_mulaw(audio, generator.sample_rate)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        output_format = "mulaw_8khz"
        sample_rate = 8000
    else:
        # WAV format
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        output_format = "wav"
        sample_rate = generator.sample_rate

    return SynthesizeResponse(
        audio_base64=audio_base64,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        generation_time_ms=int(generation_time * 1000),
        format=output_format,
    )


# ============================================================================
# WebSocket Endpoint (for Human-Like NativeCSMProvider)
# ============================================================================

@app.websocket("/v1/native/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice synthesis.

    Protocol:
    - Client sends: {"type": "synthesize", "text": "...", "voice_profile_id": "..."}
    - Server sends: {"type": "audio", "audio_base64": "...", "duration_ms": ...}
    - Client sends: {"type": "ping"} -> Server sends: {"type": "pong"}
    """
    await websocket.accept()
    print(f"WebSocket connected: session={session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "synthesize":
                text = message.get("text", "")
                voice_profile_id = message.get("voice_profile_id", "default")

                if text:
                    await synthesize_ws(websocket, text, voice_profile_id)

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        print(f"WebSocket error: session={session_id}, error={e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass


async def synthesize_ws(websocket: WebSocket, text: str, voice_profile_id: str):
    """Synthesize text and send audio over WebSocket."""
    if generator is None:
        await websocket.send_json({"type": "error", "error": "Model not loaded"})
        return

    start_time = time.time()

    # Run synthesis in thread pool
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(
        None,
        lambda: generator.generate(
            text=text,
            speaker=0,  # Could map voice_profile_id to speaker
            context=[],
            max_audio_length_ms=30000,
        ),
    )

    generation_time = time.time() - start_time
    duration_ms = int(len(audio) / generator.sample_rate * 1000)

    # Convert to mulaw 8kHz for Twilio
    mulaw_bytes = convert_to_mulaw(audio, generator.sample_rate)
    audio_base64 = base64.b64encode(mulaw_bytes).decode()

    # Send response
    response = {
        "type": "audio",
        "audio_base64": audio_base64,
        "duration_ms": duration_ms,
        "generation_time_ms": int(generation_time * 1000),
        "format": "mulaw_8khz",
    }

    await websocket.send_json(response)


# ============================================================================
# Audio Conversion Utilities
# ============================================================================

def convert_to_mulaw(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to mulaw 8kHz bytes for Twilio."""
    # Ensure audio is on CPU and 2D [1, samples]
    audio = audio.cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Resample from 24kHz to 8kHz
    resampled = F.resample(audio, sample_rate, 8000)

    # Normalize to [-1, 1]
    if resampled.abs().max() > 0:
        resampled = resampled / resampled.abs().max()

    # Apply mu-law encoding (256 quantization levels = 8-bit)
    mulaw = torchaudio.functional.mu_law_encoding(resampled, quantization_channels=256)

    # Convert to bytes (uint8)
    mulaw_bytes = mulaw.squeeze().to(torch.uint8).numpy().tobytes()

    return mulaw_bytes


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
