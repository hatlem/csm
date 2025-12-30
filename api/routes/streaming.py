"""
Streaming synthesis endpoints.

Provides:
- WebSocket streaming for real-time synthesis
- Twilio Media Streams integration
- Server-Sent Events (SSE) support
"""

import asyncio
import base64
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.core.dependencies import get_container
from api.core.logging import get_logger
from api.core.metrics import get_metrics

logger = get_logger(__name__)
router = APIRouter()


class StreamingConfig(BaseModel):
    """Streaming configuration."""

    voice_profile_id: str = "default"
    sample_rate: int = 24000
    chunk_size_ms: int = 100  # Audio chunk duration
    buffer_size_ms: int = 500  # Pre-buffer duration


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech synthesis.

    Protocol:
    1. Client connects and sends config message
    2. Client sends text messages for synthesis
    3. Server streams back audio chunks
    4. Client sends "done" to close

    Message formats:
    - Config: {"type": "config", "voice_profile_id": "...", "sample_rate": 24000}
    - Text: {"type": "text", "content": "Hello world"}
    - Audio: {"type": "audio", "data": "<base64>", "sequence": 1}
    - Done: {"type": "done"}
    """
    await websocket.accept()
    metrics = get_metrics()
    container = get_container()

    config = StreamingConfig()
    sequence = 0

    try:
        voice_engine = await container.get("voice_engine")

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            msg_type = data.get("type")

            if msg_type == "config":
                config = StreamingConfig(**data)
                await websocket.send_json({
                    "type": "config_ack",
                    "voice_profile_id": config.voice_profile_id,
                })

            elif msg_type == "text":
                text = data.get("content", "")
                if not text:
                    continue

                # Synthesize
                start_time = time.time()
                result = await voice_engine.synthesize(
                    text=text,
                    voice_profile_id=config.voice_profile_id,
                )

                # Convert to base64
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

                sequence += 1
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_base64,
                    "sequence": sequence,
                    "duration_ms": result.duration_ms,
                    "generation_time_ms": int((time.time() - start_time) * 1000),
                })

            elif msg_type == "done":
                await websocket.send_json({"type": "close_ack"})
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/twilio/stream")
async def twilio_media_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket endpoint.

    Handles bidirectional audio streaming for phone calls.

    Twilio message types:
    - connected: Connection established
    - start: Stream starting, contains streamSid
    - media: Audio chunk (base64 mulaw 8kHz)
    - stop: Stream ending
    """
    await websocket.accept()
    logger.info("Twilio Media Stream connected")

    stream_sid = None
    container = get_container()

    try:
        voice_engine = await container.get("voice_engine")

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            event = data.get("event")

            if event == "connected":
                logger.info("Twilio stream connected")

            elif event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                logger.info(f"Twilio stream started: {stream_sid}")

            elif event == "media":
                # Incoming audio from caller (for future STT integration)
                # payload = data.get("media", {}).get("payload")
                pass

            elif event == "stop":
                logger.info(f"Twilio stream stopped: {stream_sid}")
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Twilio WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def send_twilio_audio(
    websocket: WebSocket,
    stream_sid: str,
    audio_base64: str,
):
    """Send audio to Twilio Media Stream."""
    await websocket.send_json({
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": audio_base64,
        },
    })


async def send_twilio_mark(
    websocket: WebSocket,
    stream_sid: str,
    mark_name: str,
):
    """Send a mark event to Twilio for synchronization."""
    await websocket.send_json({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {
            "name": mark_name,
        },
    })


@router.post("/stream/sse")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint for streaming synthesis.

    Request body: {"text": "...", "voice_profile_id": "..."}

    Streams audio chunks as SSE events.
    """
    body = await request.json()
    text = body.get("text", "")
    voice_profile_id = body.get("voice_profile_id", "default")

    if not text:
        return {"error": "Text is required"}

    container = get_container()
    voice_engine = await container.get("voice_engine")

    async def generate():
        try:
            # Synthesize full audio
            result = await voice_engine.synthesize(
                text=text,
                voice_profile_id=voice_profile_id,
            )

            # Convert to base64
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

            # Send as SSE
            yield f"data: {json.dumps({'type': 'audio', 'data': audio_base64, 'duration_ms': result.duration_ms})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
