"""
WebSocket handler for real-time CSM voice synthesis.

Implements the protocol expected by Human-Like's NativeCSMProvider:
- Receives: {"type": "synthesize", "text": "...", "voice_profile_id": "..."}
- Returns: {"type": "audio", "audio_base64": "...", "duration_ms": ...}

Also supports raw mulaw output for Twilio Media Streams compatibility.
"""
import asyncio
import base64
import io
import json
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
import torch
import torchaudio
import torchaudio.functional as F


class CSMWebSocketHandler:
    """Handles WebSocket connections for real-time voice synthesis."""

    def __init__(self, generator):
        self.generator = generator
        self.sample_rate = generator.sample_rate  # 24000 Hz from CSM

    async def handle_connection(self, websocket: WebSocket, session_id: str):
        """Handle a WebSocket connection for voice synthesis."""
        await websocket.accept()
        print(f"WebSocket connected: session={session_id}")

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                msg_type = message.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "synthesize":
                    text = message.get("text", "")
                    voice_profile_id = message.get("voice_profile_id", "default")

                    if text:
                        await self._synthesize_and_send(
                            websocket, text, voice_profile_id
                        )

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

    async def _synthesize_and_send(
        self,
        websocket: WebSocket,
        text: str,
        voice_profile_id: str,
    ):
        """Synthesize text and send audio back via WebSocket."""
        start_time = time.time()

        # Run synthesis in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            self._generate_audio,
            text,
            voice_profile_id,
        )

        generation_time = time.time() - start_time
        duration_ms = int(len(audio) / self.sample_rate * 1000)

        # Convert to mulaw 8kHz for Twilio (or keep as WAV)
        # Twilio Media Streams expects mulaw 8kHz base64
        mulaw_audio = self._convert_to_mulaw(audio)
        audio_base64 = base64.b64encode(mulaw_audio).decode()

        # Send audio response
        response = {
            "type": "audio",
            "audio_base64": audio_base64,
            "duration_ms": duration_ms,
            "generation_time_ms": int(generation_time * 1000),
            "format": "mulaw_8khz",
        }

        await websocket.send_json(response)

    def _generate_audio(self, text: str, voice_profile_id: str) -> torch.Tensor:
        """Generate audio from text (runs in thread pool)."""
        # Map voice_profile_id to speaker index
        # For now, use speaker 0 for all profiles
        # Later: load custom LoRA weights based on profile
        speaker = 0

        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=30000,
        )

        return audio

    def _convert_to_mulaw(self, audio: torch.Tensor) -> bytes:
        """Convert audio to mulaw 8kHz format for Twilio."""
        # CSM outputs at 24kHz, Twilio needs 8kHz mulaw

        # Ensure audio is on CPU and 1D
        audio = audio.cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample from 24kHz to 8kHz
        resampled = F.resample(audio, self.sample_rate, 8000)

        # Normalize to [-1, 1]
        if resampled.abs().max() > 0:
            resampled = resampled / resampled.abs().max()

        # Convert to mulaw encoding
        # Torch's mu_law_encoding expects values in [-1, 1]
        mulaw = torchaudio.functional.mu_law_encoding(resampled, quantization_channels=256)

        # Convert to bytes (uint8)
        mulaw_bytes = mulaw.squeeze().to(torch.uint8).numpy().tobytes()

        return mulaw_bytes


def create_websocket_routes(app, generator):
    """Add WebSocket routes to FastAPI app."""
    handler = CSMWebSocketHandler(generator)

    @app.websocket("/v1/native/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await handler.handle_connection(websocket, session_id)

    return handler
