"""
Twilio Bidirectional Media Streams Integration

This module handles real-time bidirectional audio streaming between
Twilio phone calls and the CSM voice synthesis engine.

Features:
- WebSocket connection for Twilio Media Streams
- Real-time audio synthesis and streaming
- Mulaw/PCMU conversion for Twilio
- Low-latency voice response

Usage:
    When a call comes in, Twilio connects to /v1/twilio/stream/{call_sid}
    and streams audio bidirectionally.
"""

import asyncio
import base64
import json
import logging
import struct
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class TwilioMediaState:
    """State for a Twilio media stream connection."""
    call_sid: str
    stream_sid: Optional[str] = None
    account_sid: Optional[str] = None
    connected: bool = False
    audio_track: str = "inbound"
    sequence_number: int = 0
    # Audio buffer for incoming audio
    incoming_audio_buffer: bytes = field(default_factory=bytes)
    # Callback for when speech is detected
    on_speech_callback: Optional[Callable[[str], None]] = None


class MulawCodec:
    """
    Mulaw (PCMU) audio codec for Twilio.

    Twilio uses 8kHz mulaw-encoded audio for phone calls.
    We need to convert between linear PCM and mulaw.
    """

    # Mulaw encoding table
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33
    MULAW_CLIP = 32635

    @staticmethod
    def linear_to_mulaw(sample: int) -> int:
        """Convert a 16-bit linear PCM sample to 8-bit mulaw."""
        sign = (sample >> 8) & 0x80
        if sign:
            sample = -sample
        if sample > MulawCodec.MULAW_CLIP:
            sample = MulawCodec.MULAW_CLIP
        sample = sample + MulawCodec.MULAW_BIAS

        exponent = 7
        for i in range(7, 0, -1):
            if sample >= (1 << (i + 3)):
                exponent = i
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
        return mulaw

    @staticmethod
    def mulaw_to_linear(mulaw: int) -> int:
        """Convert an 8-bit mulaw sample to 16-bit linear PCM."""
        mulaw = ~mulaw & 0xFF
        sign = mulaw & 0x80
        exponent = (mulaw >> 4) & 0x07
        mantissa = mulaw & 0x0F
        sample = ((mantissa << 3) + MulawCodec.MULAW_BIAS) << exponent
        if sign:
            sample = -sample
        return sample

    @classmethod
    def encode_pcm_to_mulaw(cls, pcm_data: np.ndarray) -> bytes:
        """
        Encode PCM audio to mulaw format.

        Args:
            pcm_data: NumPy array of 16-bit PCM samples

        Returns:
            Mulaw encoded bytes
        """
        mulaw_samples = []
        for sample in pcm_data.flatten():
            # Convert float to int16 if needed
            if pcm_data.dtype == np.float32:
                sample = int(sample * 32767)
            mulaw_samples.append(cls.linear_to_mulaw(int(sample)))
        return bytes(mulaw_samples)

    @classmethod
    def decode_mulaw_to_pcm(cls, mulaw_data: bytes) -> np.ndarray:
        """
        Decode mulaw audio to PCM format.

        Args:
            mulaw_data: Mulaw encoded bytes

        Returns:
            NumPy array of 16-bit PCM samples
        """
        pcm_samples = []
        for mulaw_byte in mulaw_data:
            pcm_samples.append(cls.mulaw_to_linear(mulaw_byte))
        return np.array(pcm_samples, dtype=np.int16)


class AudioResampler:
    """
    Resample audio between different sample rates.

    CSM operates at 24kHz, but Twilio uses 8kHz.
    """

    @staticmethod
    def resample_to_8k(audio: np.ndarray, source_rate: int = 24000) -> np.ndarray:
        """Resample audio from source rate to 8kHz for Twilio."""
        if source_rate == 8000:
            return audio

        # Simple decimation for downsampling
        ratio = source_rate // 8000
        return audio[::ratio]

    @staticmethod
    def resample_from_8k(audio: np.ndarray, target_rate: int = 24000) -> np.ndarray:
        """Resample audio from 8kHz to target rate."""
        if target_rate == 8000:
            return audio

        # Simple interpolation for upsampling
        ratio = target_rate // 8000
        upsampled = np.zeros(len(audio) * ratio, dtype=audio.dtype)
        upsampled[::ratio] = audio

        # Simple linear interpolation
        for i in range(len(audio) - 1):
            for j in range(1, ratio):
                upsampled[i * ratio + j] = (
                    audio[i] * (ratio - j) + audio[i + 1] * j
                ) // ratio

        return upsampled


class TwilioMediaStreamHandler:
    """
    Handles Twilio bidirectional media stream WebSocket connections.

    This is the core handler for real-time phone call audio.
    """

    def __init__(
        self,
        voice_engine,  # VoiceEngine instance
        voice_profile_id: str = "default",
        speaker_id: int = 0,
    ):
        self.voice_engine = voice_engine
        self.voice_profile_id = voice_profile_id
        self.speaker_id = speaker_id
        self.codec = MulawCodec()
        self.resampler = AudioResampler()

        # Active connections
        self._connections: Dict[str, TwilioMediaState] = {}

    async def handle_connection(
        self,
        websocket: WebSocket,
        call_sid: str,
        on_transcript: Optional[Callable[[str], None]] = None,
    ):
        """
        Handle a Twilio Media Stream WebSocket connection.

        Args:
            websocket: FastAPI WebSocket
            call_sid: Twilio Call SID
            on_transcript: Callback when transcript is ready (for STT integration)
        """
        await websocket.accept()

        state = TwilioMediaState(
            call_sid=call_sid,
            connected=True,
            on_speech_callback=on_transcript,
        )
        self._connections[call_sid] = state

        logger.info(f"Twilio Media Stream connected: {call_sid}")

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                event = data.get("event")

                if event == "connected":
                    await self._handle_connected(state, data)

                elif event == "start":
                    await self._handle_start(state, data)

                elif event == "media":
                    await self._handle_media(state, data, websocket)

                elif event == "stop":
                    await self._handle_stop(state, data)
                    break

                elif event == "mark":
                    await self._handle_mark(state, data)

        except WebSocketDisconnect:
            logger.info(f"Twilio Media Stream disconnected: {call_sid}")

        finally:
            state.connected = False
            self._connections.pop(call_sid, None)

    async def _handle_connected(self, state: TwilioMediaState, data: Dict[str, Any]):
        """Handle connected event."""
        logger.info(f"Twilio stream connected: {data}")

    async def _handle_start(self, state: TwilioMediaState, data: Dict[str, Any]):
        """Handle start event with stream metadata."""
        start_info = data.get("start", {})
        state.stream_sid = start_info.get("streamSid")
        state.account_sid = start_info.get("accountSid")
        state.audio_track = start_info.get("track", "inbound")

        logger.info(
            f"Stream started: stream_sid={state.stream_sid}, "
            f"track={state.audio_track}"
        )

    async def _handle_media(
        self,
        state: TwilioMediaState,
        data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle incoming media (audio from caller)."""
        media = data.get("media", {})
        payload = media.get("payload", "")

        if not payload:
            return

        # Decode base64 mulaw audio
        mulaw_audio = base64.b64decode(payload)

        # Add to buffer (for speech detection/STT)
        state.incoming_audio_buffer += mulaw_audio

        # In a production system, you would:
        # 1. Buffer audio until you detect speech end
        # 2. Send to STT (Whisper, Google, etc.)
        # 3. Process the transcript
        # 4. Generate response with CSM
        # 5. Stream back to Twilio

    async def _handle_stop(self, state: TwilioMediaState, data: Dict[str, Any]):
        """Handle stop event."""
        logger.info(f"Stream stopped: {state.call_sid}")

    async def _handle_mark(self, state: TwilioMediaState, data: Dict[str, Any]):
        """Handle mark event (playback marker)."""
        mark_name = data.get("mark", {}).get("name")
        logger.debug(f"Mark reached: {mark_name}")

    async def send_audio(
        self,
        websocket: WebSocket,
        state: TwilioMediaState,
        audio_tensor,  # torch.Tensor
        sample_rate: int = 24000,
    ):
        """
        Send synthesized audio to Twilio.

        Args:
            websocket: The WebSocket connection
            state: Media stream state
            audio_tensor: Audio tensor from CSM
            sample_rate: Audio sample rate
        """
        import torch

        # Convert tensor to numpy
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor

        # Normalize to int16
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)

        # Resample to 8kHz
        audio_8k = self.resampler.resample_to_8k(audio_np, sample_rate)

        # Encode to mulaw
        mulaw_audio = self.codec.encode_pcm_to_mulaw(audio_8k)

        # Send in chunks (Twilio expects ~20ms chunks = 160 samples at 8kHz)
        chunk_size = 160
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")

            state.sequence_number += 1
            message = {
                "event": "media",
                "streamSid": state.stream_sid,
                "media": {
                    "payload": payload,
                },
            }

            await websocket.send_text(json.dumps(message))

            # Small delay to prevent buffer overflow
            await asyncio.sleep(0.018)  # ~18ms per chunk

    async def synthesize_and_send(
        self,
        websocket: WebSocket,
        state: TwilioMediaState,
        text: str,
    ):
        """
        Synthesize speech and send to Twilio.

        This is the main method for generating voice responses.
        """
        logger.info(f"Synthesizing for Twilio: {text[:50]}...")

        # Generate audio with CSM
        result = await self.voice_engine.synthesize(
            text=text,
            voice_profile_id=self.voice_profile_id,
            speaker_id=self.speaker_id,
        )

        # Send to Twilio
        await self.send_audio(
            websocket=websocket,
            state=state,
            audio_tensor=result.audio,
            sample_rate=result.sample_rate,
        )

        # Send mark to track playback completion
        mark_message = {
            "event": "mark",
            "streamSid": state.stream_sid,
            "mark": {"name": f"synthesis_{state.sequence_number}"},
        }
        await websocket.send_text(json.dumps(mark_message))

    async def clear_audio_buffer(
        self,
        websocket: WebSocket,
        state: TwilioMediaState,
    ):
        """Clear the Twilio audio buffer (e.g., for barge-in)."""
        message = {
            "event": "clear",
            "streamSid": state.stream_sid,
        }
        await websocket.send_text(json.dumps(message))
        logger.debug("Cleared Twilio audio buffer")


# =============================================================================
# FastAPI Integration
# =============================================================================

def create_twilio_stream_endpoint(app, voice_engine):
    """
    Add Twilio Media Stream endpoint to FastAPI app.

    This creates the WebSocket endpoint that Twilio connects to.
    """
    from fastapi import WebSocket, Query

    handler = TwilioMediaStreamHandler(voice_engine)

    @app.websocket("/v1/twilio/stream/{call_sid}")
    async def twilio_media_stream(
        websocket: WebSocket,
        call_sid: str,
        voice_profile_id: str = Query("default"),
    ):
        """
        Twilio Media Stream WebSocket endpoint.

        Connect TwiML to this endpoint:
            <Connect>
                <Stream url="wss://your-domain/v1/twilio/stream/{CallSid}">
                    <Parameter name="voice_profile_id" value="agent-voice-123"/>
                </Stream>
            </Connect>
        """
        handler.voice_profile_id = voice_profile_id
        await handler.handle_connection(websocket, call_sid)

    return handler


def generate_stream_twiml(
    call_sid: str,
    stream_url: str,
    voice_profile_id: str = "default",
) -> str:
    """
    Generate TwiML for bidirectional streaming.

    Returns TwiML that connects the call to our WebSocket endpoint.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}/v1/twilio/stream/{call_sid}">
            <Parameter name="voice_profile_id" value="{voice_profile_id}"/>
        </Stream>
    </Connect>
</Response>"""
