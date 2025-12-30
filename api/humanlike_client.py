"""
Human-Like Integration Client - Native Voice Agent Support

This module provides the client interface for Human-Like to integrate
with the CSM Voice Service. It enables the "native" agent type that
uses self-hosted voice synthesis instead of ElevenLabs.

Usage in Human-Like:
    from csm_voice import NativeVoiceClient

    client = NativeVoiceClient(api_url="https://voice.humanlike.io")

    # Synthesize speech for agent response
    audio = await client.synthesize(
        text="Hello, how can I help you today?",
        voice_profile_id="agent-123-voice"
    )
"""

import asyncio
import base64
import logging
import json
from typing import Optional, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
import httpx
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Configuration for native voice synthesis."""
    voice_profile_id: str
    speaker_id: int = 0
    temperature: float = 0.9
    topk: int = 50
    output_format: str = "wav"
    max_duration_ms: int = 30000


@dataclass
class SynthesisResult:
    """Result from voice synthesis."""
    audio_bytes: bytes
    duration_ms: int
    sample_rate: int
    format: str
    generation_time_ms: int
    cached: bool


class NativeVoiceClient:
    """
    Client for Human-Like to use the CSM Voice Service.

    This client provides both REST and WebSocket interfaces for
    voice synthesis, supporting both batch and streaming use cases.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws_connection: Optional[WebSocketClientProtocol] = None

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
            )
        return self._http_client

    async def close(self):
        """Close client connections."""
        if self._http_client:
            await self._http_client.aclose()
        if self._ws_connection:
            await self._ws_connection.close()

    # =========================================================================
    # Health & Status
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        client = await self._get_client()
        response = await client.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()

    async def is_ready(self) -> bool:
        """Check if service is ready for synthesis."""
        try:
            health = await self.health_check()
            return health.get("status") == "healthy" and health.get("model_loaded", False)
        except Exception:
            return False

    # =========================================================================
    # Voice Synthesis
    # =========================================================================

    async def synthesize(
        self,
        text: str,
        voice_config: VoiceConfig,
        context: Optional[list] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            context: Optional conversation context

        Returns:
            SynthesisResult with audio bytes
        """
        client = await self._get_client()

        payload = {
            "voice_profile_id": voice_config.voice_profile_id,
            "text": text,
            "speaker_id": voice_config.speaker_id,
            "temperature": voice_config.temperature,
            "topk": voice_config.topk,
            "output_format": voice_config.output_format,
            "max_duration_ms": voice_config.max_duration_ms,
        }

        if context:
            payload["context"] = context

        response = await client.post(
            f"{self.api_url}/v1/synthesize",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return SynthesisResult(
            audio_bytes=base64.b64decode(data["audio_base64"]),
            duration_ms=data["duration_ms"],
            sample_rate=data["sample_rate"],
            format=data["format"],
            generation_time_ms=data["generation_time_ms"],
            cached=data.get("cached", False),
        )

    async def synthesize_streaming(
        self,
        text: str,
        voice_config: VoiceConfig,
        on_chunk: Optional[Callable[[bytes, int, bool], None]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized audio chunk by chunk.

        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            on_chunk: Optional callback for each chunk

        Yields:
            Audio bytes for each chunk
        """
        client = await self._get_client()

        payload = {
            "voice_profile_id": voice_config.voice_profile_id,
            "text": text,
            "speaker_id": voice_config.speaker_id,
            "temperature": voice_config.temperature,
            "topk": voice_config.topk,
            "output_format": voice_config.output_format,
            "stream": True,
        }

        async with client.stream(
            "POST",
            f"{self.api_url}/v1/synthesize/stream",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    audio_bytes = base64.b64decode(data["audio_base64"])

                    if on_chunk:
                        on_chunk(audio_bytes, data["chunk_index"], data["is_final"])

                    yield audio_bytes

    # =========================================================================
    # Voice Profile Management
    # =========================================================================

    async def create_voice_profile(
        self,
        name: str,
        agent_id: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new voice profile for an agent."""
        client = await self._get_client()

        payload = {
            "name": name,
            "agent_id": agent_id,
            "description": description,
            "metadata": metadata or {},
        }

        response = await client.post(
            f"{self.api_url}/v1/profiles",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def list_voice_profiles(self) -> Dict[str, Any]:
        """List all voice profiles."""
        client = await self._get_client()
        response = await client.get(f"{self.api_url}/v1/profiles")
        response.raise_for_status()
        return response.json()

    async def get_voice_profile(self, profile_id: str) -> Dict[str, Any]:
        """Get a specific voice profile."""
        client = await self._get_client()
        response = await client.get(f"{self.api_url}/v1/profiles/{profile_id}")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Training
    # =========================================================================

    async def upload_training_samples(
        self,
        profile_id: str,
        samples: list[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Upload training samples for voice cloning.

        Args:
            profile_id: Voice profile ID
            samples: List of {"audio_base64": "...", "transcript": "..."}
        """
        client = await self._get_client()

        payload = {
            "voice_profile_id": profile_id,
            "samples": samples,
        }

        response = await client.post(
            f"{self.api_url}/v1/profiles/{profile_id}/samples",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def start_training(
        self,
        profile_id: str,
        use_lora: bool = True,
        lora_rank: int = 8,
        num_epochs: int = 10,
        gpu_type: str = "a100",
    ) -> Dict[str, Any]:
        """Start training for a voice profile."""
        client = await self._get_client()

        payload = {
            "voice_profile_id": profile_id,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "num_epochs": num_epochs,
            "gpu_type": gpu_type,
        }

        response = await client.post(
            f"{self.api_url}/v1/training",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status."""
        client = await self._get_client()
        response = await client.get(f"{self.api_url}/v1/training/{job_id}")
        response.raise_for_status()
        return response.json()


class NativeAgentSession:
    """
    WebSocket session for real-time native agent voice synthesis.

    This is the primary interface for Human-Like agents to use
    native voice in real-time conversations (e.g., phone calls).
    """

    def __init__(
        self,
        api_url: str,
        session_id: str,
        voice_config: VoiceConfig,
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url.replace("http", "ws").rstrip("/")
        self.session_id = session_id
        self.voice_config = voice_config
        self.api_key = api_key

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._on_audio_callback: Optional[Callable[[bytes, int], None]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    def on_audio(self, callback: Callable[[bytes, int], None]):
        """Set callback for received audio chunks."""
        self._on_audio_callback = callback

    async def connect(self):
        """Connect to the native agent WebSocket."""
        url = f"{self.api_url}/v1/native/ws/{self.session_id}"

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._ws = await websockets.connect(url, extra_headers=headers)
        self._connected = True
        logger.info(f"Connected to native agent session: {self.session_id}")

        # Start message receiver
        asyncio.create_task(self._receive_messages())

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        if self._ws:
            await self._ws.close()
            self._connected = False
            logger.info(f"Disconnected from native agent session: {self.session_id}")

    async def _receive_messages(self):
        """Receive and process WebSocket messages."""
        try:
            async for message in self._ws:
                data = json.loads(message)

                if data["type"] == "audio":
                    audio_bytes = base64.b64decode(data["audio_base64"])
                    duration_ms = data["duration_ms"]

                    if self._on_audio_callback:
                        self._on_audio_callback(audio_bytes, duration_ms)

                elif data["type"] == "pong":
                    pass  # Heartbeat response

        except websockets.ConnectionClosed:
            self._connected = False
            logger.info(f"WebSocket connection closed: {self.session_id}")

    async def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ):
        """
        Request speech synthesis over WebSocket.

        Audio will be delivered via the on_audio callback.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to WebSocket")

        message = {
            "type": "synthesize",
            "text": text,
            "voice_profile_id": self.voice_config.voice_profile_id,
            "speaker_id": speaker_id or self.voice_config.speaker_id,
            "temperature": temperature or self.voice_config.temperature,
            "topk": topk or self.voice_config.topk,
        }

        await self._ws.send(json.dumps(message))

    async def ping(self):
        """Send heartbeat ping."""
        if self.is_connected:
            await self._ws.send(json.dumps({"type": "ping"}))


# =============================================================================
# Twilio Integration Helper
# =============================================================================

class TwilioVoiceIntegration:
    """
    Helper for integrating native voice with Twilio calls.

    This handles the bidirectional audio streaming between
    Twilio Media Streams and the CSM voice service.
    """

    def __init__(
        self,
        voice_client: NativeVoiceClient,
        voice_config: VoiceConfig,
    ):
        self.voice_client = voice_client
        self.voice_config = voice_config

    async def generate_response_audio(
        self,
        text: str,
        call_sid: str,
    ) -> bytes:
        """
        Generate audio response for a Twilio call.

        Returns audio in mulaw format suitable for Twilio.
        """
        # Use WAV first, then convert
        result = await self.voice_client.synthesize(
            text=text,
            voice_config=self.voice_config,
        )

        # For Twilio, we need 8kHz mulaw audio
        # In production, this conversion would be done server-side
        return result.audio_bytes

    def create_twiml_play_url(
        self,
        text: str,
        base_url: str,
    ) -> str:
        """
        Create a TwiML-compatible URL for playing synthesized audio.

        Use this in <Play> verbs:
            <Play>{url}</Play>
        """
        import urllib.parse
        params = {
            "text": text,
            "voice_profile_id": self.voice_config.voice_profile_id,
        }
        return f"{base_url}/v1/twilio/audio?{urllib.parse.urlencode(params)}"


# =============================================================================
# Human-Like Agent Adapter
# =============================================================================

class NativeAgentAdapter:
    """
    Adapter that plugs into the Human-Like agent framework.

    This replaces ElevenLabs voice synthesis with native CSM voices.

    Example usage in Human-Like:
        agent = Agent(
            voice_adapter=NativeAgentAdapter(
                api_url="https://voice.humanlike.io",
                voice_profile_id="sales-rep-john"
            )
        )
    """

    AGENT_TYPE = "native"

    def __init__(
        self,
        api_url: str,
        voice_profile_id: str,
        api_key: Optional[str] = None,
        speaker_id: int = 0,
        temperature: float = 0.9,
    ):
        self.voice_client = NativeVoiceClient(api_url, api_key)
        self.voice_config = VoiceConfig(
            voice_profile_id=voice_profile_id,
            speaker_id=speaker_id,
            temperature=temperature,
        )
        self._session: Optional[NativeAgentSession] = None

    async def initialize(self):
        """Initialize the adapter."""
        if not await self.voice_client.is_ready():
            raise RuntimeError("Voice service not ready")

    async def close(self):
        """Clean up resources."""
        await self.voice_client.close()
        if self._session:
            await self._session.disconnect()

    async def text_to_speech(
        self,
        text: str,
        **kwargs,
    ) -> bytes:
        """
        Convert text to speech audio.

        This is the main interface called by Human-Like agents.
        """
        result = await self.voice_client.synthesize(
            text=text,
            voice_config=self.voice_config,
        )
        return result.audio_bytes

    async def text_to_speech_streaming(
        self,
        text: str,
        on_chunk: Callable[[bytes], None],
        **kwargs,
    ):
        """
        Stream text to speech audio.

        Calls on_chunk for each audio chunk received.
        """
        async for chunk in self.voice_client.synthesize_streaming(
            text=text,
            voice_config=self.voice_config,
        ):
            on_chunk(chunk)

    async def create_call_session(
        self,
        session_id: str,
        on_audio: Callable[[bytes, int], None],
    ) -> NativeAgentSession:
        """
        Create a real-time session for phone calls.

        Args:
            session_id: Unique session identifier (e.g., call SID)
            on_audio: Callback when audio is ready

        Returns:
            NativeAgentSession for real-time synthesis
        """
        session = NativeAgentSession(
            api_url=self.voice_client.api_url,
            session_id=session_id,
            voice_config=self.voice_config,
            api_key=self.voice_client.api_key,
        )

        session.on_audio(on_audio)
        await session.connect()

        self._session = session
        return session

    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about the configured voice."""
        return {
            "type": self.AGENT_TYPE,
            "profile_id": self.voice_config.voice_profile_id,
            "speaker_id": self.voice_config.speaker_id,
            "temperature": self.voice_config.temperature,
        }
