# Human-Like Integration Guide

## Overview

The CSM Voice Service integrates with Human-Like via the `NativeCSMProvider` class, which handles real-time voice synthesis over WebSocket.

## Configuration

### Agent Voice Settings

To enable CSM voice for an agent, update the `voice_settings` JSON field in the database:

```json
{
  "voiceProviderStrategy": "native_csm",
  "nativeVoiceProfileId": "default",
  "runpodEndpointUrl": "https://1s8c803ve11jp3-8000.proxy.runpod.net"
}
```

### Via Admin UI

1. Go to Agent Settings
2. Find "Voice Configuration" section
3. Set:
   - **Voice Provider Strategy**: `native_csm`
   - **Native Voice Profile ID**: `default` (or custom profile)
   - **RunPod Endpoint URL**: `https://1s8c803ve11jp3-8000.proxy.runpod.net`

### Via Database

```sql
UPDATE agents
SET voice_settings = '{
  "voiceProviderStrategy": "native_csm",
  "nativeVoiceProfileId": "default",
  "runpodEndpointUrl": "https://1s8c803ve11jp3-8000.proxy.runpod.net"
}'::jsonb
WHERE id = 'your-agent-id';
```

## Type Definitions

From `server/services/types/agent-config-types.ts`:

```typescript
export type VoiceProviderStrategy =
  | 'elevenlabs_managed'  // ElevenLabs handles everything
  | 'orchestrated'        // Deepgram + LangGraph + ElevenLabs TTS
  | 'native_csm';         // Deepgram + LangGraph + RunPod CSM

export interface VoiceSettings {
  voiceProviderStrategy?: VoiceProviderStrategy;
  nativeVoiceProfileId?: string;  // Voice profile for CSM
  runpodEndpointUrl?: string;     // CSM service URL
  // ... other fields
}
```

## NativeCSMProvider

Location: `server/services/voice/providers/NativeCSMProvider.ts`

### How It Works

1. **Initialization**: Receives Twilio WebSocket + config
2. **Connection**: Connects to CSM WebSocket at `/v1/native/ws/{session_id}`
3. **Streaming**: Sends text chunks, receives mulaw audio
4. **Forwarding**: Forwards audio directly to Twilio

### WebSocket Protocol

**Client → CSM:**
```json
{"type": "synthesize", "text": "Hello world", "voice_profile_id": "default"}
{"type": "ping"}
{"type": "close"}
```

**CSM → Client:**
```json
{
  "type": "audio",
  "audio_base64": "...",
  "duration_ms": 1500,
  "generation_time_ms": 800,
  "format": "mulaw_8khz"
}
{"type": "pong"}
{"type": "error", "error": "..."}
```

### Audio Format

CSM returns audio in Twilio-ready format:
- **Encoding**: G.711 mu-law
- **Sample Rate**: 8,000 Hz
- **Bit Depth**: 8-bit (uint8)
- **Container**: Raw bytes, base64 encoded

The `NativeCSMProvider` forwards this directly to Twilio without further processing.

## Flow Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Twilio    │     │  Human-Like  │     │  CSM Service │
│              │     │              │     │   (RunPod)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       │  User speaks       │                    │
       │ ──────────────────▶│                    │
       │  (mulaw 8kHz)      │                    │
       │                    │                    │
       │                    │  Deepgram STT      │
       │                    │ ─────────────▶     │
       │                    │  (text)            │
       │                    │                    │
       │                    │  LangGraph         │
       │                    │ ─────────────▶     │
       │                    │  (response text)   │
       │                    │                    │
       │                    │  WebSocket         │
       │                    │ ──────────────────▶│
       │                    │  synthesize        │
       │                    │                    │
       │                    │                    │ CSM generates
       │                    │                    │ 24kHz PCM
       │                    │                    │
       │                    │                    │ Transcode to
       │                    │                    │ 8kHz mulaw
       │                    │                    │
       │                    │  audio response    │
       │                    │ ◀──────────────────│
       │                    │  (mulaw 8kHz)      │
       │                    │                    │
       │  Forward audio     │                    │
       │ ◀──────────────────│                    │
       │  (mulaw 8kHz)      │                    │
       │                    │                    │
       │  User hears        │                    │
       ▼                    ▼                    ▼
```

## Voice Provider Factory

Location: `server/services/voice/VoiceProviderFactory.ts`

The factory creates the appropriate provider based on `voiceProviderStrategy`:

```typescript
createProvider(twilioWs, config): IVoiceProvider {
  switch (config.providerConfig?.voiceProviderStrategy) {
    case 'native_csm':
      return new NativeCSMProvider(twilioWs, config);
    case 'orchestrated':
      return new OrchestratedProvider(twilioWs, config);
    case 'elevenlabs_managed':
    default:
      return new ElevenLabsManagedProvider(twilioWs, config);
  }
}
```

## Testing Integration

### 1. Verify CSM Service is Running

```bash
curl https://1s8c803ve11jp3-8000.proxy.runpod.net/health
# Expected: {"status":"healthy","model_loaded":true,"device":"cuda"}
```

### 2. Test Synthesis

```bash
curl -X POST https://1s8c803ve11jp3-8000.proxy.runpod.net/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "max_audio_length_ms": 5000}'
```

### 3. Test WebSocket (wscat)

```bash
npm install -g wscat
wscat -c "wss://1s8c803ve11jp3-8000.proxy.runpod.net/v1/native/ws/test-session"
> {"type": "ping"}
< {"type": "pong"}
> {"type": "synthesize", "text": "Hello world", "voice_profile_id": "default"}
< {"type": "audio", "audio_base64": "...", ...}
```

### 4. Make a Test Call

1. Configure an agent with `native_csm` strategy
2. Call the agent's phone number
3. Listen for CSM-generated voice

## Troubleshooting

### No Audio

1. Check CSM service is running: `GET /health`
2. Check agent config has correct `runpodEndpointUrl`
3. Check WebSocket connects (look for logs)

### Audio is Static/Noise

1. Verify CSM is returning `mulaw_8khz` format
2. Check `convert_to_mulaw` casts to `uint8`
3. Verify sample rate is 8000 Hz

### WebSocket Disconnects

1. Add keepalive pings every 30s
2. Check RunPod pod is still running
3. Verify session_id is unique per call

### High Latency

1. Use GPU closer to users (EU datacenter for EU users)
2. Consider upgrading to A100 for faster generation
3. Reduce `max_audio_length_ms` for shorter utterances
