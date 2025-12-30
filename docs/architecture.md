# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TWILIO (Phone Network)                         │
│                        Raw Audio Transport Layer                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (mulaw 8kHz)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         HUMAN-LIKE SERVER                                │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Deepgram   │    │  LangGraph   │    │    Voice Provider        │  │
│  │     STT      │───▶│   (Brain)    │───▶│  Factory + Providers     │  │
│  │              │    │              │    │                          │  │
│  │  mulaw→text  │    │  text→text   │    │  • ElevenLabsManaged     │  │
│  └──────────────┘    └──────────────┘    │  • Orchestrated          │  │
│                                          │  • NativeCSM ◀── NEW     │  │
│                                          └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (JSON + mulaw)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CSM VOICE SERVICE (RunPod)                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Server                               │  │
│  │                                                                    │  │
│  │  HTTP Endpoints:                                                   │  │
│  │  • GET  /health              - Health check                        │  │
│  │  • POST /synthesize          - Batch synthesis (WAV/mulaw)         │  │
│  │                                                                    │  │
│  │  WebSocket Endpoints:                                              │  │
│  │  • /v1/native/ws/{session}   - Real-time streaming                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      CSM Model (GPU)                              │  │
│  │                                                                    │  │
│  │  • Sesame CSM-1B (1B backbone + 100M decoder)                     │  │
│  │  • Mimi audio codec (24kHz)                                       │  │
│  │  • SilentCipher watermarking                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Audio Transcoding                               │  │
│  │                                                                    │  │
│  │  24kHz PCM → Resample → 8kHz → mu-law encode → uint8 bytes        │  │
│  │                                                                    │  │
│  │  CRITICAL: Must cast to uint8 before .tobytes()                   │  │
│  │  Otherwise: 4 bytes per sample instead of 1 = static/noise        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Voice Provider Strategies

Human-Like supports three voice provider strategies:

### 1. `elevenlabs_managed`
- ElevenLabs handles everything (STT + LLM + TTS)
- Simplest setup, highest cost
- Best for quick prototypes

### 2. `orchestrated`
- Deepgram STT + LangGraph LLM + ElevenLabs TTS API
- More control, moderate cost
- Good balance of quality and flexibility

### 3. `native_csm` (This System)
- Deepgram STT + LangGraph LLM + **Self-hosted CSM TTS**
- Full control, lowest marginal cost
- Best for scale and custom voices

## Audio Format Requirements

### Twilio Media Streams
- **Format**: G.711 mu-law (PCMU)
- **Sample Rate**: 8,000 Hz
- **Bit Depth**: 8-bit (1 byte per sample)
- **Encoding**: Base64 in JSON messages

### CSM Model Output
- **Format**: PCM (via Mimi codec)
- **Sample Rate**: 24,000 Hz
- **Bit Depth**: 16-bit float

### Transcoding Pipeline

```python
def convert_to_mulaw(audio: torch.Tensor, sample_rate: int) -> bytes:
    # 1. Resample 24kHz → 8kHz
    resampled = F.resample(audio, sample_rate, 8000)

    # 2. Normalize to [-1, 1]
    resampled = resampled / resampled.abs().max()

    # 3. Apply mu-law encoding (256 levels = 8-bit)
    mulaw = torchaudio.functional.mu_law_encoding(resampled, 256)

    # 4. CRITICAL: Cast to uint8 (1 byte per sample)
    mulaw_bytes = mulaw.squeeze().to(torch.uint8).numpy().tobytes()

    return mulaw_bytes
```

## Data Flow

### Inbound (User Speech)
```
User speaks → Twilio → mulaw 8kHz → Human-Like → Deepgram STT → Text
```

### Processing (AI Response)
```
Text → LangGraph (Brain) → Response Text
```

### Outbound (Agent Speech)
```
Response Text → CSM WebSocket → 24kHz PCM → Transcode → mulaw 8kHz → Twilio → User hears
```

## Latency Breakdown

| Stage | Typical Latency |
|-------|-----------------|
| Twilio → Server | ~50-100ms |
| Deepgram STT | ~100-200ms |
| LangGraph LLM | ~200-500ms |
| CSM TTS | ~500-1000ms (for 1s audio) |
| Transcode | ~10ms |
| Server → Twilio | ~50-100ms |
| **Total** | **~1-2 seconds** |

## GPU Requirements

| Config | VRAM | Speed | Cost/hr |
|--------|------|-------|---------|
| RTX 4090 | 24GB | ~3-4x RT | $0.59 |
| A100 40GB | 40GB | ~2x RT | $1.29 |
| A100 80GB | 80GB | ~1.5x RT | $1.89 |
| H100 | 80GB | ~1x RT | $3.89 |

*RT = Real-time (1x RT means 1 second of audio takes 1 second to generate)*
