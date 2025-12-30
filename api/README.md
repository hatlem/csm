# CSM Voice API Service

Production-ready voice synthesis API for Human-Like integration. This service provides voice cloning, training, and real-time synthesis capabilities using the CSM (Conversational Speech Model) from Sesame AI Labs.

## Features

- **Voice Synthesis**: Generate natural-sounding speech from text
- **Voice Cloning**: Train custom voices from audio samples
- **Real-time Streaming**: Low-latency audio generation via WebSocket
- **Twilio Integration**: Bidirectional streaming for phone calls
- **Native Agent Support**: Drop-in replacement for ElevenLabs in Human-Like

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-api.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Download the CSM model (requires HuggingFace access)
python -c "from huggingface_hub import snapshot_download; snapshot_download('sesame/csm-1b', local_dir='./models/csm-1b')"

# Run the server
python -m api.main
# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build the image
docker build -t csm-voice:latest .

# Run with docker-compose
docker-compose up -d

# With GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status, model state, and GPU info.

### Voice Synthesis

```bash
POST /v1/synthesize
```

Generate speech from text.

**Request:**
```json
{
  "voice_profile_id": "default",
  "text": "Hello, how can I help you today?",
  "speaker_id": 0,
  "temperature": 0.9,
  "topk": 50,
  "output_format": "wav"
}
```

**Response:**
```json
{
  "audio_base64": "UklGRi...",
  "duration_ms": 2500,
  "sample_rate": 24000,
  "format": "wav",
  "generation_time_ms": 850
}
```

### Streaming Synthesis

```bash
POST /v1/synthesize/stream
```

Server-Sent Events stream of audio chunks.

### Voice Profiles

```bash
# Create profile
POST /v1/profiles

# List profiles
GET /v1/profiles

# Get profile
GET /v1/profiles/{profile_id}

# Upload training samples
POST /v1/profiles/{profile_id}/samples
```

### Training

```bash
# Start training
POST /v1/training
{
  "voice_profile_id": "profile-123",
  "use_lora": true,
  "lora_rank": 8,
  "num_epochs": 10,
  "gpu_type": "a100"
}

# Check status
GET /v1/training/{job_id}
```

### Native Agent WebSocket

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/v1/native/ws/session-123');

// Request synthesis
ws.send(JSON.stringify({
  type: 'synthesize',
  text: 'Hello from the native agent!',
  voice_profile_id: 'agent-voice-123'
}));

// Receive audio
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'audio') {
    playAudio(data.audio_base64);
  }
};
```

### Twilio Integration

Configure your Twilio phone number to use these webhooks:

1. **Voice URL**: `POST https://your-domain/v1/twilio/connect`
2. The webhook returns TwiML that connects to the streaming endpoint
3. Real-time bidirectional audio flows via WebSocket

```xml
<!-- Generated TwiML -->
<Response>
    <Connect>
        <Stream url="wss://your-domain/v1/twilio/stream/{CallSid}">
            <Parameter name="voice_profile_id" value="agent-voice-123"/>
        </Stream>
    </Connect>
</Response>
```

## Human-Like Integration

### Using the Native Agent Adapter

```python
from api.humanlike_client import NativeAgentAdapter

# Create adapter for your agent
adapter = NativeAgentAdapter(
    api_url="https://voice.humanlike.io",
    voice_profile_id="sales-rep-john",
    api_key="your-api-key"
)

# Use in your agent
class MyAgent:
    def __init__(self):
        self.voice = adapter

    async def respond(self, text: str) -> bytes:
        return await self.voice.text_to_speech(text)
```

### Voice Cloning Workflow

1. **Create Voice Profile**
```python
from api.humanlike_client import NativeVoiceClient

client = NativeVoiceClient("https://voice.humanlike.io", api_key="...")

# Create profile
profile = await client.create_voice_profile(
    name="John's Voice",
    agent_id="agent-123"
)
```

2. **Upload Training Samples**
```python
# Upload 30+ minutes of audio samples
samples = [
    {"audio_base64": base64.b64encode(audio1).decode(), "transcript": "Hello, this is John..."},
    {"audio_base64": base64.b64encode(audio2).decode(), "transcript": "How can I help you..."},
    # ... more samples
]

await client.upload_training_samples(profile["id"], samples)
```

3. **Start Training**
```python
job = await client.start_training(
    profile_id=profile["id"],
    use_lora=True,
    gpu_type="a100"
)

# Monitor progress
while True:
    status = await client.get_training_status(job["id"])
    print(f"Progress: {status['progress']*100:.1f}%")
    if status["status"] in ("completed", "failed"):
        break
    await asyncio.sleep(30)
```

4. **Use the Cloned Voice**
```python
result = await client.synthesize(
    text="Hello, I'm John from Human-Like!",
    voice_config=VoiceConfig(voice_profile_id=profile["id"])
)

# Play audio
with open("output.wav", "wb") as f:
    f.write(result.audio_bytes)
```

## Configuration

Environment variables (prefix with `CSM_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Server host | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `API_KEY` | API authentication key | None |
| `MODEL_PATH` | Path to CSM model | `./models/csm-1b` |
| `MODEL_DEVICE` | Device (cuda/cpu/mps) | `cuda` |
| `MODEL_DTYPE` | Model dtype | `bfloat16` |
| `DATABASE_URL` | PostgreSQL URL | `postgresql://...` |
| `REDIS_URL` | Redis URL | `redis://...` |
| `TRAINING_GPU_PROVIDER` | Cloud GPU (runpod/vast/local) | `local` |
| `RUNPOD_API_KEY` | RunPod API key | None |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Human-Like Platform                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Agent 1   │  │   Agent 2   │  │   Agent N (Native)      │ │
│  │ (ElevenLabs)│  │ (ElevenLabs)│  │ NativeAgentAdapter      │ │
│  └─────────────┘  └─────────────┘  └───────────┬─────────────┘ │
└────────────────────────────────────────────────┼────────────────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │    CSM Voice Service      │
                                    │    (FastAPI + WebSocket)  │
                                    ├───────────────────────────┤
                                    │  /v1/synthesize           │
                                    │  /v1/native/ws/{session}  │
                                    │  /v1/twilio/stream/{call} │
                                    └───────────┬───────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
           ┌────────▼────────┐      ┌───────────▼───────────┐    ┌─────────▼─────────┐
           │  VoiceEngine    │      │   TrainingService     │    │    TwilioHandler  │
           │  (CSM Model)    │      │   (Cloud GPU)         │    │    (Media Stream) │
           └────────┬────────┘      └───────────┬───────────┘    └───────────────────┘
                    │                           │
           ┌────────▼────────┐      ┌───────────▼───────────┐
           │   GPU (CUDA)    │      │  RunPod / Vast.ai     │
           │   or MPS/CPU    │      │  A100 / RTX4090       │
           └─────────────────┘      └───────────────────────┘
```

## Performance

| Metric | CPU (M1 Mac) | GPU (RTX 4090) | GPU (A100) |
|--------|--------------|----------------|------------|
| RTF (Real-Time Factor) | ~3x | ~0.3x | ~0.15x |
| Latency (1s audio) | ~3s | ~300ms | ~150ms |
| Concurrent streams | 1-2 | 5-10 | 10-20 |

RTF < 1.0 means faster than real-time.

## License

This project uses the CSM model from Sesame AI Labs, which is released under a research license. See the [original repository](https://github.com/SesameAILabs/csm) for license details.
