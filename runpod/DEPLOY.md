# Deploying CSM Voice to RunPod Serverless

## Quick Start (5 minutes)

### Option 1: Use Pre-built Image (Recommended)

1. **Go to RunPod Console**: https://console.runpod.io/serverless
2. **Click "New Endpoint"**
3. **Configure**:
   - **Name**: `csm-voice`
   - **Select GPU**: RTX 4090 ($0.35/hr) or A100 ($1.19/hr)
   - **Container Image**: `docker.io/YOUR_USERNAME/csm-voice:latest`
   - **Container Disk**: 20 GB
   - **Enable FlashBoot**: Yes (for fast cold starts)
4. **Deploy**

### Option 2: Build Your Own Image

```bash
# 1. Set your Docker Hub username
export DOCKER_USERNAME=your-username

# 2. Build the image
./runpod/build.sh

# 3. Push to Docker Hub
docker login
docker push $DOCKER_USERNAME/csm-voice:latest

# 4. Create endpoint in RunPod console using your image
```

## Endpoint Configuration

### Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| GPU | RTX 4090 | Best cost/performance for inference |
| Active Workers | 0 | Scale to zero when idle |
| Max Workers | 5 | Adjust based on expected load |
| Idle Timeout | 5s | How long to keep warm after request |
| FlashBoot | Enabled | Sub-250ms cold starts |
| Container Disk | 20 GB | Model is ~4GB |

### Environment Variables (Optional)

None required - model is baked into the image.

## API Usage

### Endpoint URL
After deployment, your endpoint URL will be:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

### Authentication
Add your RunPod API key to headers:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

### Request Format

```json
{
  "input": {
    "text": "Hello, how are you today?",
    "speaker": 0,
    "max_audio_length_ms": 30000
  }
}
```

### Response Format

```json
{
  "output": {
    "audio_base64": "UklGRv4...",
    "sample_rate": 24000,
    "duration_ms": 1523,
    "generation_time_ms": 187
  }
}
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello from CSM voice synthesis!",
      "speaker": 0
    }
  }'
```

### Python Example

```python
import requests
import base64

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"

def synthesize(text: str, speaker: int = 0) -> bytes:
    """Synthesize speech using RunPod endpoint."""
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={
            "input": {
                "text": text,
                "speaker": speaker,
            }
        },
        timeout=30,
    )

    result = response.json()

    if "error" in result.get("output", {}):
        raise Exception(result["output"]["error"])

    audio_base64 = result["output"]["audio_base64"]
    return base64.b64decode(audio_base64)

# Usage
audio_bytes = synthesize("Hello world!")
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Monitoring

### RunPod Dashboard
- View requests, latency, and errors at https://console.runpod.io/serverless
- Monitor cold start times and GPU utilization

### Logs
- Access container logs in the RunPod console
- Logs include model loading time and generation metrics

## Cost Estimation

| Usage | GPU | Cost/Hour | Est. Monthly |
|-------|-----|-----------|--------------|
| Light (100 calls/day) | RTX 4090 | $0.35 | ~$5-10 |
| Medium (1000 calls/day) | RTX 4090 | $0.35 | ~$30-50 |
| Heavy (10000 calls/day) | A100 | $1.19 | ~$200-400 |

*Assumes ~200ms generation time per request with scale-to-zero*

## Troubleshooting

### Cold Start Too Slow
- Enable FlashBoot in endpoint settings
- Keep 1 minimum worker active (costs more but instant response)

### Out of Memory
- Use A100 (80GB) instead of RTX 4090 (24GB)
- Reduce `max_audio_length_ms` in requests

### Model Loading Failed
- Check container logs for error details
- Ensure HuggingFace can download the model (no auth required for CSM)

## Integration with Human-Like

Update your Human-Like environment:

```env
CSM_RUNPOD_ENDPOINT_ID=your-endpoint-id
CSM_RUNPOD_API_KEY=your-api-key
```

The API service will automatically route synthesis requests to RunPod.
