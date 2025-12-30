# CSM Voice System Documentation

Sesame CSM (Conversational Speech Model) deployment for Human-Like voice agents.

## Table of Contents

1. [Architecture Overview](./architecture.md)
2. [RunPod Deployment](./runpod-deployment.md)
3. [Human-Like Integration](./humanlike-integration.md)
4. [Multilingual Fine-Tuning](./multilingual-finetuning.md)
5. [API Reference](./api-reference.md)

## Quick Start

### Current Deployment

- **Pod ID**: `1s8c803ve11jp3`
- **API URL**: `https://1s8c803ve11jp3-8000.proxy.runpod.net`
- **Cost**: $0.59/hr (RTX 4090)

### Test the API

```bash
# Health check
curl https://1s8c803ve11jp3-8000.proxy.runpod.net/health

# Synthesize speech
curl -X POST https://1s8c803ve11jp3-8000.proxy.runpod.net/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_audio_length_ms": 5000}'
```

### Start/Stop Pod (Save Money)

```bash
# Using RunPod MCP or API
# Stop: mcp__runpod__stop-pod with podId: "1s8c803ve11jp3"
# Start: mcp__runpod__start-pod with podId: "1s8c803ve11jp3"
```

## Key Files

| File | Purpose |
|------|---------|
| `runpod/api_server_full.py` | Main API server with HTTP + WebSocket |
| `runpod/handler.py` | RunPod serverless handler |
| `runpod/Dockerfile` | Docker image for deployment |
| `api/websocket_handler.py` | WebSocket handler class |

## Performance

- **Model**: CSM-1B (1B backbone + 100M decoder)
- **Sample Rate**: 24kHz output, converted to 8kHz mulaw for Twilio
- **Latency**: ~3-4x real-time on RTX 4090 (cold), faster on warm
- **Languages**: English (native), other languages require fine-tuning
