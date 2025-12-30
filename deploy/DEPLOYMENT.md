# CSM Voice Service - Deployment Guide

## Architecture Overview

The CSM Voice Service requires GPU for efficient inference. Here's the recommended deployment architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Human-Like Platform                           │
│                      (Railway - Existing)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │ PostgreSQL  │  │   Redis     │  │   Human-Like API            │ │
│  │ (Railway)   │  │ (Railway)   │  │   (Railway)                 │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────────┘ │
└─────────┼────────────────┼─────────────────────┼───────────────────┘
          │                │                     │
          │                │                     │ API Calls
          │                │                     ▼
          │                │         ┌───────────────────────────────┐
          │                │         │   CSM Voice Service           │
          │                │         │   (RunPod - GPU)              │
          │                │         │                               │
          │                └─────────│   - Voice Synthesis API       │
          │                          │   - WebSocket Streaming       │
          └──────────────────────────│   - Voice Profile Cache       │
                                     │                               │
                                     │   A100/RTX4090/H100           │
                                     └───────────────────────────────┘
```

## Deployment Options

### Option 1: RunPod (Recommended for Production)

RunPod provides on-demand GPU instances with good pricing.

**Setup:**

1. Create a RunPod account at https://runpod.io

2. Set up the template:
   ```bash
   # Build and push Docker image
   docker build -t your-registry/csm-voice:cuda -f Dockerfile --build-arg USE_CUDA=true .
   docker push your-registry/csm-voice:cuda
   ```

3. Create a new Pod with:
   - GPU: A100 80GB (recommended) or RTX 4090
   - Image: `your-registry/csm-voice:cuda`
   - Volume: 100GB for models
   - Exposed Port: 8000

4. Configure environment variables in RunPod dashboard:
   ```
   CSM_API_KEY=your-secret-key
   CSM_DATABASE_URL=postgresql://...@railway.app:5432/humanlike
   CSM_REDIS_URL=redis://...@railway.app:6379
   HF_TOKEN=your-huggingface-token
   ```

**Costs:** ~$1.89/hr for A100 80GB, ~$0.44/hr for RTX 4090

### Option 2: Modal (Serverless GPU)

Modal provides serverless GPU with pay-per-use pricing.

1. Install Modal: `pip install modal`

2. Create `modal_app.py`:
   ```python
   import modal

   app = modal.App("csm-voice")

   @app.cls(
       gpu=modal.gpu.A100(memory=80),
       image=modal.Image.from_dockerfile("Dockerfile"),
       secrets=[modal.Secret.from_name("csm-secrets")],
   )
   class CSMVoice:
       @modal.enter()
       def load_model(self):
           from api.voice_engine import get_engine
           self.engine = get_engine()

       @modal.method()
       async def synthesize(self, text: str, voice_profile_id: str):
           return await self.engine.synthesize(text, voice_profile_id)
   ```

3. Deploy: `modal deploy modal_app.py`

**Costs:** Pay only when synthesizing, ~$0.001 per second of GPU time

### Option 3: Railway (CPU Only - Development)

Railway doesn't have GPU support, but can run in CPU mode for development/testing.

1. Connect your GitHub repo to Railway

2. Set environment variables:
   ```
   CSM_MODEL_DEVICE=cpu
   CSM_MODEL_DTYPE=float32
   CSM_DATABASE_URL=${{Postgres.DATABASE_URL}}
   CSM_REDIS_URL=${{Redis.REDIS_URL}}
   ```

**Note:** CPU inference is ~10-20x slower than GPU. Not recommended for production.

## Connecting to Human-Like Infrastructure

### Using Shared Database

The CSM service can share Human-Like's PostgreSQL database:

1. In Railway, copy your PostgreSQL connection URL

2. Set in CSM service:
   ```
   CSM_DATABASE_URL=postgresql://user:pass@host.railway.app:5432/humanlike
   CSM_INTEGRATION_MODE=humanlike
   ```

3. Run migrations to add voice tables:
   ```bash
   python -c "from api.database import init_database; import asyncio; asyncio.run(init_database())"
   ```

### Using Shared Redis

1. Copy your Railway Redis URL

2. Set in CSM service:
   ```
   CSM_REDIS_URL=redis://default:password@host.railway.app:6379
   ```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `CSM_API_KEY` | API authentication key | Yes |
| `CSM_DATABASE_URL` | PostgreSQL connection URL | Yes |
| `CSM_REDIS_URL` | Redis connection URL | Yes |
| `CSM_MODEL_DEVICE` | `cuda` or `cpu` | Yes |
| `CSM_MODEL_DTYPE` | `bfloat16` (GPU) or `float32` (CPU) | Yes |
| `HF_TOKEN` | HuggingFace token for model download | Yes (first run) |
| `CSM_INTEGRATION_MODE` | `standalone` or `humanlike` | No |

## Health Check

Verify deployment:
```bash
curl https://your-service-url/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true
}
```

## Scaling

For high-traffic production:

1. **Horizontal Scaling**: Run multiple GPU instances behind a load balancer
2. **Caching**: Redis caches generated audio for repeated requests
3. **Voice Profile Caching**: Hot profiles are kept in GPU memory
4. **Async Processing**: Long synthesis jobs are queued in Redis

## Security

1. Always set `CSM_API_KEY` in production
2. Use HTTPS/WSS for all connections
3. Store secrets in environment variables, not code
4. Rotate API keys regularly
