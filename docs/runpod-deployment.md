# RunPod Deployment Guide

## Current Deployment

| Property | Value |
|----------|-------|
| Pod ID | `1s8c803ve11jp3` |
| Datacenter | Norway (EU-NO-1) |
| GPU | RTX 4090 (24GB) |
| Cost | $0.59/hr |
| API URL | `https://1s8c803ve11jp3-8000.proxy.runpod.net` |
| SSH | `ssh root@149.36.1.86 -p 11442` |

## Managing the Pod

### Start Pod
```bash
# Via RunPod MCP
mcp__runpod__start-pod podId="1s8c803ve11jp3"

# Via RunPod API
curl -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query": "mutation { podResume(input: {podId: \"1s8c803ve11jp3\"}) { id } }"}'
```

### Stop Pod (Save Money!)
```bash
# Via RunPod MCP
mcp__runpod__stop-pod podId="1s8c803ve11jp3"
```

### Check Pod Status
```bash
# Via RunPod MCP
mcp__runpod__get-pod podId="1s8c803ve11jp3"
```

## Setting Up a New Pod

### 1. Create Pod via MCP

```javascript
mcp__runpod__create-pod({
  name: "csm-voice-api",
  imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  gpuTypeIds: ["NVIDIA GeForce RTX 4090"],
  gpuCount: 1,
  containerDiskInGb: 50,
  volumeInGb: 100,
  volumeMountPath: "/workspace",
  ports: ["8000/http", "22/tcp"],
  env: {
    PUBLIC_KEY: "ssh-rsa YOUR_PUBLIC_KEY...",
    HF_HOME: "/workspace/cache",
    PYTHONUNBUFFERED: "1"
  }
})
```

### 2. SSH Into Pod

```bash
# Get SSH port from pod info (portMappings.22)
ssh root@<PUBLIC_IP> -p <SSH_PORT>
```

### 3. Install Dependencies

```bash
# Clone CSM repo
cd /workspace
git clone https://github.com/SesameAILabs/csm.git

# Install Python dependencies
pip install torch==2.4.0 torchaudio==2.4.0 transformers==4.49.0 \
    huggingface-hub==0.36.0 tokenizers==0.21.4 moshi==0.2.2 \
    torchtune==0.4.0 torchao==0.9.0 runpod fastapi uvicorn \
    "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master" \
    omegaconf einops sphn bitsandbytes "numpy<2"

# Login to HuggingFace (CSM is a gated model)
huggingface-cli login --token YOUR_HF_TOKEN

# Download model
cd /workspace/csm
python -c "from huggingface_hub import snapshot_download; snapshot_download('sesame/csm-1b', local_dir='./models/csm-1b')"
```

### 4. Upload and Start API Server

```bash
# From local machine
scp -P <SSH_PORT> runpod/api_server_full.py root@<PUBLIC_IP>:/workspace/csm/

# SSH in and start server
ssh root@<PUBLIC_IP> -p <SSH_PORT>
cd /workspace/csm
nohup python api_server_full.py > /workspace/api.log 2>&1 &
```

### 5. Test the API

```bash
# Health check
curl https://<POD_ID>-8000.proxy.runpod.net/health

# Synthesize
curl -X POST https://<POD_ID>-8000.proxy.runpod.net/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_audio_length_ms": 5000}'
```

## Docker Deployment (Alternative)

### Build and Push Image

The GitHub Actions workflow at `.github/workflows/build-runpod.yml` automatically builds and pushes to Docker Hub when changes are made to the `runpod/` directory.

```yaml
# Triggered on push to main with changes in:
# - runpod/**
# - generator.py
# - models.py
```

### Manual Build

```bash
cd runpod
docker build -t yourusername/csm-voice:latest \
  --build-arg HF_TOKEN=your_token .
docker push yourusername/csm-voice:latest
```

### Deploy with Docker Image

```javascript
mcp__runpod__create-pod({
  name: "csm-voice-api",
  imageName: "yourusername/csm-voice:latest",
  gpuTypeIds: ["NVIDIA GeForce RTX 4090"],
  // ... rest of config
})
```

## Serverless Deployment (For Production Scale)

### Create Endpoint

```javascript
mcp__runpod__create-endpoint({
  name: "csm-voice-serverless",
  templateId: "your_template_id",
  gpuTypeIds: ["NVIDIA GeForce RTX 4090"],
  workersMin: 0,
  workersMax: 5
})
```

### Pros/Cons vs Pod

| Aspect | Pod | Serverless |
|--------|-----|------------|
| Cost when idle | $0.59/hr | $0 |
| Cold start | None | ~30-60s |
| Scaling | Manual | Automatic |
| WebSocket | ✅ | ❌ (HTTP only) |
| Best for | Dev/Testing | Production |

## Troubleshooting

### Pod Won't Start
- **GPU host full**: Create new pod in different datacenter
- **Out of credits**: Add credits at runpod.io

### Model Won't Load
- **Gated repo error**: Run `huggingface-cli login`
- **Out of VRAM**: Use larger GPU (A100)

### Audio is Static
- **Not casting to uint8**: Check line 246 in api_server_full.py
- **Sample rate mismatch**: Verify 24kHz → 8kHz resampling

### WebSocket Connection Fails
- **Wrong URL format**: Use `wss://` not `ws://` for RunPod proxy
- **Session ID required**: Include session_id in path

## Cost Optimization

1. **Stop pod when not in use**: $0.59/hr adds up!
2. **Use serverless for production**: Pay only for compute time
3. **Choose right GPU**: RTX 4090 is best value for CSM
4. **Batch requests**: Reduce cold start overhead
