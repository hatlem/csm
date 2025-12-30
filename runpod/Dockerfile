# RunPod Serverless Worker for CSM Voice Synthesis
# Use base PyTorch image with CUDA 12.1 (matches torch 2.4.0)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (matching tested versions)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    transformers==4.49.0 \
    huggingface-hub==0.36.0 \
    tokenizers==0.21.4 \
    moshi==0.2.2 \
    torchtune==0.4.0 \
    torchao==0.9.0 \
    "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master" \
    runpod \
    "numpy<2"

# Clone CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git /app/csm

# Copy the handler
COPY handler.py /app/handler.py

# Download model weights at build time (for faster cold starts)
# Need HF token for gated model - pass as build arg
ARG HF_TOKEN
RUN python -c "from huggingface_hub import login, snapshot_download; login('${HF_TOKEN}'); snapshot_download('sesame/csm-1b', local_dir='/app/csm/models/csm-1b')"

# Download silentcipher model too
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('SesameAILabs/csm_1b_watermarker', local_dir='/root/.cache/silentcipher')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV MODEL_PATH=/app/csm/models/csm-1b

# Run the handler
CMD ["python", "-u", "/app/handler.py"]
