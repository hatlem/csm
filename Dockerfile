# =============================================================================
# CSM Voice Service - Production Docker Image
# =============================================================================
#
# Multi-stage build for optimal image size and security.
# Supports both CPU and CUDA GPU inference.
#
# Build:
#   docker build -t csm-voice:latest .
#   docker build --build-arg CUDA_VERSION=12.1 -t csm-voice:cuda .
#
# Run:
#   docker run -p 8000:8000 csm-voice:latest
#   docker run --gpus all -p 8000:8000 csm-voice:cuda
#

# =============================================================================
# Build Arguments
# =============================================================================

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1
ARG USE_CUDA=true

# =============================================================================
# Stage 1: Base with dependencies
# =============================================================================

FROM python:${PYTHON_VERSION}-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 csm

WORKDIR /app

# Copy and install Python dependencies first (for caching)
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# =============================================================================
# Stage 2: CUDA variant
# =============================================================================

FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 as cuda-base

ARG PYTHON_VERSION=3.11

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Create non-root user
RUN useradd -m -u 1000 csm

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# =============================================================================
# Stage 3: Final image
# =============================================================================

FROM base as final-cpu
FROM cuda-base as final-cuda

# Use build arg to select final stage
ARG USE_CUDA=true

# This FROM will be the last one evaluated based on build target
FROM final-${USE_CUDA:+cuda}${USE_CUDA:-cpu} as final

WORKDIR /app

# Copy application code
COPY --chown=csm:csm . .

# Create directories for models and data
RUN mkdir -p /app/models /app/voice_profiles /app/logs && \
    chown -R csm:csm /app

# Switch to non-root user
USER csm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CSM_API_HOST=0.0.0.0 \
    CSM_API_PORT=8000 \
    CSM_MODEL_PATH=/app/models/csm-1b \
    CSM_VOICE_PROFILES_DIR=/app/voice_profiles \
    CSM_ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
