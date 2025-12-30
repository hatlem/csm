#!/bin/bash
# Build and push Docker image for RunPod

set -e

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"
IMAGE_NAME="csm-voice"
TAG="${TAG:-latest}"

FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"

# Build from the project root
cd "$(dirname "$0")/.."

# Build the image
docker build -f runpod/Dockerfile -t "${FULL_IMAGE}" .

echo ""
echo "Build complete!"
echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push ${FULL_IMAGE}"
echo ""
echo "Then go to RunPod console and create a serverless endpoint with:"
echo "  Image: ${FULL_IMAGE}"
echo "  GPU: RTX 4090 or A100"
