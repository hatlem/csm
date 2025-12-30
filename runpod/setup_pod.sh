#!/bin/bash
# Setup script for CSM on RunPod
set -e

echo "=== Setting up CSM Voice Service ==="

cd /workspace

# Clone CSM repo if not exists
if [ ! -d "csm" ]; then
    echo "Cloning CSM repository..."
    git clone https://github.com/SesameAILabs/csm.git
fi

cd csm

# Install dependencies
echo "Installing Python dependencies..."
pip install -q torch torchaudio transformers huggingface_hub tokenizers moshi runpod

# Download model
echo "Downloading CSM-1B model from HuggingFace..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sesame/csm-1b',
    local_dir='./models/csm-1b',
    local_dir_use_symlinks=False
)
print('Model downloaded successfully!')
"

# Test the model
echo "Testing model load..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

from generator import load_csm_1b
gen = load_csm_1b(device='cuda')
print('Model loaded successfully!')
print(f'Sample rate: {gen.sample_rate}')
"

echo "=== Setup complete! ==="
echo "Run: python test_synthesis.py to test voice generation"
