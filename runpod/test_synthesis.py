#!/usr/bin/env python3
"""Test CSM voice synthesis on RunPod GPU."""
import torch
import torchaudio
import time

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

from generator import load_csm_1b

print("Loading model...")
start = time.time()
gen = load_csm_1b(device="cuda")
print(f"Model loaded in {time.time() - start:.2f}s")

# Test synthesis
text = "Hello! This is a test of the CSM voice synthesis model running on RunPod GPU."
print(f"\nSynthesizing: '{text}'")

start = time.time()
audio = gen.generate(
    text=text,
    speaker=0,
    context=[],
    max_audio_length_ms=30000,
)
synthesis_time = time.time() - start

duration = len(audio) / gen.sample_rate
print(f"Generated {duration:.2f}s of audio in {synthesis_time:.2f}s")
print(f"Real-time factor: {duration / synthesis_time:.2f}x")

# Save output
output_path = "/workspace/test_output.wav"
torchaudio.save(output_path, audio.unsqueeze(0).cpu(), gen.sample_rate)
print(f"Saved to: {output_path}")
