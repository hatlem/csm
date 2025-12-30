"""
RunPod Serverless Handler for CSM Voice Synthesis.

This handler receives synthesis requests and returns audio.
"""

import os
import sys
import base64
import time
import io
import torch
import torchaudio

# Add CSM directory to path for imports
sys.path.insert(0, "/app/csm")

import runpod

# Global model instance (loaded once on cold start)
generator = None


def load_model():
    """Load the CSM model on startup."""
    global generator

    if generator is not None:
        return generator

    print("Loading CSM model...")
    start_time = time.time()

    from generator import load_csm_1b

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    generator = load_csm_1b(device=device)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    return generator


def synthesize(text: str, speaker: int = 0, context: list = None,
               max_audio_length_ms: int = 30000) -> dict:
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        speaker: Speaker ID (0-based)
        context: Optional conversation context [(text, speaker), ...]
        max_audio_length_ms: Maximum audio length in milliseconds

    Returns:
        Dict with audio_base64, sample_rate, duration_ms
    """
    gen = load_model()

    # Build context segments if provided
    from generator import Segment
    context_segments = []

    if context:
        for ctx_text, ctx_speaker in context:
            # For context, we'd need pre-generated audio
            # For now, skip context audio (simplification)
            pass

    # Generate audio
    start_time = time.time()

    audio = gen.generate(
        text=text,
        speaker=speaker,
        context=context_segments,
        max_audio_length_ms=max_audio_length_ms,
    )

    generation_time = time.time() - start_time

    # Get sample rate from generator
    sample_rate = gen.sample_rate

    # Calculate duration
    duration_ms = int(len(audio) / sample_rate * 1000)

    # Convert to WAV bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
    audio_bytes = buffer.getvalue()

    # Encode to base64
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
        "duration_ms": duration_ms,
        "generation_time_ms": int(generation_time * 1000),
    }


def handler(event):
    """
    RunPod serverless handler.

    Expected input format:
    {
        "input": {
            "text": "Hello, how are you?",
            "speaker": 0,  # optional, default 0
            "context": [["Hi there!", 1], ["Hello!", 0]],  # optional
            "max_audio_length_ms": 30000  # optional
        }
    }

    Returns:
    {
        "audio_base64": "...",
        "sample_rate": 24000,
        "duration_ms": 1500,
        "generation_time_ms": 200
    }
    """
    try:
        input_data = event.get("input", {})

        # Validate required fields
        text = input_data.get("text")
        if not text:
            return {"error": "Missing required field: text"}

        # Optional fields
        speaker = input_data.get("speaker", 0)
        context = input_data.get("context")
        max_audio_length_ms = input_data.get("max_audio_length_ms", 30000)

        # Synthesize
        result = synthesize(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Pre-load model on startup for faster cold starts
print("Pre-loading model on startup...")
try:
    load_model()
    print("Model pre-loaded successfully!")
except Exception as e:
    print(f"Warning: Could not pre-load model: {e}")


# Start the serverless worker
runpod.serverless.start({"handler": handler})
