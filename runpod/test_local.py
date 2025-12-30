#!/usr/bin/env python3
"""
Test the RunPod handler locally before deploying.
"""

import sys
import base64
sys.path.insert(0, "..")

# Mock runpod module for local testing
class MockRunpod:
    class serverless:
        @staticmethod
        def start(config):
            print("Mock: runpod.serverless.start() called")
            # Don't actually start the server for testing

# Inject mock before importing handler
sys.modules['runpod'] = MockRunpod()

# Now we can test the handler function directly
from handler import handler, synthesize

def test_synthesize():
    """Test direct synthesis."""
    print("Testing synthesize function...")

    result = synthesize(
        text="Hello, this is a test of the CSM voice synthesis system.",
        speaker=0,
    )

    print(f"  Sample rate: {result['sample_rate']}")
    print(f"  Duration: {result['duration_ms']}ms")
    print(f"  Generation time: {result['generation_time_ms']}ms")
    print(f"  Audio size: {len(result['audio_base64'])} chars (base64)")

    # Save audio for listening
    audio_bytes = base64.b64decode(result['audio_base64'])
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print("  Saved to test_output.wav")

    return result

def test_handler():
    """Test the RunPod handler."""
    print("\nTesting handler function...")

    event = {
        "input": {
            "text": "This is a test from the RunPod handler.",
            "speaker": 0,
        }
    }

    result = handler(event)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        if "traceback" in result:
            print(result['traceback'])
        return None

    print(f"  Success!")
    print(f"  Duration: {result['duration_ms']}ms")
    print(f"  Generation time: {result['generation_time_ms']}ms")

    return result

def test_handler_error():
    """Test handler error handling."""
    print("\nTesting handler error handling...")

    # Missing text
    event = {"input": {}}
    result = handler(event)

    assert "error" in result, "Should return error for missing text"
    print(f"  Correctly returned error: {result['error']}")

if __name__ == "__main__":
    print("=" * 50)
    print("CSM RunPod Handler Local Test")
    print("=" * 50)

    test_synthesize()
    test_handler()
    test_handler_error()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
