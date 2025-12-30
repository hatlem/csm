"""
Generate speech using fine-tuned CSM model.

Usage:
    python finetune/generate.py --checkpoint ./checkpoints/final_model.pt --text "Hello, this is my cloned voice!"
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator import Generator, Segment, load_llama3_tokenizer
from models import Model
from huggingface_hub import hf_hub_download
from moshi.models import loaders


def load_finetuned_model(checkpoint_path: str, device: str = 'cuda') -> Generator:
    """Load fine-tuned CSM model."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load base model
    model = Model.from_pretrained("sesame/csm-1b")

    # Load fine-tuned weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device=device, dtype=torch.bfloat16)

    # Create generator
    generator = Generator(model)

    return generator


def generate_speech(
    generator: Generator,
    text: str,
    speaker: int = 0,
    context: list = None,
    max_audio_length_ms: float = 30000,
    temperature: float = 0.9,
    topk: int = 50,
) -> torch.Tensor:
    """Generate speech from text."""
    if context is None:
        context = []

    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
    )

    return audio


def main():
    parser = argparse.ArgumentParser(description='Generate speech with fine-tuned CSM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text to speak')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file')
    parser.add_argument('--speaker', type=int, default=0, help='Speaker ID')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--topk', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--max_duration_ms', type=float, default=30000, help='Max audio duration in ms')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--prompt_audio', type=str, help='Optional prompt audio for voice style')
    parser.add_argument('--prompt_text', type=str, help='Transcript of prompt audio')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Load model
    generator = load_finetuned_model(args.checkpoint, args.device)

    # Build context if prompt provided
    context = []
    if args.prompt_audio and args.prompt_text:
        print(f"Loading prompt audio from {args.prompt_audio}...")
        prompt_audio, sr = torchaudio.load(args.prompt_audio)
        prompt_audio = prompt_audio.squeeze(0)

        if sr != generator.sample_rate:
            prompt_audio = torchaudio.functional.resample(
                prompt_audio, orig_freq=sr, new_freq=generator.sample_rate
            )

        context.append(Segment(
            speaker=args.speaker,
            text=args.prompt_text,
            audio=prompt_audio,
        ))

    # Generate
    print(f"Generating speech for: '{args.text}'")
    audio = generate_speech(
        generator=generator,
        text=args.text,
        speaker=args.speaker,
        context=context,
        max_audio_length_ms=args.max_duration_ms,
        temperature=args.temperature,
        topk=args.topk,
    )

    # Save
    torchaudio.save(args.output, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {args.output}")


if __name__ == '__main__':
    main()
