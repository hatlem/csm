"""
Data preparation script for fine-tuning CSM on your voice.

Usage:
    1. Place your audio files in a directory (WAV format, 24kHz preferred)
    2. Create a transcripts.json file with format:
       [
           {"audio": "file1.wav", "text": "What I said in file 1"},
           {"audio": "file2.wav", "text": "What I said in file 2"},
           ...
       ]
    3. Run: python finetune/prepare_data.py --audio_dir /path/to/audio --output_dir ./data
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
import torchaudio
from tqdm import tqdm


def load_and_resample_audio(audio_path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)

    return audio.squeeze(0)


def segment_audio(
    audio: torch.Tensor,
    max_duration_sec: float = 10.0,
    sample_rate: int = 24000
) -> List[torch.Tensor]:
    """Split long audio into smaller segments."""
    max_samples = int(max_duration_sec * sample_rate)

    if audio.shape[0] <= max_samples:
        return [audio]

    segments = []
    for i in range(0, audio.shape[0], max_samples):
        segment = audio[i:i + max_samples]
        if segment.shape[0] > sample_rate:  # At least 1 second
            segments.append(segment)

    return segments


def prepare_dataset(
    audio_dir: str,
    transcripts_path: str,
    output_dir: str,
    speaker_id: int = 0,
    max_duration_sec: float = 10.0,
    sample_rate: int = 24000,
):
    """
    Prepare dataset for CSM fine-tuning.

    Args:
        audio_dir: Directory containing audio files
        transcripts_path: Path to JSON file with transcripts
        output_dir: Output directory for processed data
        speaker_id: Speaker ID to use (default 0)
        max_duration_sec: Maximum duration per segment
        sample_rate: Target sample rate
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load transcripts
    with open(transcripts_path, 'r') as f:
        transcripts = json.load(f)

    processed_data = []

    for item in tqdm(transcripts, desc="Processing audio"):
        audio_path = os.path.join(audio_dir, item['audio'])

        if not os.path.exists(audio_path):
            print(f"Warning: {audio_path} not found, skipping")
            continue

        try:
            audio = load_and_resample_audio(audio_path, sample_rate)

            # For long audio with sentence-level transcripts, don't segment
            if 'segments' in item:
                # Handle pre-segmented data with timestamps
                for seg in item['segments']:
                    start_sample = int(seg['start'] * sample_rate)
                    end_sample = int(seg['end'] * sample_rate)
                    segment_audio = audio[start_sample:end_sample]

                    if segment_audio.shape[0] > sample_rate * 0.5:  # At least 0.5 sec
                        processed_data.append({
                            'audio': segment_audio,
                            'text': seg['text'],
                            'speaker': speaker_id,
                        })
            else:
                # Single utterance per file
                segments = segment_audio(audio, max_duration_sec, sample_rate)

                if len(segments) == 1:
                    processed_data.append({
                        'audio': segments[0],
                        'text': item['text'],
                        'speaker': speaker_id,
                    })
                else:
                    # For segmented audio, we need corresponding text segments
                    # This is a simple split - ideally use forced alignment
                    print(f"Warning: {item['audio']} was segmented, using full text for first segment only")
                    processed_data.append({
                        'audio': segments[0],
                        'text': item['text'],
                        'speaker': speaker_id,
                    })

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Save processed data
    output_path = os.path.join(output_dir, 'train_data.pt')
    torch.save(processed_data, output_path)

    # Save metadata
    metadata = {
        'num_samples': len(processed_data),
        'speaker_id': speaker_id,
        'sample_rate': sample_rate,
        'total_duration_hours': sum(d['audio'].shape[0] for d in processed_data) / sample_rate / 3600,
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset prepared:")
    print(f"  - Samples: {metadata['num_samples']}")
    print(f"  - Total duration: {metadata['total_duration_hours']:.2f} hours")
    print(f"  - Output: {output_path}")

    return processed_data


def create_sample_transcripts(audio_dir: str, output_path: str):
    """Create a sample transcripts.json file from audio directory."""
    audio_files = list(Path(audio_dir).glob('*.wav')) + list(Path(audio_dir).glob('*.mp3'))

    transcripts = [
        {"audio": f.name, "text": "REPLACE_WITH_ACTUAL_TRANSCRIPT"}
        for f in sorted(audio_files)
    ]

    with open(output_path, 'w') as f:
        json.dump(transcripts, f, indent=2)

    print(f"Created sample transcripts file: {output_path}")
    print(f"Found {len(transcripts)} audio files. Please edit the file to add actual transcripts.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for CSM fine-tuning')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with audio files')
    parser.add_argument('--transcripts', type=str, help='Path to transcripts JSON file')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--speaker_id', type=int, default=0, help='Speaker ID')
    parser.add_argument('--max_duration', type=float, default=10.0, help='Max segment duration in seconds')
    parser.add_argument('--create_sample', action='store_true', help='Create sample transcripts file')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_transcripts(args.audio_dir, os.path.join(args.audio_dir, 'transcripts.json'))
    else:
        if not args.transcripts:
            args.transcripts = os.path.join(args.audio_dir, 'transcripts.json')

        prepare_dataset(
            audio_dir=args.audio_dir,
            transcripts_path=args.transcripts,
            output_dir=args.output_dir,
            speaker_id=args.speaker_id,
            max_duration_sec=args.max_duration,
        )
