# CSM Voice Fine-tuning

Fine-tune the CSM (Conversational Speech Model) on your own voice.

## Requirements

- GPU with 16GB+ VRAM (24GB recommended)
- 30 minutes to several hours of your voice recordings
- Transcripts of your recordings

## Quick Start

### 1. Prepare Your Data

Create a directory with your audio files (WAV format, any sample rate):

```
my_voice/
├── recording1.wav
├── recording2.wav
├── recording3.wav
└── transcripts.json
```

Create `transcripts.json`:

```json
[
    {"audio": "recording1.wav", "text": "Hello, this is what I said in recording 1."},
    {"audio": "recording2.wav", "text": "And this is what I said in recording 2."},
    {"audio": "recording3.wav", "text": "Finally, this is recording 3."}
]
```

### 2. Process the Data

```bash
cd /path/to/csm
source .venv/bin/activate

# Create sample transcripts file (optional, to see format)
python finetune/prepare_data.py --audio_dir ./my_voice --create_sample

# Process your data
python finetune/prepare_data.py \
    --audio_dir ./my_voice \
    --transcripts ./my_voice/transcripts.json \
    --output_dir ./data
```

### 3. Fine-tune the Model

```bash
# Using LoRA (recommended - faster, less memory)
python finetune/train.py \
    --data_path ./data/train_data.pt \
    --output_dir ./checkpoints \
    --use_lora \
    --num_epochs 10 \
    --learning_rate 1e-4

# Full fine-tuning (better quality, more memory needed)
python finetune/train.py \
    --data_path ./data/train_data.pt \
    --output_dir ./checkpoints \
    --num_epochs 5 \
    --learning_rate 1e-5
```

### 4. Generate Speech

```bash
python finetune/generate.py \
    --checkpoint ./checkpoints/final_model.pt \
    --text "Hello, this is my cloned voice speaking!" \
    --output my_voice_output.wav
```

## Tips for Best Results

### Recording Quality
- Use a good microphone in a quiet environment
- Record at 24kHz or higher sample rate
- Keep recordings between 3-15 seconds each
- Speak naturally, as you would in conversation

### Data Quantity
- **Minimum**: 30 minutes of audio
- **Good**: 1-2 hours of audio
- **Best**: 3-5 hours of audio

### Training Tips
- Start with LoRA fine-tuning (faster iteration)
- Use lower learning rate (1e-5) for full fine-tuning
- Monitor loss - it should decrease steadily
- Save checkpoints frequently and compare outputs

## Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Use LoRA instead of full fine-tuning
- Use gradient checkpointing (add to train.py if needed)

### Poor Voice Quality
- Add more training data
- Ensure transcripts are accurate
- Try different temperature values during generation
- Use voice prompts during generation

### Model Not Learning
- Check that audio files are valid
- Verify transcripts match audio
- Try higher learning rate
- Ensure data is preprocessed correctly

## Advanced: Using Voice Prompts

For better voice consistency, provide a voice prompt during generation:

```bash
python finetune/generate.py \
    --checkpoint ./checkpoints/final_model.pt \
    --text "This is the new text to speak." \
    --prompt_audio ./my_voice/recording1.wav \
    --prompt_text "Hello, this is what I said in recording 1." \
    --output output_with_prompt.wav
```

This helps the model maintain your voice characteristics.
