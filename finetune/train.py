"""
Fine-tuning script for CSM (Conversational Speech Model).

This script fine-tunes the CSM model on your voice data to create a personalized
voice clone. It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Requirements:
    - Prepared dataset from prepare_data.py
    - GPU with at least 16GB VRAM (24GB+ recommended)
    - Access to sesame/csm-1b model on HuggingFace

Usage:
    python finetune/train.py --data_path ./data/train_data.pt --output_dir ./checkpoints
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator import load_llama3_tokenizer, Segment
from models import Model
from huggingface_hub import hf_hub_download
from moshi.models import loaders


class VoiceDataset(Dataset):
    """Dataset for CSM fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        audio_tokenizer,
        max_seq_len: int = 2048,
        device: str = 'cpu',
    ):
        self.data = torch.load(data_path)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def _tokenize_text(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment."""
        text_tokens = self.tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio segment."""
        with torch.no_grad():
            audio_tokens = self.audio_tokenizer.encode(
                audio.unsqueeze(0).unsqueeze(0).to(self.device)
            )[0]

        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1, device=audio_tokens.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1).cpu()
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        text_tokens, text_mask = self._tokenize_text(item['text'], item['speaker'])

        # Tokenize audio
        audio_tokens, audio_mask = self._tokenize_audio(item['audio'])

        # Concatenate
        tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        mask = torch.cat([text_mask, audio_mask], dim=0)

        # Truncate if needed
        if tokens.size(0) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            mask = mask[:self.max_seq_len]

        return {
            'tokens': tokens,
            'mask': mask,
            'text_len': text_tokens.size(0),
            'audio_len': audio_tokens.size(0),
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(item['tokens'].size(0) for item in batch)

    tokens_batch = []
    mask_batch = []
    text_lens = []
    audio_lens = []

    for item in batch:
        seq_len = item['tokens'].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            tokens_padded = F.pad(item['tokens'], (0, 0, 0, pad_len), value=0)
            mask_padded = F.pad(item['mask'], (0, 0, 0, pad_len), value=False)
        else:
            tokens_padded = item['tokens']
            mask_padded = item['mask']

        tokens_batch.append(tokens_padded)
        mask_batch.append(mask_padded)
        text_lens.append(item['text_len'])
        audio_lens.append(item['audio_len'])

    return {
        'tokens': torch.stack(tokens_batch),
        'mask': torch.stack(mask_batch),
        'text_lens': torch.tensor(text_lens),
        'audio_lens': torch.tensor(audio_lens),
    }


class LoRALayer(nn.Module):
    """LoRA adapter layer for efficient fine-tuning."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def add_lora_to_model(model: Model, rank: int = 8, alpha: float = 16):
    """Add LoRA adapters to the model."""
    lora_layers = {}

    # Add LoRA to attention layers in backbone
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear) and ('q_proj' in name or 'v_proj' in name):
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
            lora_layers[f'backbone.{name}'] = lora

    # Add LoRA to decoder attention
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Linear) and ('q_proj' in name or 'v_proj' in name):
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
            lora_layers[f'decoder.{name}'] = lora

    return lora_layers


def compute_loss(
    model: Model,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    text_lens: torch.Tensor,
    lora_layers: dict = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for next-token prediction.

    Only compute loss on audio tokens (not text tokens).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch_size, seq_len, _ = tokens.size()

    # Get embeddings
    embeds = model._embed_tokens(tokens)
    masked_embeds = embeds * mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)

    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Forward through backbone (without KV cache for training)
    input_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Disable caches for training
    model.backbone.reset_caches()
    h = model.backbone(h, input_pos=input_pos, mask=causal_mask).to(dtype=dtype)

    # Get logits for codebook 0
    logits = model.codebook0_head(h)  # (batch, seq_len, vocab_size)

    # Create targets (shifted by 1)
    targets = tokens[:, 1:, 0].long()  # First codebook

    # Create loss mask (only for audio tokens)
    loss_mask = torch.zeros(batch_size, seq_len - 1, device=device)
    for i in range(batch_size):
        text_len = text_lens[i].item()
        # Start computing loss after text tokens
        loss_mask[i, text_len:] = 1.0

    # Compute cross-entropy loss
    logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    loss_mask_flat = loss_mask.reshape(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = (loss * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)

    return loss


def train(
    model: Model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    output_dir: str,
    device: str,
    lora_layers: dict = None,
    save_every: int = 100,
    log_every: int = 10,
):
    """Training loop."""
    model.train()
    global_step = 0

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            tokens = batch['tokens'].to(device)
            mask = batch['mask'].to(device)
            text_lens = batch['text_lens']

            optimizer.zero_grad()
            loss = compute_loss(model, tokens, mask, text_lens, lora_layers)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step % save_every == 0:
                save_path = os.path.join(output_dir, f'checkpoint_{global_step}.pt')
                save_checkpoint(model, lora_layers, optimizer, global_step, save_path)

        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')

        # Save epoch checkpoint
        save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        save_checkpoint(model, lora_layers, optimizer, global_step, save_path)

    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pt')
    save_checkpoint(model, lora_layers, optimizer, global_step, final_path)
    print(f'Training complete! Final model saved to {final_path}')


def save_checkpoint(model, lora_layers, optimizer, step, path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if lora_layers:
        checkpoint['lora_state_dict'] = {k: v.state_dict() for k, v in lora_layers.items()}

    torch.save(checkpoint, path)
    print(f'Saved checkpoint to {path}')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CSM on your voice')
    parser.add_argument('--data_path', type=str, required=True, help='Path to prepared training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to MPS or CPU")
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Load model
    print("Loading CSM model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=args.device, dtype=torch.bfloat16)

    # Load tokenizers
    print("Loading tokenizers...")
    text_tokenizer = load_llama3_tokenizer()

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=args.device)
    audio_tokenizer.set_num_codebooks(32)

    # Create dataset
    print("Loading dataset...")
    dataset = VoiceDataset(
        data_path=args.data_path,
        tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        device=args.device,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Setup LoRA if requested
    lora_layers = None
    if args.use_lora:
        print("Adding LoRA adapters...")
        lora_layers = add_lora_to_model(model, args.lora_rank, args.lora_alpha)
        for lora in lora_layers.values():
            lora.to(args.device)

        # Only train LoRA parameters
        for param in model.parameters():
            param.requires_grad = False

        trainable_params = []
        for lora in lora_layers.values():
            trainable_params.extend(lora.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    else:
        # Full fine-tuning
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        device=args.device,
        lora_layers=lora_layers,
    )


if __name__ == '__main__':
    main()
