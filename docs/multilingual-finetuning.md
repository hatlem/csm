# Multilingual Fine-Tuning Guide

## Overview

CSM-1B is primarily trained on English. To support European languages (Norwegian, Swedish, Danish, German, French, Spanish, etc.), we need to fine-tune the model.

## Critical Risks & Warnings

### 1. The "Accent Bleed" Problem
CSM-1B has only 1.1B parameters with a Llama backbone that is English-dominant. Trying to stuff multiple phonology rulesets into one set of weights causes "accent bleed" where Norwegian sounds vaguely Swedish.

### 2. The Llama G2P Bottleneck
The Llama backbone converts Text Tokens → Audio Tokens (Mimi). It likely does NOT know implicit Grapheme-to-Phoneme (G2P) rules for Scandinavian languages:
- "kjøtt" in Norwegian = /çœt/ (like "shyot")
- "kött" in Swedish = /ɕœt/ (different!)
- 10-20h is **NOT enough** to teach these rules from scratch

### 3. Mimi Codec Limitation
Before training, TEST if Mimi can even reconstruct Scandinavian phonetic features:
```bash
# Encode Danish audio with stød, decode it back
# If stød is smoothed out, no amount of training will fix it
```

## The Scandinavian Challenge

Norwegian, Swedish, and Danish look similar in text but sound significantly different:

| Feature | Norwegian | Swedish | Danish |
|---------|-----------|---------|--------|
| **Pitch accent** | ✅ Tones | ✅ Tones | ❌ No tones |
| **Stød** (glottal stop) | ❌ | ❌ | ✅ Critical! |
| **Uvular R** | ✅ Some dialects | ❌ | ✅ |
| **Vowels** | 9 | 9 | 12+ |

## Recommended Architecture

### LoRA Adapters (THE ONLY VIABLE PATH)

**Why NOT single conditioned model:**
- 1B model lacks capacity for strict separation
- Will cause accent bleed between similar languages
- Hard to debug and fix issues

**Why LoRA wins:**
- Freeze the "reasoning" logic (base model)
- Swap out "pronunciation" logic per language
- Modular, debuggable, independent training
- Train Norwegian adapter without breaking Danish

**Separate adapter files:**
```
models/
├── csm-1b/              # Base model (frozen)
├── lora_no.safetensors  # Norwegian adapter
├── lora_sv.safetensors  # Swedish adapter
├── lora_da.safetensors  # Danish adapter
├── lora_de.safetensors  # German adapter
└── lora_fr.safetensors  # French adapter
```

### Input Format (with adapter selection)

```
<lang:no> Hei, hvordan har du det?
<lang:sv> Hej, hur mår du?
<lang:da> Hej, hvordan har du det?
<lang:de> Hallo, wie geht es dir?
<lang:fr> Bonjour, comment allez-vous?
<lang:es> Hola, ¿cómo estás?
```

**Pros:**
- Single model to deploy
- Shared representations across languages
- Efficient storage

**Cons:**
- Requires careful training to prevent bleed
- Need balanced data per language

## Available Datasets

### Norwegian (Excellent Coverage)

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| [NbAiLab/NST](https://huggingface.co/datasets/NbAiLab/NST) | **411h** train + 115h test | Apache 2.0 | Studio recordings |
| [NbAiLab/NPSC](https://huggingface.co/datasets/NbAiLab/NPSC) | ~59h | CC-0 | Parliament |
| NB Tale | ~12h | CC-0 | 380 speakers |

**Best choice:** NST - high quality, diverse speakers, studio recordings.

### Swedish

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| [Nord-Parl-TTS](https://arxiv.org/html/2509.17988) | **5,090h** | Open | Parliament |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | ~500h | CC-0 | EU Parliament |
| [NST Swedish](https://huggingface.co/datasets/jimregan/nst_swedish_tts) | ~100h | Apache 2.0 | Studio |

**Best choice:** Nord-Parl-TTS - massive, but may need filtering.

### Danish

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | ~13,600h (unlabeled) | CC-0 | EU Parliament |
| Common Voice | ~11h (validated) | CC-0 | Crowdsourced |

**Challenge:** Limited high-quality labeled Danish data.

### Other European Languages

| Language | Best Dataset | Hours |
|----------|--------------|-------|
| German | VoxPopuli + Common Voice | ~2,000h+ |
| French | VoxPopuli + Common Voice | ~1,500h+ |
| Spanish | Common Voice | ~1,700h+ |
| Dutch | VoxPopuli | ~500h |
| Italian | VoxPopuli | ~500h |
| Polish | Common Voice | ~200h |

## Training Data Format

### Prepare Dataset

```python
# Each sample needs:
{
    "text": "<lang:no> Hei, jeg heter Andreas.",
    "audio_path": "/data/norwegian/sample_001.wav",
    "speaker_id": "no_speaker_001",
    "language": "no"
}
```

### Data Processing Pipeline

```python
import torchaudio
from datasets import load_dataset

def prepare_sample(sample, language_code):
    """Prepare a sample for CSM fine-tuning."""
    # Load audio
    waveform, sample_rate = torchaudio.load(sample["audio_path"])

    # Resample to 24kHz if needed
    if sample_rate != 24000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)

    # Add language token
    text = f"<lang:{language_code}> {sample['text']}"

    return {
        "text": text,
        "audio": waveform,
        "speaker_id": sample.get("speaker_id", "default")
    }

# Load Norwegian NST dataset
nst = load_dataset("NbAiLab/NST", split="train")
norwegian_data = [prepare_sample(s, "no") for s in nst]
```

## Fine-Tuning Approaches

### 1. LoRA Fine-Tuning (Recommended)

```python
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel

# Load CSM base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="sesame/csm-1b",
    max_seq_length=2048,
    load_in_4bit=True,  # Save memory
)

# Configure LoRA
lora_config = LoraConfig(
    r=64,  # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(
    model=model,
    train_dataset=multilingual_dataset,
    args=TrainingArguments(
        output_dir="./csm-multilingual-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=100,
    ),
)

trainer.train()
```

### 2. Full Fine-Tuning (For Maximum Quality)

From [Speechmatics guide](https://blog.speechmatics.com/sesame-finetune):

> "Since researchers are interested in fine-tuning into new languages, some have elected to fine-tune by modifying the original weights rather than using techniques like LoRA."

Full fine-tuning is better for large domain shifts like new languages, but requires more compute.

## Training Strategy (Expert-Recommended)

### Phase 0: Mimi Codec Sanity Check (CRITICAL!)

Before spending GPU hours, verify Mimi can reconstruct target phonetics:

```python
import torchaudio
from moshi.models import loaders

# Load Mimi codec
mimi = loaders.get_mimi("cuda")

# Test with Danish audio containing stød
danish_audio, sr = torchaudio.load("danish_with_stod.wav")
danish_audio = torchaudio.functional.resample(danish_audio, sr, 24000)

# Encode → Decode (no LLM, just codec)
with torch.no_grad():
    codes = mimi.encode(danish_audio.unsqueeze(0).cuda())
    reconstructed = mimi.decode(codes)

# Save and listen - is the stød preserved?
torchaudio.save("danish_reconstructed.wav", reconstructed.squeeze().cpu(), 24000)
```

**If stød/pitch accent is smoothed out, stop here. Consider alternative TTS.**

### Phase 1: Anchor Training (English-Scandi Bridge)

Don't train on pure Norwegian yet. Train adapter on **English spoken with Norwegian accent**.

**Why this works:**
- Bridges the gap between English (which Llama knows) and target language
- Teaches timbre and prosody while keeping text comprehensible
- Lower risk of catastrophic forgetting

```python
# Use accent datasets or TTS-generated accented English
anchor_data = [
    {"text": "Hello, how are you?", "audio": "norwegian_accent_english.wav"},
    {"text": "The weather is nice today.", "audio": "norwegian_accent_english2.wav"},
]
```

### Phase 2: Synthetic Pre-Training

Your real data (10-50h) is not enough. Bridge the gap:

```python
# Generate 100h of "perfect" Norwegian audio using Azure/OpenAI TTS
import azure.cognitiveservices.speech as speechsdk

synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=None
)

# Use Norwegian voice
ssml = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="nb-NO">
    <voice name="nb-NO-FinnNeural">
        {text}
    </voice>
</speak>
"""

# Generate training data
for text in norwegian_texts:
    result = synthesizer.speak_ssml_async(ssml.format(text=text)).get()
    # Save audio...
```

**Training order:**
1. **100h synthetic** → Learn basic rules (may sound robotic)
2. **10-50h real NST** → Fix prosody, add naturalness

### Phase 3: Language-Specific LoRA Training

Train each language adapter independently:

```python
# Norwegian adapter
trainer_no = Trainer(
    model=peft_model,
    train_dataset=norwegian_dataset,  # 100h synthetic + 50h real
    args=TrainingArguments(
        output_dir="./lora_no",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        warmup_steps=500,
    ),
)
trainer_no.train()
trainer_no.model.save_pretrained("./lora_no")

# Repeat for Swedish, Danish, etc.
```

### Phase 4: Confusion Testing (CRITICAL for Scandinavian)

```python
confusion_tests = [
    # Norwegian-specific phonemes
    ("no", "Kjøtt og fisk er godt", "Should have Norwegian 'kj' sound"),
    ("no", "Jeg bor i Oslo", "Should have Norwegian rhythm"),

    # Swedish-specific phonemes
    ("sv", "Kött och fisk är gott", "Should have Swedish 'tj' sound"),
    ("sv", "Jag bor i Stockholm", "Should have Swedish pitch accent"),

    # Danish-specific phonemes
    ("da", "Kød og fisk er godt", "MUST have stød glottal stop"),
    ("da", "Jeg bor i København", "Should have Danish soft 'd'"),
]

# Automated language ID test
from transformers import pipeline
lang_classifier = pipeline("audio-classification", model="facebook/mms-lid-126")

for lang, text, note in confusion_tests:
    # Load appropriate adapter
    model.load_adapter(f"./lora_{lang}")
    audio = model.generate(text)

    # Verify language
    result = lang_classifier(audio)
    predicted = result[0]["label"]
    confidence = result[0]["score"]

    print(f"[{lang}] '{text[:30]}...' → Predicted: {predicted} ({confidence:.2%})")
    print(f"    Note: {note}")

    # Should be >90% correct language
    assert predicted == lang or confidence > 0.9
```

### Phase 5 (Optional): Phoneme Pre-Processing

If G2P fails (model can't learn spelling→sound mapping), use phonemes:

```python
from phonemizer import phonemize

def to_phonemes(text, language):
    """Convert text to IPA phonemes."""
    lang_map = {"no": "nb", "sv": "sv", "da": "da", "de": "de"}
    return phonemize(
        text,
        language=lang_map[language],
        backend="espeak",
        strip=True
    )

# Instead of "Hei, hvordan har du det?"
# Feed: "hɛj vʊɾɑn hɑɾ dʉ de"
```

This bypasses the "reading" problem entirely.

## GPU Requirements & Cost Estimates (Revised)

### LoRA Settings for Audio

Audio requires more capacity than text. Use high-rank LoRA:

| Setting | Text Tasks | Audio Tasks (CSM) |
|---------|------------|-------------------|
| **Rank (r)** | 8-16 | **64-128** |
| **Alpha** | 16-32 | **2x rank (128-256)** |
| **Target modules** | q,v | **q,k,v,o** (all attention) |
| **Quantization** | Optional | **4-bit recommended** (save VRAM for high rank) |

### LoRA Fine-Tuning (Per Language)

| Setup | VRAM | Time (50h data) | Cost |
|-------|------|-----------------|------|
| 1x RTX 4090 (4-bit) | 24GB | ~48-72h | ~$30-45 |
| 1x A100 40GB | 40GB | ~24-36h | ~$50-75 |
| 1x A100 80GB | 80GB | ~16-24h | ~$50-75 |

### Full Training Pipeline (All Languages)

| Phase | Data | Time | Cost |
|-------|------|------|------|
| Phase 0: Mimi test | N/A | 1h | ~$1 |
| Phase 1: Anchor (EN-NO accent) | 10h | 8h | ~$10 |
| Phase 2: Synthetic pre-train | 100h | 24h | ~$30 |
| Phase 3: Real data fine-tune (NO) | 50h | 24h | ~$30 |
| Phase 3: Swedish adapter | 50h | 24h | ~$30 |
| Phase 3: Danish adapter | 50h | 24h | ~$30 |
| Phase 4: Confusion testing | N/A | 4h | ~$5 |
| **Total (3 Scandinavian)** | | ~130h | **~$165** |

### Recommended Approach

**Use 4-bit quantized base + high-rank LoRA on 1x A100 80GB:**

1. **Phase 0**: Test Mimi codec with Scandinavian audio (1 hour)
2. **Phase 1**: Train anchor adapter (English with Norwegian accent)
3. **Phase 2**: Generate 100h synthetic Norwegian, pre-train
4. **Phase 3**: Fine-tune on 50h real NST data
5. **Phase 4**: Confusion test Norwegian vs Swedish vs Danish
6. **Repeat** Phase 2-4 for Swedish and Danish

## Deployment

### Loading LoRA Adapter

```python
from peft import PeftModel

# Load base model
base_model = load_csm_1b(device="cuda")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "path/to/csm-multilingual-lora",
)

# Generate with language token
audio = model.generate("<lang:no> Hei, hvordan har du det?")
```

### API Changes

Update `api_server_full.py` to accept language parameter:

```python
class SynthesizeRequest(BaseModel):
    text: str
    speaker: int = 0
    language: str = "en"  # New: language code
    max_audio_length_ms: int = 30000

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    # Prepend language token
    text = f"<lang:{request.language}> {request.text}"
    audio = generator.generate(text=text, ...)
```

## Quality Evaluation

### Metrics

1. **MOS (Mean Opinion Score)**: Human rating 1-5
2. **Language Accuracy**: Can native speakers identify the language?
3. **Intelligibility**: Word Error Rate on resynthesized speech
4. **Naturalness**: Prosody, rhythm, intonation

### Automated Testing

```python
from transformers import pipeline

# Language identification
lang_classifier = pipeline("audio-classification", model="facebook/mms-lid-126")

def test_language_accuracy(audio, expected_lang):
    result = lang_classifier(audio)
    predicted_lang = result[0]["label"]
    return predicted_lang == expected_lang
```

## Timeline

| Week | Task |
|------|------|
| 1 | Prepare Norwegian dataset, set up training infrastructure |
| 2 | Train Norwegian LoRA, evaluate quality |
| 3 | Add Swedish and Danish, train jointly |
| 4 | Confusion testing, fix language bleed issues |
| 5 | Add German, French, Spanish |
| 6 | Final evaluation, deployment |

## Resources

- [Speechmatics: How to Finetune Sesame CSM](https://blog.speechmatics.com/sesame-finetune)
- [Unsloth TTS Fine-tuning Docs](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)
- [CSM-1B on HuggingFace](https://huggingface.co/sesame/csm-1b)
- [CSM Fine-tuning Discussion](https://huggingface.co/sesame/csm-1b/discussions/9)
- [Example LoRA: keanteng/sesame-csm-elise-lora](https://huggingface.co/keanteng/sesame-csm-elise-lora)
