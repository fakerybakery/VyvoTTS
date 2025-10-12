<div align="center">
<h2>
    VyvoTTS: LLM-Based Text-to-Speech Training Framework 🚀
</h2>
<div>
    <div align="center">
    <img width="400" alt="VyvoTTS Logo" src="assets/logo.png" style="max-width: 100%; height: auto;">
</div>
</div>
<div>
    <a href="https://github.com/Vyvo-Labs/VyvoTTS" target="_blank">
        <img src="https://img.shields.io/github/stars/Vyvo-Labs/VyvoTTS?style=for-the-badge&color=FF6B6B&labelColor=2D3748" alt="GitHub stars">
    </a>
    <a href="https://github.com/Vyvo-Labs/VyvoTTS/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-MIT-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="MIT License">
    </a>
    <a href="https://python.org" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.8+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.8+">
    </a>
    <a href="https://huggingface.co/spaces/Vyvo/VyvoTTS-LFM2" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD93D?style=for-the-badge&labelColor=2D3748" alt="HuggingFace Spaces">
    </a>
</div>
</div>

This library was developed by the VyvoTTS team. A Text-to-Speech (TTS) training and inference framework built on top of the LLM model.

## ✨ Features

- **Pre-training**: Train LLM models from scratch with custom datasets
- **Fine-tuning**: Adapt pre-trained models for specific TTS tasks
- **LoRA Adaptation**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **Voice Cloning**: Clone voices using advanced neural techniques
- **Multi-GPU Support**: Distributed training with accelerate

## 📦 Installation

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

## 🚀 Quick Start

### Dataset Preparation

VyvoTTS provides a unified tokenizer that works with both Qwen3 and LFM2 models. The tokenizer reads configuration from YAML files for flexibility.

#### Tokenizer Usage

```python
from vyvotts.audio_tokenizer import process_dataset

# For Qwen3
process_dataset(
    original_dataset="MrDragonFox/Elise",
    output_dataset="username/dataset-name",
    model_type="qwen3",
    text_field="text"
)

# For LFM2
process_dataset(
    original_dataset="MrDragonFox/Elise",
    output_dataset="username/dataset-name",
    model_type="lfm2",
    text_field="text"
)
```

**Parameters:**
- `original_dataset`: HuggingFace dataset path to process
- `output_dataset`: Output dataset path on HuggingFace Hub
- `model_type`: Model type - either "qwen3" or "lfm2" (default: "qwen3")
- `text_field`: Name of text field in dataset (e.g., "text_scribe", "text")

### Training

#### Fine-tuning
⚠️ GPU Requirements:** 30GB VRAM minimum required for fine-tuning

Configure your fine-tuning parameters in `vyvotts/configs/lfm2_ft.yaml` and run:

```bash
accelerate launch --config_file vyvotts/configs/accelerate_finetune.yaml vyvotts/train.py
```

💻 For lower-end GPUs (6GB+):** Use the Unsloth FP8/FP4 training notebook:
```bash
uv pip install jupyter notebook
uv jupyter notebook notebook/vyvotts-lfm2-train.ipynb
```

#### Pre-training
Configure your pre-training parameters in `vyvotts/configs/lfm2_config.yaml` and run:

```bash
accelerate launch --config_file vyvotts/configs/accelerate_pretrain.yaml vyvotts/train.py
```

### Inference

VyvoTTS provides multiple inference backends optimized for different use cases:

#### 1. Transformers Inference (Standard)
Standard inference using HuggingFace Transformers with full precision.

```python
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference
import soundfile as sf

# Initialize engine
engine = VyvoTTSTransformersInference(
    model_name="Vyvo/VyvoTTS-LFM2-Neuvillette"
)

# Generate speech
audio, timing_info = engine.generate(
    text="Hello, this is a test of the text to speech system.",
    voice=None,  # Optional: specify voice name
    max_new_tokens=1200,
    temperature=0.6,
    top_p=0.95
)

# Save audio
if audio is not None:
    audio_numpy = audio.detach().squeeze().cpu().numpy()
    sf.write("output.wav", audio_numpy, 24000)
```

#### 2. Unsloth Inference (Memory Efficient)
Optimized inference with 4-bit/8-bit quantization support.

```python
from vyvotts.inference.unsloth_inference import VyvoTTSUnslothInference

# Initialize engine with 4-bit quantization
engine = VyvoTTSUnslothInference(
    model_name="Vyvo/VyvoTTS-v2-Neuvillette",
    load_in_4bit=True  # Use 4-bit quantization for lower memory
)

# Generate and save audio
audio = engine.generate(
    text="Hey there, my name is Elise.",
    voice=None,
    max_new_tokens=1200,
    temperature=0.6
)

if audio is not None:
    engine.save_audio(audio, "output.wav")
```

#### 3. HQQ Quantized Inference (4-bit)
High-quality 4-bit quantization with gemlite backend for faster inference.

```python
from vyvotts.inference.transformers_hqq_inference import VyvoTTSHQQInference

# Initialize engine with HQQ quantization
engine = VyvoTTSHQQInference(
    model_name="Vyvo/VyvoTTS-LFM2-Neuvillette",
    nbits=4,  # 4-bit quantization
    group_size=64
)

# Generate speech
audio, timing_info = engine.generate(
    text="Hello world, this is HQQ inference.",
    voice=None,
    max_new_tokens=1200,
    temperature=0.6
)

print(f"Generation time: {timing_info['generation_time']:.2f}s")
```

#### 4. vLLM Inference (Fastest)
Production-ready inference with vLLM for maximum throughput.

```python
from vyvotts.inference.vllm_inference import VyvoTTSInference

# Initialize engine
engine = VyvoTTSInference(
    model_name="Vyvo/VyvoTTS-LFM2-Neuvillette"
)

# Generate speech
audio = engine.generate(
    text="Hello world, this is vLLM inference.",
    voice="zoe"  # Optional voice identifier
)

if audio is not None:
    import soundfile as sf
    audio_numpy = audio.detach().squeeze().cpu().numpy()
    sf.write("output.wav", audio_numpy, 24000)
```

## 👨‍🍳 Roadmap

- [ ] Transformers.js support
- [X] vLLM support
- [X] Pretrained model release
- [X] Training and inference code release

## 🙏 Acknowledgements

We would like to thank the following projects and teams that made this work possible:

- [Orpheus TTS](https://github.com/canopyai/orpheus-tts) - For foundational TTS research and implementation
- [LiquidAI](https://huggingface.co/LiquidAI) - For the LFM2 model architecture and pre-trained weights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
