#!/usr/bin/env python3
"""
Script to process MrDragonFox/EN_Emilia_Yodas_616h dataset and push to HuggingFace Hub.
Tokenizes using VyvoTTS tokenizer.
"""

import torch
from vyvotts.audio_tokenizer import process_dataset


def prepare_and_process_dataset():
    """
    Process dataset using VyvoTTS tokenizer
    """
    # Configuration
    source_dataset = "MrDragonFox/EN_Emilia_Yodas_616h"
    output_dataset = "mrfakename/MDF-EN-Emilia-YODAS"

    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: CUDA not available, processing will be slower!")

    print("="*60)
    print("Processing dataset using VyvoTTS audio tokenizer")
    print("="*60)

    # Choose model type: "lfm2", "qwen3", or "granite"
    model_type = "granite"

    # Batch size for processing multiple audio files at once per GPU
    # Increase this to improve GPU utilization (default: 8)
    # With low VRAM usage (2-4GB), you can safely increase to 16-32
    batch_size = 16

    print(f"\nProcessing with VyvoTTS tokenizer ({model_type.upper()} model)")
    print(f"Input dataset: {source_dataset}")
    print(f"Output dataset: {output_dataset}")
    print(f"Batch size: {batch_size} audio files per GPU")

    # Process using VyvoTTS tokenizer
    # This will:
    # - Tokenize audio using SNAC codec with multi-GPU support via Accelerate
    # - Process multiple audio files in batches for better GPU utilization
    # - Tokenize text using the specified model tokenizer
    # - Create training sequences with proper formatting
    # - Push final tokenized dataset to HuggingFace Hub
    process_dataset(
        original_dataset=source_dataset,
        output_dataset=output_dataset,
        model_type=model_type,
        text_field="text_scribe",
        target_sample_rate=24000,
        batch_size=batch_size
    )

    print("\n" + "="*60)
    print("✓ Dataset processing complete!")
    print(f"  Final tokenized dataset pushed to: {output_dataset}")
    print("="*60)


if __name__ == "__main__":
    prepare_and_process_dataset()
