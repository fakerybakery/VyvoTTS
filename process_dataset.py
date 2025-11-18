#!/usr/bin/env python3
"""
Script to process MrDragonFox/EN_Emilia_Yodas_616h dataset and push to HuggingFace Hub.
Renames columns to: audio, text_scribe and tokenizes using VyvoTTS tokenizer.
"""

import os
import torch
from datasets import load_dataset
from vyvotts.audio_tokenizer import process_dataset


def prepare_and_process_dataset():
    """
    Step 1: Rename columns and prepare dataset
    Step 2: Process using VyvoTTS tokenizer
    """
    # Configuration
    source_dataset = "MrDragonFox/EN_Emilia_Yodas_616h"
    output_dataset = "mrfakename/MDF-EN-Emilia-YODAS"
    temp_dataset = "temp_renamed_dataset"  # Temporary local dataset

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
    print("STEP 1: Loading and preparing dataset with renamed columns")
    print("="*60)

    # Load the source dataset
    print(f"\nLoading dataset: {source_dataset}")
    dataset = load_dataset(source_dataset)

    print(f"Dataset loaded. Splits: {list(dataset.keys())}")
    first_split = list(dataset.keys())[0]
    print(f"Sample columns: {dataset[first_split].column_names}")

    # Common column name variations
    audio_columns = ['audio', 'Audio', 'file', 'path', 'audio_file']
    text_columns = ['text', 'Text', 'transcript', 'transcription', 'sentence', 'text_scribe']

    def rename_and_cleanup_columns(split_dataset):
        """Rename columns to standardized names and keep only necessary columns."""
        columns = split_dataset.column_names
        rename_dict = {}

        # Find and rename audio column
        for col in columns:
            if col in audio_columns or 'audio' in col.lower():
                if col != 'audio':
                    rename_dict[col] = 'audio'
                break

        # Find and rename text column
        for col in columns:
            if col in text_columns or 'text' in col.lower() or 'transcript' in col.lower():
                if col != 'text_scribe':
                    rename_dict[col] = 'text_scribe'
                break

        if rename_dict:
            print(f"  Renaming columns: {rename_dict}")
            split_dataset = split_dataset.rename_columns(rename_dict)

        # Keep only audio and text_scribe columns
        columns_to_keep = ['audio', 'text_scribe']
        columns_to_remove = [col for col in split_dataset.column_names if col not in columns_to_keep]

        if columns_to_remove:
            print(f"  Removing columns: {columns_to_remove}")
            split_dataset = split_dataset.remove_columns(columns_to_remove)

        return split_dataset

    # Process all splits
    from datasets import DatasetDict
    processed_dataset = {}

    for split_name, split_data in dataset.items():
        print(f"\nProcessing split: {split_name}")
        processed_dataset[split_name] = rename_and_cleanup_columns(split_data)
        print(f"  Final columns: {processed_dataset[split_name].column_names}")
        print(f"  Number of examples: {len(processed_dataset[split_name])}")

    # Convert back to DatasetDict if multiple splits, otherwise keep single split
    if len(processed_dataset) > 1:
        processed_dataset = DatasetDict(processed_dataset)
    else:
        processed_dataset = list(processed_dataset.values())[0]

    # Show sample
    print("\n" + "="*60)
    print("Sample from prepared dataset:")
    sample = processed_dataset[list(processed_dataset.keys())[0]][0] if isinstance(processed_dataset, DatasetDict) else processed_dataset[0]
    for key, value in sample.items():
        if key == 'audio':
            print(f"  {key}: {type(value)} (audio data)")
        else:
            print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
    print("="*60)

    # Save dataset locally for tokenization step
    print(f"\nSaving prepared dataset locally to: {temp_dataset}")
    processed_dataset.save_to_disk(temp_dataset)
    print("✓ Dataset saved locally")

    print("\n" + "="*60)
    print("STEP 2: Tokenizing dataset using VyvoTTS audio tokenizer")
    print("="*60)

    # Choose model type: "lfm2", "qwen3", or "granite"
    model_type = "granite"

    print(f"\nProcessing with VyvoTTS tokenizer ({model_type.upper()} model)")
    print(f"Input dataset: {temp_dataset}")
    print(f"Output dataset: {output_dataset}")

    # Process using VyvoTTS tokenizer
    # This will:
    # - Tokenize audio using SNAC codec on CUDA
    # - Tokenize text using the specified model tokenizer
    # - Create training sequences with proper formatting
    # - Push final tokenized dataset to HuggingFace Hub
    process_dataset(
        original_dataset=temp_dataset,
        output_dataset=output_dataset,
        model_type=model_type,
        text_field="text_scribe",
        target_sample_rate=24000
    )

    print("\n" + "="*60)
    print("✓ Dataset processing complete!")
    print(f"  Final tokenized dataset pushed to: {output_dataset}")
    print("="*60)


if __name__ == "__main__":
    prepare_and_process_dataset()
