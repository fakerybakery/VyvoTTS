import os
import yaml
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from huggingface_hub import snapshot_download
from snac import SNAC
from transformers import AutoTokenizer
from accelerate import Accelerator
from typing import List
import numpy as np


def load_config(config_path):
    """
    Load tokenizer configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def tokenise_audio(waveform, snac_model, ds_sample_rate, target_sample_rate, audio_tokens_start, device="cuda"):
    """
    Tokenize audio waveform using SNAC codec.

    Args:
        waveform: Audio array from dataset
        snac_model: SNAC model instance
        ds_sample_rate: Original dataset sample rate
        target_sample_rate: Target sample rate (24000)
        audio_tokens_start: Offset for audio tokens
        device: Device to use for processing

    Returns:
        List of audio token IDs with proper offsets applied
    """
    # Convert to tensor and prepare for processing
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to target sample rate if needed
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to(device)

    # Generate SNAC codes
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Interleave codes from 3 codebooks with proper offsets
    # SNAC uses hierarchical vector quantization with 3 levels
    all_codes = []
    num_frames = codes[0].shape[1]

    for i in range(num_frames):
        # Level 0: 1 code per frame
        all_codes.append(codes[0][0][i].item() + audio_tokens_start)

        # Level 1: 2 codes per frame
        all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)

        # Level 2: 4 codes per frame
        all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2 * 4096))
        all_codes.append(codes[2][0][4*i + 1].item() + audio_tokens_start + (3 * 4096))

        # Continue level 1 and 2 interleaving
        all_codes.append(codes[1][0][2*i + 1].item() + audio_tokens_start + (4 * 4096))
        all_codes.append(codes[2][0][4*i + 2].item() + audio_tokens_start + (5 * 4096))
        all_codes.append(codes[2][0][4*i + 3].item() + audio_tokens_start + (6 * 4096))

    return all_codes


def tokenise_audio_batch(waveforms: List, snac_model, ds_sample_rate, target_sample_rate, audio_tokens_start, device="cuda"):
    """
    Tokenize a batch of audio waveforms using SNAC codec for efficient GPU utilization.

    Args:
        waveforms: List of audio arrays from dataset
        snac_model: SNAC model instance
        ds_sample_rate: Original dataset sample rate
        target_sample_rate: Target sample rate (24000)
        audio_tokens_start: Offset for audio tokens
        device: Device to use for processing

    Returns:
        List of lists of audio token IDs with proper offsets applied
    """
    if not waveforms:
        return []

    # Convert all waveforms to tensors and resample
    resampler = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
    processed_waveforms = []

    for waveform in waveforms:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        waveform = resampler(waveform)
        processed_waveforms.append(waveform)

    # Find max length for padding
    max_len = max(w.shape[-1] for w in processed_waveforms)

    # Pad all waveforms to same length
    padded_waveforms = []
    original_lengths = []

    for waveform in processed_waveforms:
        original_lengths.append(waveform.shape[-1])
        if waveform.shape[-1] < max_len:
            padding = max_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(waveform)

    # Stack into batch
    batch = torch.stack(padded_waveforms).to(device)

    # Generate SNAC codes for entire batch
    with torch.inference_mode():
        codes = snac_model.encode(batch)

    # Process each item in batch
    all_results = []
    for batch_idx in range(len(waveforms)):
        all_codes = []
        num_frames = codes[0].shape[1]

        for i in range(num_frames):
            # Level 0: 1 code per frame
            all_codes.append(codes[0][batch_idx][i].item() + audio_tokens_start)

            # Level 1: 2 codes per frame
            all_codes.append(codes[1][batch_idx][2*i].item() + audio_tokens_start + 4096)

            # Level 2: 4 codes per frame
            all_codes.append(codes[2][batch_idx][4*i].item() + audio_tokens_start + (2 * 4096))
            all_codes.append(codes[2][batch_idx][4*i + 1].item() + audio_tokens_start + (3 * 4096))

            # Continue level 1 and 2 interleaving
            all_codes.append(codes[1][batch_idx][2*i + 1].item() + audio_tokens_start + (4 * 4096))
            all_codes.append(codes[2][batch_idx][4*i + 2].item() + audio_tokens_start + (5 * 4096))
            all_codes.append(codes[2][batch_idx][4*i + 3].item() + audio_tokens_start + (6 * 4096))

        all_results.append(all_codes)

    return all_results


def remove_duplicate_frames(codes_list):
    """
    Remove consecutive duplicate audio frames to reduce redundancy.

    Each frame consists of 7 codes (1 + 2 + 4 from 3 SNAC codebook levels).
    Frames with identical first codes are considered duplicates.

    Args:
        codes_list: List of audio codes

    Returns:
        Deduplicated codes list
    """
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    # Keep first frame
    result = codes_list[:7]
    removed_frames = 0

    # Check each subsequent frame
    for i in range(7, len(codes_list), 7):
        current_first_code = codes_list[i]
        previous_first_code = result[-7]

        if current_first_code != previous_first_code:
            result.extend(codes_list[i:i+7])
        else:
            removed_frames += 1

    return result


def process_dataset(
    original_dataset,
    output_dataset,
    model_type="qwen3",
    text_field="text_scribe",
    target_sample_rate=24000,
    batch_size=8
):
    """
    Process dataset: tokenize audio and text, create training sequences.
    Uses Accelerate for multi-GPU support and batch processing for efficiency.

    Args:
        original_dataset: HuggingFace dataset path to process
        output_dataset: HuggingFace dataset path for output
        model_type: Model type - "qwen3", "lfm2", or "granite" (default: "qwen3")
        text_field: Name of text field in dataset (default: "text_scribe")
        target_sample_rate: Target audio sample rate (default: 24000)
        batch_size: Number of audio files to process per batch (default: 8)
    """
    # Initialize Accelerate
    accelerator = Accelerator()
    device = accelerator.device
    # Set tokenizer and config based on model type
    if model_type == "qwen3":
        tokenizer_model = "Qwen/Qwen3-0.6B"
        config_path = "vyvotts/configs/inference/qwen3.yaml"
    elif model_type == "lfm2":
        tokenizer_model = "LiquidAI/LFM2-350M"
        config_path = "vyvotts/configs/inference/lfm2.yaml"
    elif model_type == "granite":
        tokenizer_model = "ibm-granite/granite-4.0-h-1b-base"
        config_path = "vyvotts/configs/inference/granite.yaml"
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'qwen3', 'lfm2', or 'granite'")

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    TOKENIZER_LENGTH = config['TOKENIZER_LENGTH']
    START_OF_TEXT = config['START_OF_TEXT']
    END_OF_TEXT = config['END_OF_TEXT']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_AI = config['START_OF_AI']
    END_OF_AI = config['END_OF_AI']
    PAD_TOKEN = config['PAD_TOKEN']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    # Download dataset
    print(f"Downloading dataset: {original_dataset}")
    snapshot_download(
        repo_id=original_dataset,
        repo_type="dataset",
        revision="main",
        max_workers=64,
    )

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(original_dataset, split="train")
    ds_sample_rate = ds[0]["audio"]["sampling_rate"]

    # Load SNAC model
    if accelerator.is_main_process:
        print(f"Loading SNAC model: hubertsiuzdak/snac_24khz on {accelerator.num_processes} GPU(s)")
        print(f"Batch size: {batch_size} audio files per GPU")

    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)
    snac_model.eval()

    # Define batched processing function
    def add_codes_batch(examples):
        """Add audio codes to a batch of dataset examples."""
        batch_waveforms = []
        batch_indices = []

        # Collect valid audio samples
        for idx, audio_data in enumerate(examples.get("audio", [])):
            if audio_data and "array" in audio_data:
                batch_waveforms.append(audio_data["array"])
                batch_indices.append(idx)

        # Initialize codes_list for all examples
        codes_lists = [None] * len(examples.get("audio", []))

        # Process batch if we have valid waveforms
        if batch_waveforms:
            try:
                batch_codes = tokenise_audio_batch(
                    batch_waveforms,
                    snac_model,
                    ds_sample_rate,
                    target_sample_rate,
                    AUDIO_TOKENS_START,
                    device
                )

                # Map results back to original indices
                for batch_idx, original_idx in enumerate(batch_indices):
                    codes_lists[original_idx] = batch_codes[batch_idx]

            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error processing batch: {e}")

        examples["codes_list"] = codes_lists
        return examples

    # Process dataset: tokenize audio with batching
    if accelerator.is_main_process:
        print(f"Tokenizing audio in batches of {batch_size}...")

    with accelerator.main_process_first():
        ds = ds.map(
            add_codes_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=["audio"]
        )

    # Load text tokenizer
    if accelerator.is_main_process:
        print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    num_proc = os.cpu_count() - 2

    # Filter out failed tokenizations
    if accelerator.is_main_process:
        print("Filtering invalid examples...")

    with accelerator.main_process_first():
        ds = ds.filter(lambda x: x["codes_list"] is not None)
        ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

    # Remove duplicate frames
    def remove_duplicate_frames_wrapper(example):
        """Wrapper for remove_duplicate_frames."""
        example["codes_list"] = remove_duplicate_frames(example["codes_list"])
        return example

    if accelerator.is_main_process:
        print("Removing duplicate frames...")

    with accelerator.main_process_first():
        ds = ds.map(remove_duplicate_frames_wrapper, num_proc=num_proc)

    if accelerator.is_main_process:
        print(f"""
NOTE: Text prompt customization
You can modify the text prompt in create_input_ids() below.
For multispeaker models, ensure your dataset has a "source" field.
- Single-speaker: uses example['{text_field}']
- Multi-speaker: uses example['source']: example['{text_field}']
""")

    def create_input_ids(example):
        """
        Create training input sequence with proper formatting.

        Format: [HUMAN] text [/HUMAN] [AI] [SPEECH] audio_codes [/SPEECH] [/AI]
        """
        # Determine whether to include the source field
        if "source" in example:
            text_prompt = f"{example['source']}: {example[text_field]}"
        else:
            text_prompt = example[text_field]

        # Tokenize text input
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        example["text_tokens"] = text_ids

        # Construct full sequence with special tokens
        input_ids = (
            [START_OF_HUMAN]
            + example["text_tokens"]
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + example["codes_list"]
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )

        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)

        return example

    # Create final training sequences
    if accelerator.is_main_process:
        print("Creating input sequences...")

    with accelerator.main_process_first():
        ds = ds.map(
            create_input_ids,
            num_proc=num_proc,
            remove_columns=[text_field, "codes_list"]
        )

    # Keep only training columns
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

    with accelerator.main_process_first():
        ds = ds.remove_columns(columns_to_remove)

    # Upload processed dataset (only from main process)
    if accelerator.is_main_process:
        print(f"Pushing dataset to: {output_dataset}")
        ds.push_to_hub(output_dataset)
        print("Done!")

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
