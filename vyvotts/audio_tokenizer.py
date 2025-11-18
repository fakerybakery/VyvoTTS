import os
import yaml
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from huggingface_hub import snapshot_download
from snac import SNAC
from transformers import AutoTokenizer


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


def tokenise_audio(waveform, snac_model, ds_sample_rate, target_sample_rate, audio_tokens_start):
    """
    Tokenize audio waveform using SNAC codec.

    Args:
        waveform: Audio array from dataset
        snac_model: SNAC model instance
        ds_sample_rate: Original dataset sample rate
        target_sample_rate: Target sample rate (24000)
        audio_tokens_start: Offset for audio tokens

    Returns:
        List of audio token IDs with proper offsets applied
    """
    # Convert to tensor and prepare for processing
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to target sample rate if needed
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")

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
    target_sample_rate=24000
):
    """
    Process dataset: tokenize audio and text, create training sequences.

    Args:
        original_dataset: HuggingFace dataset path to process
        output_dataset: HuggingFace dataset path for output
        model_type: Model type - "qwen3", "lfm2", or "granite" (default: "qwen3")
        text_field: Name of text field in dataset (default: "text_scribe")
        target_sample_rate: Target audio sample rate (default: 24000)
    """
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
    print("Loading SNAC model: hubertsiuzdak/snac_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    # Define processing functions
    def add_codes(example):
        """Add audio codes to dataset example."""
        codes_list = None

        try:
            audio_data = example.get("audio")
            if audio_data and "array" in audio_data:
                audio_array = audio_data["array"]
                codes_list = tokenise_audio(
                    audio_array,
                    snac_model,
                    ds_sample_rate,
                    target_sample_rate,
                    AUDIO_TOKENS_START
                )
        except Exception as e:
            print(f"Skipping row due to error: {e}")

        example["codes_list"] = codes_list
        return example

    # Process dataset: tokenize audio
    print("Tokenizing audio...")
    ds = ds.map(add_codes, remove_columns=["audio"])

    # Load text tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    num_proc = os.cpu_count() - 2

    # Filter out failed tokenizations
    print("Filtering invalid examples...")
    ds = ds.filter(lambda x: x["codes_list"] is not None)
    ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

    # Remove duplicate frames
    def remove_duplicate_frames_wrapper(example):
        """Wrapper for remove_duplicate_frames."""
        example["codes_list"] = remove_duplicate_frames(example["codes_list"])
        return example

    print("Removing duplicate frames...")
    ds = ds.map(remove_duplicate_frames_wrapper, num_proc=num_proc)

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
    print("Creating input sequences...")
    ds = ds.map(
        create_input_ids,
        num_proc=num_proc,
        remove_columns=[text_field, "codes_list"]
    )

    # Keep only training columns
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    # Upload processed dataset
    print(f"Pushing dataset to: {output_dataset}")
    ds.push_to_hub(output_dataset)
    print("Done!")
