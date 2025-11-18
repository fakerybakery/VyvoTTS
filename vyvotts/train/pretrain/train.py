import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml
import wandb
from huggingface_hub import HfApi
from pathlib import Path
import os

# Get config file path relative to this script
CONFIG_FILE = Path(__file__).parent.parent.parent / "configs" / "train" / "granite_pretrain.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
max_seq_length = config.get("max_seq_length", 8192)  # Default to 8192 if not specified

# Parse ratio from config (e.g., "2:1" -> 2)
ratio_str = config["ratio"]
initial_ratio = int(ratio_str.split(":")[0])
final_ratio = 1  # Target ratio is 1:1


class GradualRatioDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total, initial_ratio=2, final_ratio=1, total_steps=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = total_steps

        # Calculate length based on the maximum ratio to ensure we have enough data
        max_ratio = max(initial_ratio, final_ratio)
        num_cycles_ds1 = len(dataset1) // (batch_total * max_ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)

        # Use initial ratio for length calculation
        self.length = self.num_cycles * (initial_ratio + 1) * batch_total

        # For tracking current step
        self.current_step = 0

    def set_current_step(self, step):
        """Called by trainer to update current step for ratio calculation"""
        self.current_step = step

    def get_current_ratio(self):
        """Calculate current ratio based on training progress"""
        if self.total_steps is None or self.total_steps == 0:
            return self.initial_ratio

        # Linear interpolation from initial_ratio to final_ratio
        progress = min(self.current_step / self.total_steps, 1.0)
        current_ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(int(round(current_ratio)), self.final_ratio)

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):
        current_ratio = self.get_current_ratio()

        # Compute the cycle length in terms of samples with current ratio
        cycle_length = (current_ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < current_ratio * self.batch_total:
            # Text dataset (dataset1)
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            ds1_index = cycle * current_ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch

            # Handle index overflow by wrapping around
            if ds1_index >= len(self.dataset1):
                ds1_index = ds1_index % len(self.dataset1)

            return self.dataset1[ds1_index]
        else:
            # TTS dataset (dataset2)
            sample_in_batch = pos_in_cycle - current_ratio * self.batch_total
            ds2_index = cycle * self.batch_total + sample_in_batch

            # Handle index overflow by wrapping around
            if ds2_index >= len(self.dataset2):
                ds2_index = ds2_index % len(self.dataset2)

            return self.dataset2[ds2_index]


class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class DeepSpeedTrainer(Trainer):
    def __init__(self, *args, initial_ratio=2, final_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()

        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.text_step = 0
        self.audio_step = 0

        # Calculate total steps for gradual ratio adjustment
        self.total_steps = self.calculate_total_steps()

    def calculate_total_steps(self):
        """Calculate total training steps"""
        num_update_steps_per_epoch = len(self.train_dataset) // (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        return int(num_update_steps_per_epoch * self.args.num_train_epochs)

    def get_current_ratio(self):
        """Get current ratio based on training progress"""
        if self.total_steps == 0:
            return self.initial_ratio

        progress = min(self.state.global_step / self.total_steps, 1.0)
        current_ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return max(int(round(current_ratio)), self.final_ratio)

    def get_train_dataloader(self):
        # Update dataset with total steps info
        if hasattr(self.train_dataset, 'total_steps'):
            self.train_dataset.total_steps = self.total_steps

        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Update dataset with current step
        if hasattr(self.train_dataset, 'set_current_step'):
            self.train_dataset.set_current_step(self.state.global_step)

        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if self.is_world_process_zero():
            current_ratio = self.get_current_ratio()
            global_step = self.state.global_step

            # Log current ratio
            if "loss" in logs:
                wandb.log({"current_ratio": current_ratio, "global_step": global_step})

            # Each cycle is (current_ratio + 1) steps
            cycle_length = current_ratio + 1
            step_in_cycle = global_step % cycle_length

            # Only log to wandb if 'loss' is in the logs dictionary
            if "loss" in logs:
                if step_in_cycle < current_ratio:
                    # Text loss
                    wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
                    self.text_step += 1
                else:
                    # Audio loss
                    wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
                    self.audio_step += 1


def data_collator(features):
    input_ids = [f["input_ids"][:max_seq_length] for f in features]  # Truncate to max_seq_length

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"][:max_seq_length] for f in features]  # Truncate

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"][:max_seq_length] for f in features]  # Truncate

    # Debug: Print actual lengths before padding
    lengths = [len(ids) for ids in input_ids]
    max_len = max(lengths)
    print(f"[DEBUG] Batch sizes before padding: min={min(lengths)}, max={max_len}, avg={sum(lengths)/len(lengths):.0f}")

    if max_len > max_seq_length:
        print(f"[WARNING] Found sequence longer than {max_seq_length}: {max_len} tokens!")

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    print(f"[DEBUG] Final batch shape: {input_ids.shape}, VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


wandb.init(project=project_name, name=run_name)

# DeepSpeed will initialize distributed environment automatically
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Initialize model with proper dtype for Flash Attention 2.0
print("="*60)
print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Load efficiently on CPU first
)

print(f"Model loaded on CPU")

# Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()

print("Gradient checkpointing: ENABLED")
print("DeepSpeed ZeRO-3 with CPU offload will shard this model across GPUs")
print("="*60)

number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

ds1 = load_dataset(dsn1, split="train")
ds2 = load_dataset(dsn2, split="train")

# Filter out sequences longer than max_seq_length to prevent OOM
print("="*60)
print(f"Filtering sequences longer than {max_seq_length} tokens...")
print(f"Dataset 1 before filtering: {len(ds1)} examples")
print(f"Dataset 2 before filtering: {len(ds2)} examples")

# Check some sample lengths before filtering
sample_lengths_1 = [len(ds1[i]["input_ids"]) for i in range(min(10, len(ds1)))]
sample_lengths_2 = [len(ds2[i]["input_ids"]) for i in range(min(10, len(ds2)))]
print(f"Sample lengths from ds1: {sample_lengths_1}")
print(f"Sample lengths from ds2: {sample_lengths_2}")

ds1 = ds1.filter(lambda x: len(x["input_ids"]) <= max_seq_length)
ds2 = ds2.filter(lambda x: len(x["input_ids"]) <= max_seq_length)

print(f"Dataset 1 after filtering: {len(ds1)} examples")
print(f"Dataset 2 after filtering: {len(ds2)} examples")

# Check sample lengths after filtering
if len(ds1) > 0:
    sample_lengths_1 = [len(ds1[i]["input_ids"]) for i in range(min(10, len(ds1)))]
    print(f"Sample lengths from ds1 after filtering: {sample_lengths_1}")
if len(ds2) > 0:
    sample_lengths_2 = [len(ds2[i]["input_ids"]) for i in range(min(10, len(ds2)))]
    print(f"Sample lengths from ds2 after filtering: {sample_lengths_2}")
print("="*60)

batch_total = batch_size * number_processes

# Calculate total steps for the dataset
num_update_steps_per_epoch = len(ds1) // (batch_size * number_processes)
total_steps = int(num_update_steps_per_epoch * epochs)

train_dataset = GradualRatioDataset(
    ds1, ds2, batch_total,
    initial_ratio=initial_ratio,
    final_ratio=final_ratio,
    total_steps=total_steps
)

# DeepSpeed config path
DEEPSPEED_CONFIG = Path(__file__).parent.parent.parent / "configs" / "train" / "deepspeed_config.json"

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,  # Accumulate to reduce memory pressure
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    deepspeed=str(DEEPSPEED_CONFIG),  # Use DeepSpeed ZeRO-3 with CPU offload
    report_to="wandb",
    save_steps=save_steps,
    save_strategy="steps",
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    max_grad_norm=1.0,  # Gradient clipping for stability
    gradient_checkpointing=True,  # Enable gradient checkpointing
    dataloader_num_workers=2,  # Use some workers for data loading
    dataloader_pin_memory=True,
)

trainer = DeepSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    initial_ratio=initial_ratio,
    final_ratio=final_ratio
)

print("="*60)
print(f"Starting training with ratio progression: {initial_ratio}:1 -> {final_ratio}:1")
print(f"Total steps: {total_steps}")
print(f"Max sequence length: {max_seq_length} tokens")
print(f"Batch size per GPU: {batch_size}")
print(f"Gradient accumulation: 4 steps")
print(f"Effective batch size per GPU: {batch_size * 4}")
print(f"Number of GPUs: {number_processes}")
print(f"Gradient checkpointing: ENABLED")
print(f"DeepSpeed ZeRO-3: ENABLED")
print(f"CPU Offload (params + optimizer): ENABLED")
print(f"Expected VRAM per GPU: 8-15GB (down from 192GB!)")
print("="*60)

trainer.train()
