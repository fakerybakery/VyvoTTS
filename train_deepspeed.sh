#!/bin/bash

# Launch VyvoTTS pretraining with DeepSpeed ZeRO-3
# This uses all available GPUs by default

# Number of GPUs to use (auto-detect or set manually)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=========================================="
echo "Launching VyvoTTS Training with DeepSpeed"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Use deepspeed launcher (better than torchrun for DeepSpeed)
deepspeed --num_gpus=$NUM_GPUS vyvotts/train/pretrain/train.py

# Alternative: Use specific GPUs only
# CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 vyvotts/train/pretrain/train.py
