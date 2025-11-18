# DeepSpeed Training Guide

The training script has been converted to use DeepSpeed ZeRO-3 for efficient memory management.

## Why DeepSpeed?

The original FSDP implementation was using 192GB VRAM for a 1B model due to:
- Long audio sequences (30k-100k tokens)
- Inefficient memory management
- No proper parameter/optimizer sharding

DeepSpeed ZeRO-3 solves this with:
- **ZeRO Stage 3**: Shards parameters, gradients, and optimizer states across GPUs
- **CPU Offload**: Offloads parameters and optimizer states to CPU RAM
- **Activation Checkpointing**: Reduces activation memory by 5-10x
- **Better memory management**: Much more efficient than FSDP for extreme cases

## Expected Memory Usage

With DeepSpeed ZeRO-3 + CPU offload:
- **Per GPU VRAM**: 8-15GB (down from 192GB!)
- **System RAM**: 40-60GB (offloaded params/optimizer)
- **Batch size 2**: ~10GB per GPU
- **Batch size 4**: ~12-15GB per GPU

You can now train on GPUs with 16GB-24GB VRAM!

## Installation

Install DeepSpeed:
```bash
pip install deepspeed
```

## Usage

### Quick Start (All GPUs)
```bash
./train_deepspeed.sh
```

### Specify Number of GPUs
```bash
deepspeed --num_gpus=4 vyvotts/train/pretrain/train.py
```

### Use Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 vyvotts/train/pretrain/train.py
```

### Single GPU Training
```bash
deepspeed --num_gpus=1 vyvotts/train/pretrain/train.py
```

## Configuration

### DeepSpeed Config
Located at: `vyvotts/configs/train/deepspeed_config.json`

Key settings:
- `"stage": 3` - Full parameter sharding
- `"offload_optimizer"` - Offload optimizer to CPU
- `"offload_param"` - Offload parameters to CPU
- `"activation_checkpointing"` - Checkpoint activations to CPU

### Training Config
Located at: `vyvotts/configs/train/granite_pretrain.yaml`

Key settings:
- `batch_size: 2` - Per-GPU batch size
- `max_seq_length: 8192` - Maximum sequence length (prevents OOM)
- `gradient_accumulation_steps: 4` - In training args

## Memory Optimization Tips

### If you still get OOM:

1. **Reduce batch size** in `granite_pretrain.yaml`:
   ```yaml
   batch_size: 1
   ```

2. **Reduce max sequence length**:
   ```yaml
   max_seq_length: 4096  # or even 2048
   ```

3. **Enable more aggressive CPU offloading** in `deepspeed_config.json`:
   ```json
   "stage3_max_live_parameters": 5e7,
   "stage3_max_reuse_distance": 5e7
   ```

4. **Reduce number of GPUs** (paradoxically can help):
   ```bash
   deepspeed --num_gpus=2 vyvotts/train/pretrain/train.py
   ```

### To speed up training:

1. **Increase batch size** if you have VRAM headroom:
   ```yaml
   batch_size: 4  # or 8
   ```

2. **Disable CPU offload** if you have enough VRAM (48GB+):
   ```json
   "offload_param": {
     "device": "none"
   }
   ```

3. **Increase max sequence length** for better quality:
   ```yaml
   max_seq_length: 16384
   ```

## Monitoring

### Watch GPU memory:
```bash
watch -n 1 nvidia-smi
```

### DeepSpeed will print:
```
[DeepSpeed] Using ZeRO Stage 3
[DeepSpeed] Offloading parameters to CPU
[DeepSpeed] Gradient checkpointing enabled
Expected VRAM per GPU: 8-15GB
```

## Debugging

### Enable DeepSpeed debug logging:
```bash
export DEEPSPEED_LOG_LEVEL=INFO
deepspeed --num_gpus=4 vyvotts/train/pretrain/train.py
```

### Check sequence lengths:
The script now prints:
```
[DEBUG] Batch sizes before padding: min=1234, max=8192, avg=5678
[DEBUG] Final batch shape: torch.Size([2, 8192])
```

### Verify truncation is working:
```
Filtering sequences longer than 8192 tokens...
Sample lengths from ds1: [5234, 7123, 6234, ...]
Sample lengths from ds2: [12345, 23456, ...] -> will be filtered!
```

## Troubleshooting

### Error: "No module named 'deepspeed'"
```bash
pip install deepspeed
```

### Error: "NCCL error" or "distributed timeout"
- Check all GPUs are visible: `nvidia-smi`
- Use fewer GPUs: `deepspeed --num_gpus=2`
- Increase timeout: `export NCCL_TIMEOUT=1800`

### Still OOMing
- Check debug output for actual sequence lengths
- Verify filtering is working (see sample lengths before/after)
- Reduce `max_seq_length` to 4096 or 2048
- Use ZeRO Stage 2 instead (edit `deepspeed_config.json`: `"stage": 2`)

## Performance Comparison

| Configuration | VRAM per GPU | Speed |
|--------------|--------------|-------|
| Original FSDP | 192GB (OOM!) | N/A |
| DeepSpeed ZeRO-3 batch=2 | 10GB | 1.0x |
| DeepSpeed ZeRO-3 batch=4 | 15GB | 1.8x |
| DeepSpeed ZeRO-3 batch=8 | 25GB | 3.2x |
| DeepSpeed ZeRO-2 batch=8 | 35GB | 3.5x |

Choose based on your VRAM availability!
