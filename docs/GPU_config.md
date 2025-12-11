# GPU-Specific Configuration Guide

Quick reference for adjusting `config.yaml` for different GPU configurations.

## Configuration Files

- **config.yaml**: Main configuration file with all training parameters
- **train_from_config.py**: Config-based training script (recommended)
- **train_grpo.py**: Original hardcoded training script (legacy)

## Recommended Workflow

Use the config-based training for easy GPU switching:

```bash
# 1. Edit config.yaml with your GPU settings
# 2. Run training
uv run GRPO/train_from_config.py
```

## GPU Configuration Presets

### Tesla V100 32GB (Current Default)

```yaml
data:
  train_batch_size: 8
  dataloader_num_workers: 4

rollout:
  dtype: "float16"  # V100 doesn't support bfloat16
  n: 8
  log_prob_micro_batch_size: 8
  gpu_memory_utilization: 0.85

actor:
  ppo_mini_batch_size: 16
  ppo_micro_batch_size_per_gpu: 2

trainer:
  mixed_precision: "fp16"
```

**Expected Performance:**
- GPU Memory: ~27-28GB (85%)
- Throughput: ~8-10 samples/sec
- Training stability: Good with gradient clipping

---

### A100 40GB

```yaml
data:
  train_batch_size: 16  # 2x V100
  dataloader_num_workers: 8

rollout:
  dtype: "bfloat16"  # A100 optimized
  n: 16  # 2x V100
  log_prob_micro_batch_size: 16
  gpu_memory_utilization: 0.90

actor:
  ppo_mini_batch_size: 32  # 2x V100
  ppo_micro_batch_size_per_gpu: 4

trainer:
  mixed_precision: "bf16"  # A100 native
```

**Expected Performance:**
- GPU Memory: ~36GB (90%)
- Throughput: ~20-25 samples/sec
- Training stability: Excellent (bf16 more stable)

---

### A100 80GB

```yaml
data:
  train_batch_size: 32  # 4x V100
  dataloader_num_workers: 16

rollout:
  dtype: "bfloat16"
  n: 32  # 4x V100
  log_prob_micro_batch_size: 32
  gpu_memory_utilization: 0.90

actor:
  ppo_mini_batch_size: 64  # 4x V100
  ppo_micro_batch_size_per_gpu: 8

trainer:
  mixed_precision: "bf16"
```

**Expected Performance:**
- GPU Memory: ~72GB (90%)
- Throughput: ~40-50 samples/sec
- Training stability: Excellent

---

### RTX 3090 24GB

```yaml
data:
  train_batch_size: 4  # 0.5x V100
  dataloader_num_workers: 4

rollout:
  dtype: "float16"
  n: 4  # 0.5x V100
  log_prob_micro_batch_size: 4
  gpu_memory_utilization: 0.80  # Conservative

actor:
  ppo_mini_batch_size: 8  # 0.5x V100
  ppo_micro_batch_size_per_gpu: 2

trainer:
  mixed_precision: "fp16"
```

**Expected Performance:**
- GPU Memory: ~19-20GB (80%)
- Throughput: ~4-6 samples/sec
- Training stability: Good with lower batch sizes

---

### RTX 4090 24GB

```yaml
data:
  train_batch_size: 6  # 0.75x V100
  dataloader_num_workers: 6

rollout:
  dtype: "float16"
  n: 6  # 0.75x V100
  log_prob_micro_batch_size: 6
  gpu_memory_utilization: 0.85

actor:
  ppo_mini_batch_size: 12  # 0.75x V100
  ppo_micro_batch_size_per_gpu: 2

trainer:
  mixed_precision: "fp16"
```

**Expected Performance:**
- GPU Memory: ~20-21GB (85%)
- Throughput: ~8-12 samples/sec (faster than 3090)
- Training stability: Excellent

---

## Key Parameters Explained

### train_batch_size
Number of prompts processed per training step. Higher = better GPU utilization but more memory.

### rollout.n
Number of response samples generated per prompt. Higher = better GRPO gradients but more memory and time.

### ppo_mini_batch_size
Total batch size for PPO updates. Should be `train_batch_size * rollout.n / 4` typically.

### ppo_micro_batch_size_per_gpu
Actual batch size processed per GPU. Lower = less memory, higher = faster.

### gpu_memory_utilization
Target percentage of GPU memory for vLLM. Leave headroom for PyTorch training.

## Memory Optimization Tips

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch sizes** (in order of impact):
   - `rollout.n` (biggest impact)
   - `train_batch_size`
   - `ppo_mini_batch_size`

2. **Reduce sequence lengths**:
   - `max_prompt_length`
   - `max_response_length`

3. **Adjust micro batch size**:
   - Increase `ppo_micro_batch_size_per_gpu` (trades speed for memory)

4. **Lower memory utilization**:
   - Reduce `gpu_memory_utilization` to 0.75-0.80

## Performance Optimization Tips

To maximize training speed:

1. **Increase batch sizes** (if memory allows)
2. **Increase `dataloader_num_workers`** (match CPU cores)
3. **Use appropriate precision**:
   - A100: Use `bf16` (better than fp16)
   - V100/3090/4090: Use `fp16`
4. **Increase `rollout.n`** for better sample diversity

## Quick Comparison Table

| GPU | Batch | Rollout.n | Memory | Throughput | Cost/Hour |
|-----|-------|-----------|--------|------------|-----------|
| V100 32GB | 8 | 8 | 85% | 8-10/s | Low |
| A100 40GB | 16 | 16 | 90% | 20-25/s | Medium |
| A100 80GB | 32 | 32 | 90% | 40-50/s | High |
| RTX 3090 | 4 | 4 | 80% | 4-6/s | Low |
| RTX 4090 | 6 | 6 | 85% | 8-12/s | Medium |

## Testing Your Configuration

After editing config.yaml, test memory usage:

```bash
# Start training and monitor GPU memory
nvidia-smi -l 1

# Watch for OOM errors in the training log
# If memory usage is below 70%, you can increase batch sizes
# If hitting OOM, reduce batch sizes
```
