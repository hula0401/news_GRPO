# News GRPO Training Platform

GRPO (Group Relative Policy Optimization) training platform using VERL framework. Currently configured for GSM8K dataset with Qwen 0.5B model, designed to be extended for news data training.

## Overview

This repository provides a complete pipeline for training language models using GRPO reinforcement learning:
- Data preparation pipeline for GSM8K (extensible to news data)
- GRPO training with VERL framework
- Inference and evaluation tools
- GPU server deployment ready

## Project Structure

```
news_GRPO/
├── GRPO/
│   ├── prepare_gsm8k.py      # Dataset preparation script
│   ├── train_grpo.py          # Main training script
│   ├── inference.py           # Inference and evaluation
│   ├── config.yaml            # Training configuration
│   ├── setup_verl.sh          # VERL installation script
│   └── GRPO_gsm8k.ipynb      # Original Colab notebook (reference)
├── data/
│   └── gsm8k/                 # Prepared GSM8K dataset (generated)
├── checkpoints/               # Model checkpoints (generated)
├── docs/                      # Documentation
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Requirements

- Python >= 3.12
- CUDA-compatible GPU (for training)
- 16GB+ GPU memory recommended
- Linux environment (for GPU server deployment)

## Installation

### On GPU Server

1. Clone the repository and navigate to project directory

2. Install dependencies using uv:
```bash
uv sync
```
if uv is not installed:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install VERL and apply patches:
```bash
chmod +x GRPO/setup_verl.sh
bash GRPO/setup_verl.sh
```

### Manual VERL Installation

If the setup script fails:
```bash
uv pip install git+https://github.com/volcengine/verl.git
uv pip install 'numpy<2.0.0'
```

## Usage

### 1. Prepare GSM8K Dataset

Download and format the GSM8K dataset:
```bash
uv run GRPO/prepare_gsm8k.py
```

This creates `data/gsm8k/train.parquet` and `data/gsm8k/test.parquet`.

### 2. Train with GRPO

Start training with default configuration:
```bash
uv run GRPO/train_grpo.py
```

Customize training parameters:
```bash
uv run GRPO/train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --experiment-name my-experiment \
  --n-gpus 1 \
  --batch-size 4 \
  --learning-rate 1e-6
```

Training arguments:
- `--model`: HuggingFace model path or local checkpoint
- `--experiment-name`: Name for this training run
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)
- `--data-dir`: GSM8K data directory (default: data/gsm8k)
- `--n-gpus`: Number of GPUs to use
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for actor model

### 3. Run Inference

Test the trained model:
```bash
uv run GRPO/inference.py \
  --model checkpoints/qwen-grpo-gsm8k/final_model \
  --question "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
```

Interactive mode:
```bash
uv run GRPO/inference.py \
  --model checkpoints/qwen-grpo-gsm8k/final_model
```

## Configuration

Edit `GRPO/config.yaml` to customize hyperparameters:
- Model settings (checkpoint, attention implementation)
- Data batch sizes and sequence lengths
- GRPO algorithm parameters (temperature, rollout samples)
- Training settings (GPUs, learning rate)

## Common Issues

### Package Compatibility
- **Numpy version conflict**: VERL requires numpy < 2.0. The setup script handles this automatically.
- **Flash Attention compilation**: The setup script patches VERL to use SDPA attention instead, avoiding lengthy Flash Attention compilation.

### Memory Issues
- Reduce `train_batch_size` and `ppo_micro_batch_size_per_gpu` in training command or config
- Enable gradient checkpointing (enabled by default)
- Use smaller model (e.g., Qwen2.5-0.5B instead of larger variants)

### Ray/VERL Issues
- Ensure Ray is properly installed: `uv pip install ray>=2.10.0`
- Check GPU availability: `nvidia-smi`
- Set environment variables (handled by setup script)

## Next Steps

### Extending to News Data

1. Create a news data preparation script following `prepare_gsm8k.py` structure
2. Format news data with:
   - `prompt`: The input news context
   - `reward_model`: Define your reward criteria (quality, coherence, etc.)
3. Update config.yaml with news-specific parameters
4. Run training with news data path

### Model Scaling

To use larger models:
1. Update `model.path` in config or `--model` argument
2. Increase GPU memory or use multi-GPU training (`--n-gpus`)
3. Adjust batch sizes based on available memory

## Resources

- [VERL GitHub](https://github.com/volcengine/verl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Qwen Models](https://huggingface.co/Qwen)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)

## Development

This project uses:
- **uv** for dependency management
- **VERL** for GRPO training framework
- **Transformers** for model loading and inference
- **Datasets** for data handling
