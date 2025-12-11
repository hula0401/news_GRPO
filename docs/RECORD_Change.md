# Change Record

## 2025-12-11: Optimized Training Parameters for V100 32GB

Optimized all training hyperparameters to maximize throughput and GPU utilization on Tesla V100 32GB.

### Parameter Changes (train_grpo.py)
**Data Configuration:**
- train_batch_size: 4 → 8 (line 34)
- dataloader_num_workers: 2 → 4 (line 119)

**Rollout Configuration:**
- rollout.n: 4 → 8 samples per prompt (line 131)
- log_prob_micro_batch_size: 4 → 8 (line 133)
- Added gpu_memory_utilization=0.85 for V100 (line 134)

**Actor Training:**
- ppo_mini_batch_size: 4 → 16 (line 138)
- ppo_micro_batch_size_per_gpu: 4 → 2 (line 139)
- Added ppo_epochs=1 for GRPO (line 140)

**Training Settings:**
- Added save_freq=100 steps (line 152)
- Added total_epochs=3 (line 153)
- Added mixed_precision=fp16 (line 156)
- Added max_grad_norm=1.0 for gradient clipping (line 157)

### Performance Impact
- **Increased throughput**: 2x more rollout samples (8 vs 4) for better GRPO
- **Better GPU utilization**: 85% memory target with larger batch sizes
- **Faster data loading**: 4 workers instead of 2
- **Memory efficient**: Reduced micro batch size with larger mini batch
- **Stable training**: Gradient clipping and mixed precision

### V100 Optimization Strategy
1. Float16 precision (V100 optimized, not bfloat16)
2. Gradient checkpointing enabled (save memory)
3. SDPA attention (no Flash Attention compilation)
4. Larger rollout samples for better policy gradients
5. Balanced batch sizes to avoid OOM

### Configuration File System
Enhanced existing config.yaml with comprehensive GPU-specific tuning:

**config.yaml** (updated):
- Centralized configuration for all training parameters
- Updated with V100-optimized values (batch_size: 8, rollout.n: 8, etc.)
- Added GPU-specific presets (V100, A100, RTX 3090/4090)
- Added memory estimation guide
- Added performance tuning tips
- Added reward function configuration
- Comprehensive comments for each parameter

**train_from_config.py** (new):
- Loads all settings from config.yaml
- Validates configuration and data files
- Builds VERL command from config
- Displays training summary before starting

**Usage**:
```bash
# Use default config.yaml
uv run GRPO/train_from_config.py

# Or specify custom config
uv run GRPO/train_from_config.py --config path/to/custom_config.yaml
```

**Benefits**:
- Single source of truth for all parameters
- No code changes needed for different GPUs
- Easy to version control configurations
- Quick A/B testing of hyperparameters
- Clear documentation of training runs

## 2025-12-09: Initial GRPO Training Platform Setup

Built complete GRPO training platform using VERL framework for Qwen 0.5B model with GSM8K dataset. Platform designed to be extended for news data training on GPU servers.

### Components Created

**Dependencies (pyproject.toml)**
- Added PyTorch, Transformers, Accelerate, Datasets
- Added VLLM, Ray, PEFT, BitsAndBytes for efficient training
- Set numpy<2.0.0 for VERL compatibility
- Added optional dev dependencies for Jupyter

**Data Preparation (GRPO/prepare_gsm8k.py)**
- Downloads GSM8K dataset from HuggingFace
- Formats data for VERL GRPO training with prompt and reward_model fields
- Saves as parquet files in data/gsm8k/
- Includes structured logging following project logging rules

**Training Script (GRPO/train_grpo.py)**
- GRPOTrainer class for managing training pipeline
- Configurable hyperparameters via CLI arguments
- Validates dataset presence before training
- Sets environment variables for optimal GPU performance
- Builds VERL training command with all necessary parameters
- Supports multi-GPU training configuration

**Inference Script (GRPO/inference.py)**
- MathSolver class for model inference
- Supports both single question and batch processing
- Extracts final answers from model responses
- Interactive and CLI modes
- Auto-detects CUDA availability

**Configuration (GRPO/config.yaml)**
- Centralized hyperparameter configuration
- Model, data, training, and GRPO algorithm settings
- Documentation for each parameter
- Default values optimized for Qwen 0.5B on single GPU

**Setup Script (GRPO/setup_verl.sh)**
- Automates VERL installation from GitHub
- Handles numpy downgrade for compatibility
- Applies SDPA patch to avoid Flash Attention compilation delays
- Sets required environment variables

**Documentation (README.md)**
- Complete setup and usage instructions
- Project structure overview
- Training and inference examples
- Common issues and solutions
- Extension guide for news data

### Design Decisions

- **Script-based vs Notebook**: Converted from Colab notebook to Python scripts for better GPU server deployment and version control
- **Modular Architecture**: Separated data preparation, training, and inference for flexibility
- **Configuration Management**: YAML config for easy hyperparameter tuning without code changes
- **Logging Strategy**: Follows project logging rules (DEBUG for HTTP, INFO for pipeline events, removed all print statements)
- **Error Handling**: Validation checks before training, clear error messages
- **Extensibility**: Structure allows easy adaptation to news data by following GSM8K example

### Known Limitations

- VERL must be installed from git (not on PyPI)
- Requires manual SDPA patch for optimal performance
- Numpy 2.x incompatibility requires downgrade
- Designed for Linux GPU servers (Colab compatibility issues noted in original notebook)

### Next Steps

To adapt for news data:
1. Create prepare_news_data.py following prepare_gsm8k.py pattern
2. Define reward model criteria for news quality/coherence
3. Update config.yaml with news-specific parameters
4. Run training pipeline on GPU server
