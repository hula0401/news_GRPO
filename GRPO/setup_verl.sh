#!/bin/bash
# Setup script for VERL installation on GPU server
# This script installs VERL from source with SDPA attention

set -e

echo "===== VERL Setup for GRPO Training ====="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "CUDA version: $cuda_version"
else
    echo "WARNING: nvcc not found. CUDA is required for GPU training."
    echo "Please ensure CUDA is installed and nvcc is in PATH."
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
uv pip install torch transformers accelerate datasets vllm peft bitsandbytes

# Install VERL from source
echo ""
echo "Installing VERL from GitHub..."
uv pip install git+https://github.com/volcengine/verl.git

# Downgrade numpy for compatibility
echo ""
echo "Ensuring numpy compatibility..."
uv pip install 'numpy<2.0.0'

# Set environment variables
echo ""
echo "Setting up environment variables..."
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="true"

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Configuration:"
echo "- Using SDPA (Scaled Dot Product Attention) for efficient attention"
echo "- No Flash Attention compilation required"
echo ""
echo "Next steps:"
echo "1. Prepare dataset: uv run GRPO/prepare_gsm8k.py"
echo "2. Start training: uv run GRPO/train_grpo.py"
echo ""
