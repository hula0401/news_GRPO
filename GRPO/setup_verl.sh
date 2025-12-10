#!/bin/bash
# Setup script for VERL installation on GPU server
# This script installs VERL from source and applies necessary patches

set -e

echo "===== VERL Setup for GRPO Training ====="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install core dependencies first
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

# Apply SDPA patch for faster training
echo ""
echo "Applying SDPA patch for optimized attention..."

# Find VERL installation path
verl_path=$(python3 -c "import verl; import os; print(os.path.dirname(verl.__file__))" 2>/dev/null || echo "")

if [ -z "$verl_path" ]; then
    echo "WARNING: VERL not found. Patch may need to be applied manually."
else
    target_file="$verl_path/workers/fsdp_workers.py"

    if [ -f "$target_file" ]; then
        # Check if already patched
        if grep -q "attn_implementation='sdpa'" "$target_file"; then
            echo "VERL is already patched for SDPA."
        else
            echo "Patching $target_file..."
            # Create backup
            cp "$target_file" "$target_file.backup"

            # Apply patch
            sed -i.tmp 's/actor_module_class.from_pretrained(/actor_module_class.from_pretrained(attn_implementation="sdpa", /g' "$target_file"
            rm "$target_file.tmp"

            echo "Patch applied successfully!"
        fi
    else
        echo "WARNING: Target file not found: $target_file"
        echo "Patch may need to be applied manually."
    fi
fi

# Set environment variables
echo ""
echo "Setting up environment variables..."
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="true"

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Next steps:"
echo "1. Prepare dataset: uv run GRPO/prepare_gsm8k.py"
echo "2. Start training: uv run GRPO/train_grpo.py"
echo ""
