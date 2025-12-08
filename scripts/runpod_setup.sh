#!/bin/bash
# RunPod Setup Script for LLaMA-Factory Fine-Tuning
# Run this after SSH into your RunPod A100 instance

set -e

echo "=========================================="
echo "RunPod Setup for NMT Fine-Tuning"
echo "=========================================="

# 1. Install LLaMA-Factory
echo "[1/5] Installing LLaMA-Factory..."
cd /workspace
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --quiet

# 2. Login to Hugging Face (required for Mistral)
echo "[2/5] Hugging Face Login..."
echo "You need a HuggingFace token with access to Mistral models."
echo "Get one at: https://huggingface.co/settings/tokens"
huggingface-cli login

# 3. Create project directory
echo "[3/5] Setting up project..."
mkdir -p /workspace/nmt-project/data
mkdir -p /workspace/nmt-project/configs
mkdir -p /workspace/nmt-project/outputs

# 4. Download training data
echo "[4/5] Downloading training data..."
echo "Please upload your files or use:"
echo "  - gdown (Google Drive)"
echo "  - wget (direct URL)"
echo "  - rsync/scp (from local machine)"
echo ""
echo "Required files:"
echo "  /workspace/nmt-project/data/train.jsonl"
echo "  /workspace/nmt-project/configs/dataset_info.json"
echo "  /workspace/nmt-project/configs/train_qlora.yaml"

# 5. Verify GPU
echo "[5/5] Verifying GPU..."
nvidia-smi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your data files to /workspace/nmt-project/"
echo "2. Copy dataset_info.json to /workspace/LLaMA-Factory/data/"
echo "3. Run training with:"
echo "   cd /workspace/LLaMA-Factory"
echo "   llamafactory-cli train /workspace/nmt-project/configs/train_qlora.yaml"
echo ""
