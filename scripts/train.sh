#!/bin/bash
# Training Launch Script for LLaMA-Factory
# Run this after setup and data upload

set -e

PROJECT_DIR="/workspace/nmt-project"
LLAMA_FACTORY_DIR="/workspace/LLaMA-Factory"
CONFIG_FILE="${PROJECT_DIR}/configs/train_qlora.yaml"

echo "=========================================="
echo "Starting NMT Fine-Tuning"
echo "=========================================="

# Verify files exist
echo "Checking required files..."
if [ ! -f "${PROJECT_DIR}/data/train.jsonl" ]; then
    echo "ERROR: train.jsonl not found!"
    echo "Upload it to ${PROJECT_DIR}/data/"
    exit 1
fi

# Copy dataset info to LLaMA-Factory
echo "Copying dataset config..."
cp ${PROJECT_DIR}/configs/dataset_info.json ${LLAMA_FACTORY_DIR}/data/

# Update config to use correct paths
echo "Updating config paths..."
sed -i "s|dataset_dir: ./data|dataset_dir: ${PROJECT_DIR}/data|g" ${CONFIG_FILE}
sed -i "s|output_dir: ./outputs|output_dir: ${PROJECT_DIR}/outputs|g" ${CONFIG_FILE}

# Show training config
echo ""
echo "Training Configuration:"
echo "----------------------"
cat ${CONFIG_FILE}
echo ""

# Start training
echo "Starting training..."
cd ${LLAMA_FACTORY_DIR}

# Use accelerate for multi-GPU or single GPU
llamafactory-cli train ${CONFIG_FILE}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: ${PROJECT_DIR}/outputs/mistral-nmt-qlora"
echo ""
echo "Next: Run evaluation with the fine-tuned model"
