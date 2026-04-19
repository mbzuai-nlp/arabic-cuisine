#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------
# Multi-GPU SFT Training Launcher
# ---------------------------------
# Set which GPUs to use (e.g. "4,5,6,7" for 4 GPUs, "4,5" for 2)
# GPUs 0,1 are A100s currently occupied; GPUs 4,5 are free A40s; 2,3,6,7 have ~34-36 GB free
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Avoid memory fragmentation OOMs during quantization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Count the number of GPUs
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "[INFO] Using GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port=29500 \
  scripts/train.py \
  --config config.yaml \
  --models models.yaml \
  --dataset data/arabic_cuisine_all_data.jsonl
