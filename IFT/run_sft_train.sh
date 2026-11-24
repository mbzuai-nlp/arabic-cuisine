#!/usr/bin/env bash
set -euo pipefail

# Run from the IFT/ directory regardless of where the script is called from
cd "$(dirname "$0")"

# Path to the AraCuisine-IFT training data (JSONL file, folder of JSONL files,
# or a saved Hugging Face dataset). Override with DATASET=... ./run_sft_train.sh
DATASET="${DATASET:-../data/ift}"

# ---------------------------------
# Multi-GPU SFT Training Launcher
# ---------------------------------
# Default: use all 8 GPUs. Override at runtime if needed:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sft_train.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Avoid memory fragmentation OOMs during quantization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL / threading hygiene for stable multi-GPU runs
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# Tokenizers parallelism warning suppression (HF emits a noisy fork warning
# when DataLoader workers spawn after tokenizer use)
export TOKENIZERS_PARALLELISM=false

# Count the number of GPUs
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "[INFO] Using GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port=29500 \
  scripts/train.py \
  --config config.yaml \
  --models models.yaml \
  --dataset "${DATASET}"
