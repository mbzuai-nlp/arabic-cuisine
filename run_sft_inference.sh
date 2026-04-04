#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------
# Force usage of CUDA device 0 ONLY
# ---------------------------------
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------
# Configuration
# ---------------------------------
PYTHON_BIN=python
TEST_SCRIPT="dialect_translation/scripts/sft/test.py"
DATA_PATH="dialect_translation/data/test.jsonl"
OUTPUT_DIR="dialect_translation/output/sft"

# Explicit model list (recommended)
MODELS=(
  "dialect_translation/models/gemma9b"
  "dialect_translation/models/qwen3_8b"
  "dialect_translation/models/llama33_8b"
)

# Optional sample limit (comment out for full run)
LIMIT=5
# LIMIT=""

# ---------------------------------
# Run sequentially
# ---------------------------------
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Starting sequential SFT evaluation on GPU 0"
echo

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_PATH")

  echo "================================================"
  echo "[INFO] Evaluating model: ${MODEL_NAME}"
  echo "================================================"

  if [[ -n "${LIMIT}" ]]; then
    $PYTHON_BIN "$TEST_SCRIPT" \
      --data_path "$DATA_PATH" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --limit "$LIMIT"
  else
    $PYTHON_BIN "$TEST_SCRIPT" \
      --data_path "$DATA_PATH" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR"
  fi

  echo "[INFO] Finished model: ${MODEL_NAME}"
  echo
done

echo "[DONE] All models evaluated sequentially on CUDA device 0"
