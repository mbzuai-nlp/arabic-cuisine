#!/usr/bin/env bash
set -euo pipefail

# Run from the IFT/ directory regardless of where the script is called from
cd "$(dirname "$0")"

# ---------------------------------
# Force usage of CUDA device 0 ONLY
# ---------------------------------
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------
# Configuration
# ---------------------------------
PYTHON_BIN=python
TEST_SCRIPT="scripts/test.py"
DATA_PATH="${DATA_PATH:-../data/benchmark/aracuisine_bench_option_order_v1.json}"
OUTPUT_DIR="outputs/sft"

# Explicit model list (recommended)
MODELS=(
  "models/allam_7b"
  "models/gemma12b"
  "models/fanar_9b"
  "models/llama33_8b"
  "models/qwen3_8b"
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
