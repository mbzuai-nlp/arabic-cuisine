#!/bin/bash
# =============================================================================
# Arabic Cuisine Benchmark (ACB) evaluation script
#
# Iterates over 5 fine-tuned ArabicCuisineLab models hosted on HuggingFace Hub,
# evaluating EVERY checkpoint subfolder (3 per repo = 15 runs total) using the
# `arabic_cuisine_benchmark` task registered in our fork of lm-evaluation-harness:
#   https://github.com/Anwarvic/lm-evaluation-harness  (jais branch)
#
# Backend:  HuggingFace transformers (--model hf), single GPU on cuda:0.
# Output:   eval_results/arabic_cuisine_benchmark/<MODEL>/<CHECKPOINT>/
#
# Requires:
#   * lm-eval installed editable from the harness (`pip install -e .[dev]`)
#   * HF_TOKEN exported (the repos are private)
#   * The data file at:
#       /Users/sarfrazahmad/Research/arabic-cuisines/data/final_cuisine_benchmark.json
# =============================================================================

set -uo pipefail

SEED=42
BATCH_SIZE=8                # HF backend; lower if you OOM. set to "auto" for auto-tune.
DEVICE=cuda:0
OVERWRITE=false             # set true to re-run completed evals
TASK="arabic_cuisine_benchmark"

RESULTS_PATH="/Users/sarfrazahmad/Research/arabic-cuisines/eval_results"

# -----------------------------------------------------------------------------
# Models + their 3 checkpoint subfolders inside each HF repo.
# HF Trainer default naming is `checkpoint-<global_step>`. EDIT the step numbers
# below to match what was actually pushed to each repo. To inspect a repo's
# subfolders quickly:
#     huggingface-cli ls --repo-type model ArabicCuisineLab/<repo-name>
# -----------------------------------------------------------------------------
declare -A MODEL_CHECKPOINTS
MODEL_CHECKPOINTS["ArabicCuisineLab/arabic-cuisine-jais-2-8b-chat-sft"]="checkpoint-XXX checkpoint-YYY checkpoint-ZZZ"
MODEL_CHECKPOINTS["ArabicCuisineLab/arabic-cuisine-allam-7b-sft"]="checkpoint-XXX checkpoint-YYY checkpoint-ZZZ"
MODEL_CHECKPOINTS["ArabicCuisineLab/arabic-cuisine-fanar-9b-sft"]="checkpoint-XXX checkpoint-YYY checkpoint-ZZZ"
MODEL_CHECKPOINTS["ArabicCuisineLab/arabic-cuisine-llama-3-8b-sft"]="checkpoint-XXX checkpoint-YYY checkpoint-ZZZ"
MODEL_CHECKPOINTS["ArabicCuisineLab/arabic-cuisine-qwen3-8b-sft"]="checkpoint-XXX checkpoint-YYY checkpoint-ZZZ"

# -----------------------------------------------------------------------------
# Auth: HF token. Either export HF_TOKEN before running, or drop it into
# ~/.cache/huggingface/token (huggingface-cli login). The line below is a
# best-effort fallback; comment out if you manage tokens elsewhere.
# -----------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "\033[33mWARNING:\033[0m HF_TOKEN is not set. Private repos will fail to download."
fi

# -----------------------------------------------------------------------------
# Pretty print plan
# -----------------------------------------------------------------------------
echo "Task:           ${TASK}"
echo "Backend:        hf (transformers)"
echo "Device:         ${DEVICE}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Overwrite:      ${OVERWRITE}"
echo "Results dir:    ${RESULTS_PATH}"
echo
echo "Runs to perform:"
TOTAL=0
for REPO in "${!MODEL_CHECKPOINTS[@]}"; do
    for CKPT in ${MODEL_CHECKPOINTS[$REPO]}; do
        echo "  * ${REPO} :: ${CKPT}"
        TOTAL=$((TOTAL + 1))
    done
done
echo "Total runs:     ${TOTAL}"
echo

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for REPO in "${!MODEL_CHECKPOINTS[@]}"; do
    MODEL_NAME=$(basename "${REPO}")
    for CKPT in ${MODEL_CHECKPOINTS[$REPO]}; do

        OUT_PATH="${RESULTS_PATH}/${TASK}/${MODEL_NAME}/${CKPT}"
        mkdir -p "${OUT_PATH}"

        echo -e "\n\033[1;34m==> ${MODEL_NAME} / ${CKPT}\033[0m"

        # Skip if results already present and not overwriting
        if [ "$OVERWRITE" = false ] && find "$OUT_PATH" -maxdepth 2 -name 'results_*.json' 2>/dev/null | grep -q .; then
            echo "    (results already exist, skipping)"
            continue
        fi

        # Free GPU memory between runs (best-effort, ignore failures)
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

        # All 5 models are instruction-tuned ⇒ apply chat template.
        lm-eval \
            --model hf \
            --model_args "pretrained=${REPO},subfolder=${CKPT},dtype=bfloat16,trust_remote_code=True" \
            --tasks "${TASK}" \
            --batch_size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --apply_chat_template \
            --log_samples \
            --output_path "${OUT_PATH}/results.json" \
            --use_cache "${OUT_PATH}/cache" \
            --seed "${SEED}" \
            --trust_remote_code \
            --verbosity DEBUG

        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo -e "\033[31m    FAILED (exit=$STATUS) — continuing with next checkpoint\033[0m"
        fi
    done
done

echo -e "\n\033[1;32mAll runs completed.\033[0m  Results under: ${RESULTS_PATH}/${TASK}/"
