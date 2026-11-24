#!/bin/bash
# =============================================================================
# Arabic Cuisine Benchmark (ACB) evaluation — LoRA / QLoRA aware
#
# What this script does, step by step, for each (repo, checkpoint) pair:
#   1.  Discovers the 3 checkpoint subfolders inside the HF repo via
#       huggingface_hub.list_repo_files (no need to hard-code step numbers).
#   2.  snapshot_downloads ONLY the inference-relevant files for that
#       checkpoint (adapter_config.json, adapter_model.*, chat_template.jinja,
#       tokenizer*) into a local cache. Optimizer/scheduler/RNG files are
#       skipped → ~75% smaller download.
#   3.  Reads `base_model_name_or_path` from adapter_config.json so we don't
#       have to maintain a separate adapter→base mapping.
#   4.  Runs lm-eval with:
#         --model hf
#         --model_args pretrained=<base>,peft=<adapter_dir>,tokenizer=<adapter_dir>,...
#         --apply_chat_template
#       The `tokenizer=<adapter_dir>` makes --apply_chat_template use the
#       model's TRAINED chat template (chat_template.jinja, shipped with each
#       checkpoint), not whatever the base would've supplied.
#   5.  Writes results to:
#         eval_results/arabic_cuisine_benchmark/<MODEL>/<CHECKPOINT>/
#       and skips runs whose results already exist (unless OVERWRITE=true).
#
# Sanity check: if SANITY_CHECK=true, the script first runs an eval on the
# BASE model alone (no --peft) for the first repo's first checkpoint. If that
# accuracy comes out within ~0.5 pts of the corresponding adapter run, the
# adapter is silently NOT being applied — that's the failure mode we want to
# catch up front.
#
# Requirements:
#   * lm-evaluation-harness (jais branch) installed editable: `pip install -e ".[dev]"`
#   * huggingface_hub:   pip install huggingface_hub
#   * peft:              pip install peft
#   * HF_TOKEN exported (the ArabicCuisineLab repos are private)
#   * Data files at: data/benchmark/aracuisine_bench_option_order_v{1..5}.json (one run per version)
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Knobs
# -----------------------------------------------------------------------------
SEED=42
BATCH_SIZE=8                  # drop if you OOM; "auto" if you trust the harness
DEVICE=cuda:0
OVERWRITE=false               # true = re-run even if results exist
EVAL_BASES=true               # true = run base-only eval for every unique base
TASK="arabic_cuisine_benchmark"

PROJECT_DIR="/workspace/arabic-cuisines"
RESULTS_PATH="${PROJECT_DIR}/eval_results"
ADAPTER_CACHE="${PROJECT_DIR}/.adapter_cache"   # snapshot_download lives here

# Repos to evaluate (5 models). Checkpoints inside each repo are auto-discovered.
REPOS=(
    "ArabicCuisineLab/arabic-cuisine-llama-3-8b-sft"
    "ArabicCuisineLab/arabic-cuisine-jais-2-8b-chat-sft"
    "ArabicCuisineLab/arabic-cuisine-allam-7b-sft"
    "ArabicCuisineLab/arabic-cuisine-fanar-9b-sft"
    "ArabicCuisineLab/arabic-cuisine-qwen3-8b-sft"
    "ArabicCuisineLab/arabic-cuisine-gemma-3-12b-it-sft"
)

# Files we need from each checkpoint folder for inference. snapshot_download
# uses these as glob patterns; everything else (optimizer.pt, scheduler.pt,
# trainer_state.json, …) is skipped.
INFERENCE_PATTERNS=(
    "adapter_config.json"
    "adapter_model.safetensors"
    "adapter_model.bin"
    "chat_template.jinja"
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "added_tokens.json"
    "spiece.model"
    "vocab.json"
    "merges.txt"
)

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "\033[31mERROR:\033[0m HF_TOKEN is not set. The 5 repos are private; aborting."
    exit 1
fi

mkdir -p "${ADAPTER_CACHE}" "${RESULTS_PATH}"

# -----------------------------------------------------------------------------
# Helper: discover checkpoint subfolders inside a repo
# Stdout: space-separated list of "checkpoint-<step>" names, sorted by step.
# -----------------------------------------------------------------------------
discover_checkpoints() {
    local repo="$1"
    python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
files = api.list_repo_files("${repo}")
ckpts = sorted(
    {f.split("/")[0] for f in files if f.startswith("checkpoint-") and "/" in f},
    key=lambda s: int(s.split("-")[1]),
)
print(" ".join(ckpts))
PY
}

# -----------------------------------------------------------------------------
# Helper: download just one checkpoint's inference files into ADAPTER_CACHE.
# Stdout: two whitespace-separated tokens — "<local_dir> <base_model_id>"
# -----------------------------------------------------------------------------
fetch_adapter() {
    local repo="$1"
    local ckpt="$2"
    python - <<PY
import json, os, sys
from huggingface_hub import snapshot_download

repo, ckpt = "${repo}", "${ckpt}"
patterns = [f"{ckpt}/{p}" for p in [
    "adapter_config.json","adapter_model.safetensors","adapter_model.bin",
    "chat_template.jinja","tokenizer.json","tokenizer.model","tokenizer_config.json",
    "special_tokens_map.json","added_tokens.json","spiece.model","vocab.json","merges.txt",
]]
local_repo = snapshot_download(
    repo_id=repo,
    allow_patterns=patterns,
    cache_dir="${ADAPTER_CACHE}",
    token=os.environ["HF_TOKEN"],
)
adapter_dir = os.path.join(local_repo, ckpt)
cfg_path = os.path.join(adapter_dir, "adapter_config.json")
if not os.path.isfile(cfg_path):
    print(f"ERROR: adapter_config.json not found at {cfg_path}", file=sys.stderr)
    sys.exit(1)
with open(cfg_path) as fh:
    cfg = json.load(fh)
base = cfg.get("base_model_name_or_path")
if not base:
    print(f"ERROR: base_model_name_or_path missing from {cfg_path}", file=sys.stderr)
    sys.exit(1)
print(f"{adapter_dir} {base}")
PY
}

# -----------------------------------------------------------------------------
# Helper: invoke lm-eval. $1=model_args  $2=output_dir  $3=run_label
# -----------------------------------------------------------------------------
run_lm_eval() {
    local model_args="$1"
    local out_path="$2"
    local label="$3"

    mkdir -p "${out_path}"
    if [ "$OVERWRITE" = false ] && find "$out_path" -maxdepth 3 -name 'results_*.json' 2>/dev/null | grep -q .; then
        echo "    (results already exist, skipping ${label})"
        return 0
    fi

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # NOTE: no --apply_chat_template. Training used the Alpaca-style
    # "Instruction:\n...\n\nResponse:\n..." surface form (scripts/utils.py
    # format_instruction_prompt), NOT chat-template wrapping. The YAML
    # reproduces that surface form so eval matches train.
    lm-eval \
        --model hf \
        --model_args "${model_args}" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --log_samples \
        --output_path "${out_path}/results.json" \
        --use_cache "${out_path}/cache" \
        --seed "${SEED}" \
        --trust_remote_code \
        --verbosity DEBUG
}

# -----------------------------------------------------------------------------
# Discover all checkpoints up front so the run plan is visible before we start.
# -----------------------------------------------------------------------------
echo -e "\n\033[1mDiscovering checkpoints on HuggingFace…\033[0m"
declare -A MODEL_CHECKPOINTS
TOTAL=0
for REPO in "${REPOS[@]}"; do
    echo "  ${REPO}"
    CKPTS=$(discover_checkpoints "${REPO}")
    if [ -z "${CKPTS}" ]; then
        echo -e "    \033[31mNo checkpoint-* subfolders found. Check the repo.\033[0m"
        continue
    fi
    MODEL_CHECKPOINTS[$REPO]="${CKPTS}"
    for c in ${CKPTS}; do
        echo "    - ${c}"
        TOTAL=$((TOTAL + 1))
    done
done
echo "Total LoRA runs planned: ${TOTAL}"
[ "$EVAL_BASES" = true ] && echo "Plus base-only eval for every unique base (up to ${#REPOS[@]})."

# -----------------------------------------------------------------------------
# Base-only evaluations: evaluate every unique base model (read from each
# adapter's adapter_config.json) with NO --peft. Deduplicated across repos so
# if two adapters share a base, that base is only run once. These give you a
# reference number per model family to compare adapter scores against.
# -----------------------------------------------------------------------------
if [ "$EVAL_BASES" = true ] && [ ${#MODEL_CHECKPOINTS[@]} -gt 0 ]; then
    echo -e "\n\033[1;36m=== Base-only evaluations (one per unique base) ===\033[0m"
    declare -A SEEN_BASES
    for REPO in "${REPOS[@]}"; do
        [ -z "${MODEL_CHECKPOINTS[$REPO]:-}" ] && continue
        FIRST_CKPT=$(echo "${MODEL_CHECKPOINTS[$REPO]}" | awk '{print $1}')

        FETCH_OUT=$(fetch_adapter "${REPO}" "${FIRST_CKPT}") || {
            echo -e "    \033[31mfailed to fetch adapter for ${REPO}; skipping its base\033[0m"
            continue
        }
        read ADAPTER_DIR BASE_MODEL <<< "${FETCH_OUT}"

        # Dedup: same base shared between adapters → eval it once
        if [ -n "${SEEN_BASES[$BASE_MODEL]:-}" ]; then
            echo "    (${BASE_MODEL} already covered, skipping)"
            continue
        fi
        SEEN_BASES[$BASE_MODEL]=1

        BASE_OUT="${RESULTS_PATH}/${TASK}/_base_only/$(basename ${BASE_MODEL})"
        echo -e "\n  -> base = \033[1;34m${BASE_MODEL}\033[0m  (from ${REPO})"
        run_lm_eval \
            "pretrained=${BASE_MODEL},dtype=bfloat16,trust_remote_code=True" \
            "${BASE_OUT}" \
            "base-only ${BASE_MODEL}"
    done
fi

# -----------------------------------------------------------------------------
# Main loop: every (repo, checkpoint) gets one LoRA-aware eval.
# -----------------------------------------------------------------------------
for REPO in "${REPOS[@]}"; do
    [ -z "${MODEL_CHECKPOINTS[$REPO]:-}" ] && continue
    MODEL_NAME=$(basename "${REPO}")
    for CKPT in ${MODEL_CHECKPOINTS[$REPO]}; do

        OUT_PATH="${RESULTS_PATH}/${TASK}/${MODEL_NAME}/${CKPT}"
        echo -e "\n\033[1;34m==> ${MODEL_NAME} / ${CKPT}\033[0m"

        if [ "$OVERWRITE" = false ] && find "$OUT_PATH" -maxdepth 3 -name 'results_*.json' 2>/dev/null | grep -q .; then
            echo "    (results already exist, skipping)"
            continue
        fi

        # 1. Pull just this checkpoint's inference files (cached on subsequent runs).
        FETCH_OUT=$(fetch_adapter "${REPO}" "${CKPT}") || {
            echo -e "    \033[31mfailed to fetch adapter; skipping\033[0m"
            continue
        }
        read ADAPTER_DIR BASE_MODEL <<< "${FETCH_OUT}"
        echo "    adapter -> ${ADAPTER_DIR}"
        echo "    base    -> ${BASE_MODEL}"

        # 2. Build model_args: base loaded fresh, LoRA applied, tokenizer (and
        #    chat template) taken from the adapter folder.
        # MODEL_ARGS="pretrained=${BASE_MODEL},peft=${ADAPTER_DIR},tokenizer=${ADAPTER_DIR},dtype=bfloat16,trust_remote_code=True"
        MODEL_ARGS="pretrained=${BASE_MODEL},peft=${ADAPTER_DIR},dtype=bfloat16,trust_remote_code=True"

        run_lm_eval "${MODEL_ARGS}" "${OUT_PATH}" "${MODEL_NAME}/${CKPT}"
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo -e "\033[31m    FAILED (exit=$STATUS) — continuing\033[0m"
        fi
    done
done

echo -e "\n\033[1;32mAll runs done.\033[0m  Results: ${RESULTS_PATH}/${TASK}/"
if [ "$EVAL_BASES" = true ]; then
    echo "Per-base accuracies live under _base_only/<base_name>/ — compare each"
    echo "model family's adapter runs against its corresponding base. If an"
    echo "adapter run is within ~0.5 pts of its base, the adapter isn't being applied."
fi
