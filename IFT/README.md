# AraCuisine-IFT: Instruction Fine-Tuning

This directory contains the code used to fine-tune models on **AraCuisine-IFT**, the Arabic culinary instruction-tuning corpus introduced in the paper. All commands below are run from inside `IFT/`.

AraCuisine-IFT has over 475K Arabic instruction–response pairs built from about 65K recipes. Recipes that match or closely resemble a benchmark recipe were removed before any pair was generated, so the training data does not overlap with AraCuisine-Bench.

## Contents

```text
IFT/
├── config.yaml              # training hyperparameters
├── models.yaml              # base models and per-model LoRA settings
├── run_sft_train.sh         # multi-GPU training launcher (torchrun)
├── run_sft_inference.sh     # batch generation over several adapters
└── scripts/
    ├── train.py             # LoRA supervised fine-tuning
    ├── test.py              # batch generation on a held-out split
    ├── infer.py             # single-prompt inference
    ├── zero_shot.py         # generation with a base model (no adapter)
    └── utils.py             # data loading and prompt formatting
```

## Install

```bash
pip install -r ../requirements.txt
```

## Data

AraCuisine-IFT ships in [`../data/ift/`](../data/ift/) as 10 xz-compressed JSONL parts. Unpack them once from the repository root:

```bash
xz -dk data/ift/*.jsonl.xz
```

This writes 10 `.jsonl` files (~1.2 GB, 475,654 records). The training script reads every `.jsonl` file in the folder, so no merging is needed.

The scripts accept any of the following:

- a directory with one or more `.jsonl` files
- a single `.jsonl` file
- a Hugging Face dataset saved with `save_to_disk`

Each record follows this schema:

```json
{
  "instruction": "Why is this dish suitable for vegetarians?",
  "input": "",
  "output": "Detailed Arabic answer grounded in the recipe.",
  "language": "ar"
}
```

## Train

Train every target listed under `train_targets` in `models.yaml`:

```bash
./run_sft_train.sh                              # uses ../data/ift
DATASET=/path/to/other.jsonl ./run_sft_train.sh  # or any other data
```

By default the launcher uses all 8 GPUs. Set `CUDA_VISIBLE_DEVICES` to use fewer. To run on a single process:

```bash
python scripts/train.py \
  --config config.yaml \
  --models models.yaml \
  --dataset ../data/ift
```

Notes:

- `config.yaml` and `models.yaml` match the paper setup below. All five models share one LoRA configuration, and no validation split is used (`train.eval_split: 0.0`).
- `train.prompt_format: chat_template` formats each pair with the model's own chat template. The format is saved in `run_meta.json`, and `test.py` and `infer.py` reuse it.
- If `--dataset` is a saved Hugging Face dataset with existing splits, those splits are used as is.
- LoRA adapters are written to `models/<alias>/`.

### Paper configuration

The models in the paper (Table 14) were trained with these shared settings:

| Setting | Value | Setting | Value |
|---|---|---|---|
| Max steps | 7,437 | Optimizer | paged AdamW 8-bit |
| Batch size per GPU | 1 | Learning rate | 1.0e-4 |
| Gradient accumulation | 4 | Scheduler | cosine |
| GPUs | 8 | Warmup ratio | 0.03 |
| Effective batch size | 32 | Weight decay | 0.0 |
| Validation split | none | Max grad norm | 1.0 |
| LoRA rank / alpha | 16 / 32 | LoRA dropout | 0.05 |
| 4-bit loading | enabled | Precision | bfloat16 |
| Gradient checkpointing | enabled | Attention | SDPA |
| Packing | disabled | Checkpoint saving | every 500 steps |
| Random seed | 42 | Prompt format | model chat template |

LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

Base models:

| Alias | Hugging Face ID |
|---|---|
| `allam_7b` | `humain-ai/ALLaM-7B-Instruct-preview` |
| `gemma12b` | `google/gemma-3-12b-it` |
| `fanar_9b` | `QCRI/Fanar-1-9B-Instruct` |
| `llama33_8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen3_8b` | `Qwen/Qwen3-8B` |

## Test

Run batch generation over a held-out split or a JSONL file or folder:

```bash
python scripts/test.py \
  --model_path models/gemma12b \
  --data_path data \
  --output_dir outputs/sft \
  --limit 20
```

If the data has no `test` split, the first available split is used. Each output record looks like:

```json
{
  "id": "0",
  "instruction": "...",
  "input": "...",
  "reference": "...",
  "prediction": "...",
  "model": "gemma12b"
}
```

To run all adapters listed in `run_sft_inference.sh` one after another on GPU 0:

```bash
./run_sft_inference.sh
```

## Inference

From a direct prompt:

```bash
python scripts/infer.py \
  --model_path models/gemma12b \
  --instruction "Why is this Arabic dish suitable for vegetarians?" \
  --input_text ""
```

From a dataset sample:

```bash
python scripts/infer.py \
  --model_path models/gemma12b \
  --data_path data \
  --split train \
  --sample_index 0
```

## Benchmark evaluation

Fine-tuned adapters are scored on AraCuisine-Bench with `lm-evaluation-harness`. See [Evaluation](../README.md#evaluation) in the main README.
