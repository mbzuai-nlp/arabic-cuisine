# Arabic Cuisine IFT SFT

This repository targets instruction fine-tuning for Arabic cuisine question-answer data. The training, batch testing, and single-sample inference scripts operate on plain instruction-format records.

## Data Format

The SFT scripts expect one of these inputs:

- A directory containing one or more `.jsonl` files.
- A single `.jsonl` file.
- A Hugging Face dataset saved with `load_from_disk`.

Each record should follow this schema:

```json
{
  "instruction": "Explain why this dish is suitable for vegetarians.",
  "input": "",
  "output": "Detailed Arabic answer.",
  "language": "ar"
}
```

## Install

```bash
pip install -r requirements.txt
```

## Train

Train all target models defined in `models.yaml` against the Arabic cuisine data folder:

```bash
python scripts/train.py \
  --config config.yaml \
  --models models.yaml \
  --dataset data
```

Notes:

- If `--dataset` points to JSONL data, the trainer will create a train/test split automatically using `train.eval_split` from `config.yaml`.
- If `--dataset` points to a saved Hugging Face dataset with existing splits, those splits are used directly.
- LoRA adapters are saved under `models/<alias>/`.

## Test

Run batch generation against a held-out split or a JSONL file/folder:

```bash
python scripts/test.py \
  --model_path models/gemma9b \
  --data_path data \
  --output_dir outputs/sft \
  --limit 20
```

If the dataset source does not have a `test` split, the script falls back to the first available split.

Output records are saved as JSON and include:

```json
{
  "id": "0",
  "instruction": "...",
  "input": "...",
  "reference": "...",
  "prediction": "...",
  "model": "gemma9b"
}
```

## Infer

Infer from a direct prompt:

```bash
python scripts/infer.py \
  --model_path models/gemma9b \
  --instruction "Explain why this Arabic dish is suitable for vegetarians." \
  --input_text ""
```

Or infer from an existing dataset sample:

```bash
python scripts/infer.py \
  --model_path models/gemma9b \
  --data_path data \
  --split train \
  --sample_index 0
```

## Configuration

- `config.yaml` controls training hyperparameters, logging, saving, and auto-created eval splits.
- `models.yaml` defines the base models and LoRA overrides for each training target.
- `run_sft_inference.sh` runs `scripts/test.py` sequentially across the configured adapter folders.

## Project Layout

```text
.
├── config.yaml
├── models.yaml
├── data/
├── models/
├── outputs/
├── run_sft_inference.sh
└── scripts/
    ├── infer.py
    ├── test.py
    ├── train.py
    └── utils.py
```
