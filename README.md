<h1 align="center">AraCuisine</h1>

<h3 align="center">Culturally Grounded Instruction Tuning and Benchmarking for Arabic Culinary Reasoning</h3>

<p align="center">
   <a href=""><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/AraCuisine"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=for-the-badge" alt="Hugging Face Dataset"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Paper-EACL%202027-B31B1B?style=for-the-badge" alt="Paper"></a>
  <img src="https://img.shields.io/badge/Questions-1%2C532-4C9A2A?style=for-the-badge" alt="Questions">
  <img src="https://img.shields.io/badge/Language-Arabic-0E7C3A?style=for-the-badge" alt="Language: Arabic">
  <br>
  <img src="https://img.shields.io/badge/IFT%20Pairs-475K%2B-A41E11?style=flat-square" alt="IFT Pairs: 475K+">
  <img src="https://img.shields.io/badge/Recipes-65K-2088FF?style=flat-square" alt="Recipes: 65K">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Models%20Evaluated-5-orange?style=flat-square" alt="Models Evaluated">
</p>

## Table of Contents

- [TL;DR](#tldr)
- [Dataset at a glance](#dataset-at-a-glance)
- [How it's built](#how-its-built)
- [Repository layout](#repository-layout)
- [Evaluation](#evaluation)
- [Instruction fine-tuning](#instruction-fine-tuning)
- [Setup](#setup)
- [Citation](#citation)

## TL;DR

An instruction-tuning corpus and a human-validated benchmark for **Arabic culinary
reasoning**, built from about **65K Arabic recipes** across the Arabian Peninsula, the
Levant, the Nile Valley, and the Maghreb. **AraCuisine-IFT** has **475K+**
instruction–response pairs, and **AraCuisine-Bench** has **1,532** five-choice questions
on factual, procedural, regional, and dietary or religious reasoning.

This repository holds the **evaluation harness and fine-tuning code**.
The released data lives on the Hugging Face Hub at
[`MBZUAI/AraCuisine`](https://huggingface.co/datasets/MBZUAI/AraCuisine):

```python
from datasets import load_dataset

bench = load_dataset("MBZUAI/AraCuisine", "bench_v1", split="test")  # bench_v1 ... bench_v5
ift = load_dataset("MBZUAI/AraCuisine", "ift", split="train")
```

## Dataset at a glance

|            |                                                                                     |
| ---------- | ----------------------------------------------------------------------------------- |
| Questions  | 1,532                                                                               |
| IFT pairs  | 475K+                                                                               |
| Language   | Arabic (`ar`)                                                                       |
| Format     | Single-answer multiple choice (5 options)                                           |
| Versions   | 5 option orders (v1 to v5) to remove gold-answer position bias                      |
| Categories | 7 (Procedural, Ingredients, Explanation, Provenance, Religious, Dietary, Technical) |
| Sources    | ~65K recipes from 5 Arabic culinary platforms                                       |

Every benchmark item is grounded in a source recipe, paired with **plausible hard
distractors**, and verified by **three native annotators** (Cohen's κ = 0.87).
Benchmark recipes are removed from the IFT pool, so the two never overlap.

## How it's built

Both resources come from one **recipe-grounded pipeline**. Each IFT pair stays traceable
to its recipe, its canonical ingredients, and its constraint labels.

| Phase                    | What happens                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Collection**           | Crawl recipes from 5 platforms, drop non-recipe pages and duplicates, extract title, ingredients, steps |
| **Normalization**        | Map ~33K ingredient phrases to ~1K canonical classes, expert-validated (Cohen's κ = 0.90)               |
| **Symbolic constraints** | Ingredient-level rules for diabetic, vegetarian, Islamic, Christian-fasting, and kosher suitability     |
| **IFT synthesis**        | Generate instruction–response pairs for knowledge, procedure, and cultural tasks                        |
| **Enrichment**           | Rewrite pairs for variety while keeping labels and recipe evidence fixed                                |
| **Benchmark curation**   | Build five-choice items, triple annotation, then drop 65 ambiguous items                                |

The pipeline ends in two released resources:

- [`data/benchmark/`](data/benchmark/): **AraCuisine-Bench** in five option orders
  (`aracuisine_bench_option_order_v1.json` to `_v5.json`). Each item holds `item_id`,
  `question`, `options`, `correct_answer`, and `category`. See [Option-order versions](#option-order-versions).
  `aracuisine_bench_manifest.tsv` adds a stable `item_id`, the `doc_id` (position in
  every version file), the 7-way `reasoning_category`, and the `cultural_subset` flag
  for the 302 cultural items.
- [`data/ift/`](data/ift/): **AraCuisine-IFT**, 475,654 `instruction`, `input`, `output`
  records (~1.2 GB) stored as 10 xz-compressed JSONL parts (~100 MB in total), used by
  the code in [`IFT/`](IFT/).
- [`results/`](results/): the 50 prediction logs behind the paper's tables (5 models ×
  base/IFT × 5 versions), the benchmark files they were scored on, and a script that
  rebuilds every table from them.

Dietary and religious labels follow the operational rules in the paper. They are
dataset criteria, not medical or religious advice.

## Repository layout

```
arabic-cuisine/
├── data/
│   ├── benchmark/     # AraCuisine-Bench, 5 option-order versions (v1-v5)
│   └── ift/           # AraCuisine-IFT, 10 xz-compressed JSONL parts
├── results/           # paper prediction logs, evaluation snapshot, score.py
├── hf/                # Hugging Face dataset card and stats (synced by GitHub Action)
├── IFT/               # LoRA fine-tuning, testing, and inference code
├── eval.sh            # lm-evaluation-harness runner (base models + adapters)
└── requirements.txt
```

## Evaluation

Treat each item as a single-answer MCQ: present `question` + `options`, and score each
option by log-likelihood. **Accuracy** is the primary metric. Report the **mean ± std
over the five option-order versions**, ideally broken down by `category`.

### Option-order versions

Models can favour an answer position (for example, always picking `A`). To remove this
**gold-answer position bias**, every item is evaluated in five versions that differ only
in the order of the options:

- `v1` is the base order. `v2` to `v5` rotate the five options by one to four positions,
  so the gold answer sits in a different position in each version (all but one item).
- Questions, option texts, gold answers, and item order are identical across versions.
  Only the letters `A` to `E` and the position of each option change.
- Within each version, gold answers are spread evenly over `A` to `E` (about 306 each).

All results in the paper are the mean ± std over five option-order versions. The
paper was scored on an earlier snapshot of the benchmark, before typo fixes and before
the versions were regenerated as rotations. That snapshot and all 50 prediction logs
are in [`results/`](results/), and `python results/score.py` rebuilds the paper's
tables from them exactly. Use `data/benchmark/` for new evaluations.

**Open models and LoRA adapters** (via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)):

```bash
bash eval.sh        # base models + adapters; results in eval_results/
```

Run the task once per version by pointing `data_files` in the lm-eval task config to
each file in `data/benchmark/`, then average the five accuracies. To get per-category
and cultural-subset scores, join lm-eval's per-sample logs (`--log_samples`) with
`data/benchmark/aracuisine_bench_manifest.tsv` on `doc_id`.

## Instruction fine-tuning

Training uses the paper's configuration (LoRA, 7,437 steps, effective batch 32). See
[`IFT/README.md`](IFT/README.md) for details.

```bash
xz -dk data/ift/*.jsonl.xz      # unpack the IFT parts (~1.2 GB)
cd IFT
./run_sft_train.sh               # trains on ../data/ift/ by default
```

## Setup

```bash
git clone git@github.com:mbzuai-nlp/arabic-cuisine.git
cd arabic-cuisine
pip install -r requirements.txt
```

Set `HF_TOKEN` to download models, and provide API keys for zero-shot API runs via a
`.env` file in the project root.

## Citation

```bibtex
@inproceedings{freihat2027aracuisine,
      title={AraCuisine: Culturally Grounded Instruction Tuning and Benchmarking for Arabic Culinary Reasoning},
      author={Abed Alhakim Freihat and Younes Samih and Mostafa Awad and Mohamed Anwar and Sarfraz Ahmad and Farah Atif and Momina Ahsan and George Ibrahim and Fajri Koto and Preslav Nakov},
      booktitle={Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics},
      year={2027},
}
```
