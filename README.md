<h1 align="center">AraCuisine</h1>

<h3 align="center">Culturally Grounded Instruction Tuning and Benchmarking for Arabic Culinary Reasoning</h3>

<p align="center">
   <a href=""><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/AraCuisine-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-FFD21E?style=for-the-badge" alt="Hugging Face Benchmark"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/AraCuisine-IFT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IFT%20Data-FFD21E?style=for-the-badge" alt="Hugging Face IFT Data"></a>
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
[`MBZUAI/AraCuisine-Bench`](https://huggingface.co/datasets/MBZUAI/AraCuisine-Bench) and
[`MBZUAI/AraCuisine-IFT`](https://huggingface.co/datasets/MBZUAI/AraCuisine-IFT):

```python
from datasets import load_dataset

bench = load_dataset("MBZUAI/AraCuisine-Bench", split="test")
ift = load_dataset("MBZUAI/AraCuisine-IFT", split="train")
```

## Dataset at a glance

|            |                                                                                     |
| ---------- | ----------------------------------------------------------------------------------- |
| Questions  | 1,532                                                                               |
| IFT pairs  | 475K+                                                                               |
| Language   | Arabic (`ar`)                                                                       |
| Format     | Single-answer multiple choice (5 options)                                           |
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

- **AraCuisine-Bench**: the evaluation set, shipped here as
  [`data/final_cuisine_benchmark.json`](data/final_cuisine_benchmark.json). Each item holds
  `question`, `options`, `correct_answer`, and `category`.
- **AraCuisine-IFT**: the training set of `instruction`, `input`, `output` records, used
  by the code in [`IFT/`](IFT/).

Dietary and religious labels follow the operational rules in the paper. They are
dataset criteria, not medical or religious advice.

## Repository layout

```
arabic-cuisines/
├── data/              # AraCuisine-Bench
├── IFT/               # LoRA fine-tuning, testing, and inference code
├── eval.sh            # lm-evaluation-harness runner (base models + adapters)
└── requirements.txt
```

## Evaluation

Treat each item as a single-answer MCQ: present `question` + `options`, and score each
option by log-likelihood. **Accuracy** is the primary metric. We average it over five
option orders and recommend reporting it broken down by `category`.

**Open models and LoRA adapters** (via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)):

```bash
bash eval.sh        # base models + adapters; results in eval_results/
```

## Instruction fine-tuning

Training uses the paper's configuration (LoRA, 7,437 steps, effective batch 32). See
[`IFT/README.md`](IFT/README.md) for details.

```bash
cd IFT
DATASET=data/aracuisine_ift.jsonl ./run_sft_train.sh
```

## Setup

```bash
git clone git@github.com:mbzuai-nlp/arabic-cuisines.git
cd arabic-cuisines
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
