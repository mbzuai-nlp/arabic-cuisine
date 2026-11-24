---
license: cc-by-nc-4.0
language:
  - ar
pretty_name: AraCuisine-IFT
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
  - question-answering
tags:
  - arabic
  - cuisine
  - culture
  - instruction-tuning
  - sft
configs:
  - config_name: default
    data_files:
      - split: train
        path: aracuisine_ift.jsonl
---

<h1 align="center">AraCuisine-IFT: Arabic Culinary Instruction-Tuning Data</h1>

<p align="center">
  <b>Abed Alhakim Freihat</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Younes Samih</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Mostafa Awad</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Mohamed Anwar</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Sarfraz Ahmad</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Farah Atif</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Momina Ahsan</b><sup>1</sup> &nbsp;·&nbsp;
  <b>George Ibrahim</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Fajri Koto</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Preslav Nakov</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) &nbsp;·&nbsp; <sup>2</sup> IBM Research AI
</p>

<p align="center">
  <a href="https://github.com/mbzuai-nlp/arabic-cuisine"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/Paper-EACL%202027-B31B1B" alt="Paper: EACL 2027">
  <a href="https://huggingface.co/datasets/MBZUAI/AraCuisine-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Companion-AraCuisine--Bench-FFD21E" alt="AraCuisine-Bench"></a>
</p>

---

**AraCuisine-IFT** is an Arabic instruction-tuning corpus of **475,654**
instruction–response pairs built from about **65K Arabic recipes**. It covers
culinary knowledge and execution (ingredients, steps, techniques, times, and
temperatures) and cultural alignment (regional provenance, dietary and health
suitability, and religious compatibility).

Each pair is generated from structured recipe evidence and then rewritten for
natural, varied Arabic while keeping its label and evidence fixed. Recipes in
[AraCuisine-Bench](https://huggingface.co/datasets/MBZUAI/AraCuisine-Bench) and their
near-duplicates were removed before generation, so the two never overlap.

## Dataset at a glance

|              |                                                      |
| ------------ | ---------------------------------------------------- |
| Pairs        | 475,654                                              |
| Language     | Arabic (`ar`)                                      |
| Format       | Instruction–response (Alpaca-style fields)           |
| Source       | 64,987 recipes from 5 Arabic culinary platforms      |
| Avg. length  | ~19 words per instruction, ~197 words per response   |
| Files        | `aracuisine_ift.jsonl` (~1.2 GB), `stats.json`        |

## Schema

Each record has exactly these fields:

```json
{
  "instruction": "ما هي المكونات اللازمة لإعداد البقسماط؟",
  "input": "",
  "output": "تتألف مكونات إعداد البقسماط، بكل بساطة، من العناصر التالية: ...",
  "language": "ar"
}
```

| Field         | Description                                     |
| ------------- | ----------------------------------------------- |
| `instruction` | User request in Arabic                          |
| `input`       | Optional extra context (empty in this release)  |
| `output`      | Target response in Arabic, often in Markdown    |
| `language`    | Always `ar`                                    |

## How it's built

| Stage                    | What happens                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Recipe cleaning**      | Extract clean titles, ingredients, steps, and metadata from crawled recipes                       |
| **Normalization**        | Map ~33K ingredient phrases to ~1K canonical classes, expert-validated (Cohen's κ = 0.90)        |
| **Symbolic constraints** | Label diabetic, vegetarian, Islamic, Christian-fasting, and kosher suitability from ingredients |
| **IFT synthesis**        | Generate pairs for ingredients, procedures, provenance, suitability, and recipe similarity       |
| **Enrichment**           | Rewrite instructions and responses for variety while keeping labels and evidence fixed           |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("MBZUAI/AraCuisine-IFT", split="train")
ex = ds[0]
print(ex["instruction"])
print(ex["output"])
```

### Fine-tuning

In the paper, five models (ALLaM-7B, Gemma-3-12B, Fanar-1-9B, Llama-3-8B, Qwen3-8B)
were fine-tuned with LoRA (rank 16, 7,437 steps, effective batch 32) using each
model's chat template. Accuracy on AraCuisine-Bench rose for all five, by 2.43 to
9.22 points. Training code and the full configuration are in the
[GitHub repository](https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/IFT).

## Sources

Recipes were collected from five public Arabic culinary platforms: Atyab Tabkha
(~30K), Just Food (~16K), Mawdoo3 (~11.5K), Sayidaty Kitchen (~6K), and WebTeb
(~1.5K), for 64,987 recipes in total. Only derived data is released, not raw web pages.

## Limitations

- Dietary, health, and religious labels follow operational ingredient-level rules
  defined in the paper. They are dataset criteria, not medical or religious advice.
  Models trained on this data should not be used for such advice without expert review.
- Responses were rewritten with a language model under consistency checks. Rare
  wording errors may remain.
- Coverage reflects the five source platforms. Rare dishes, local names, and
  community-specific practices may be underrepresented.

## Citation

```bibtex
@inproceedings{freihat2027aracuisine,
      title={AraCuisine: Culturally Grounded Instruction Tuning and Benchmarking for Arabic Culinary Reasoning},
      author={Abed Alhakim Freihat and Younes Samih and Mostafa Awad and Mohamed Anwar and Sarfraz Ahmad and Farah Atif and Momina Ahsan and George Ibrahim and Fajri Koto and Preslav Nakov},
      booktitle={Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics},
      year={2027},
}
```

## License

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Users should also respect the
terms of the original recipe sources.
