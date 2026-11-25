---
license: cc-by-nc-4.0
language:
  - ar
pretty_name: AraCuisine
annotations_creators:
  - expert-generated
language_creators:
  - found
  - machine-generated
multilinguality:
  - monolingual
source_datasets:
  - original
size_categories:
  - 1K<n<10K
  - 100K<n<1M
task_categories:
  - question-answering
  - multiple-choice
  - text-generation
task_ids:
  - multiple-choice-qa
  - open-domain-qa
tags:
  - arabic
  - cuisine
  - culture
  - benchmark
  - instruction-tuning
  - multiple-choice
configs:
  - config_name: bench_v1
    default: true
    data_files:
      - split: test
        path: bench/aracuisine_bench_option_order_v1.json
  - config_name: bench_v2
    data_files:
      - split: test
        path: bench/aracuisine_bench_option_order_v2.json
  - config_name: bench_v3
    data_files:
      - split: test
        path: bench/aracuisine_bench_option_order_v3.json
  - config_name: bench_v4
    data_files:
      - split: test
        path: bench/aracuisine_bench_option_order_v4.json
  - config_name: bench_v5
    data_files:
      - split: test
        path: bench/aracuisine_bench_option_order_v5.json
  # AraCuisine-IFT: re-enable once uploaded
  # - config_name: ift
  #   data_files:
  #     - split: train
  #       path: ift/aracuisine_ift.parquet
---

<h1 align="center">AraCuisine: Culturally Grounded Instruction Tuning and Benchmarking for Arabic Culinary Reasoning</h1>

<p align="center">
  <b>Abed Alhakim Freihat</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Younes Samih</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Mostafa Awad</b><sup>1</sup>
  <br>
  <b>Mohamed Anwar</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Sarfraz Ahmad</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Farah Atif</b><sup>1</sup>
  <br>
  <b>Momina Ahsan</b><sup>1</sup> &nbsp;·&nbsp;
  <b>George Ibrahim</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Fajri Koto</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Preslav Nakov</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) &nbsp;·&nbsp; <sup>2</sup> IBM Research AI
</p>

<p align="center">
  <a href="#citation"><img src="https://img.shields.io/badge/Paper-EACL%202027-B31B1B" alt="Paper: EACL 2027" style="display:inline-block"></a>&nbsp;<a href="https://github.com/mbzuai-nlp/arabic-cuisine"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub" style="display:inline-block"></a>
</p>

---

**AraCuisine** is a resource for Arabic culinary reasoning, built from about **65K
Arabic recipes** across the Arabian Peninsula, the Levant, the Nile Valley, and the
Maghreb. It has two parts:

- **AraCuisine-Bench**: **1,532** human-validated five-choice questions on factual,
  procedural, regional, and dietary or religious reasoning.
- **AraCuisine-IFT**: **475,654** Arabic instruction–response pairs for fine-tuning.

Benchmark recipes and their near-duplicates were removed before any IFT pair was
generated, so the two parts never overlap.

## Dataset at a glance

|                 | AraCuisine-Bench                               | AraCuisine-IFT                                 |
| --------------- | ---------------------------------------------- | ---------------------------------------------- |
| Size            | 1,532 questions                                | 475,654 pairs                                  |
| Language        | Arabic (`ar`)                                  | Arabic (`ar`)                                  |
| Format          | Single-answer multiple choice (5 options)      | Instruction–response                           |
| Configs         | `bench_v1` to `bench_v5` (5 option orders)     | `ift` (coming soon)                            |
| Split           | `test`                                         | `train`                                        |
| Validation      | 3 native annotators, Cohen's κ = 0.87          | Expert-validated ingredient layer, κ = 0.90    |
| Where           | `bench/` in this repository                    | [`data/ift/` on GitHub](https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/data/ift) (HF soon) |

## Usage

```python
from datasets import load_dataset

# Benchmark (one config per option order)
bench = load_dataset("MBZUAI/AraCuisine", "bench_v1", split="test")

# Instruction-tuning data: coming soon here, available now on GitHub
# https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/data/ift
```

## AraCuisine-Bench

Each item is grounded in a source recipe and paired with plausible distractors drawn
from related ingredients, steps, regions, or constraints.

```json
{
  "item_id": "aracuisine-1230",
  "question": "تعتبر وصفة الحريرة من الوصفات الشهيرة في",
  "options": ["A. المطبخ الكويتي", "B. المطبخ المغربي", "C. المطبخ المصري", "D. المطبخ اللبناني", "E. المطبخ الأردني"],
  "correct_answer": "B. المطبخ المغربي",
  "category": "country"
}
```

| Field            | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `item_id`        | Stable identifier, same across all five versions     |
| `question`       | Question stem in Arabic                              |
| `options`        | Five answer strings, each prefixed with `A.` to `E.` |
| `correct_answer` | The gold option, identical to one entry in `options` |
| `category`       | Fine-grained label (see below)                       |

### Option-order versions

Models can favour an answer position, for example always picking `A`. To remove
this **gold-answer position bias**, the benchmark ships in five versions that differ
only in the order of the options:

- `bench_v1` is the base order. `bench_v2` to `bench_v5` rotate the five options by
  one to four positions, so the gold answer sits in a different position in each
  version (all but one item).
- Questions, option texts, gold answers, and item order are identical across versions.
- Within each version, gold answers are spread evenly over `A` to `E` (about 306 each).

All results in the paper are the **mean ± std accuracy over five option-order
versions**. The paper was scored on an earlier snapshot of the benchmark, before typo
fixes and before the versions were regenerated as rotations. That snapshot, all 50
prediction logs, and a script that rebuilds the paper's tables exactly are in the
[GitHub repository](https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/results).

### Category distribution

| Reasoning category                  | Fine-grained labels                                                        | Questions |
| ----------------------------------- | -------------------------------------------------------------------------- | --------- |
| Procedural Execution                | `steps`                                                                    | 682       |
| Ingredient Analysis                 | `ingredients`                                                              | 356       |
| Culinary Explanation                | `explanation`                                                              | 145       |
| Geographic Provenance               | `country`, `region`                                                        | 119       |
| Religious Compatibility             | `(not_)suitable_christians`, `not_suitable_moslems`, `not_suitable_jews`   | 102       |
| Dietary/Health/Lifestyle Compliance | `(not_)suitable_diabetics`, `(not_)suitable_vegetarians`                   | 81        |
| Technical Parameters                | `time`, `temperature`                                                      | 47        |

Per-label counts and gold-answer position counts are in [`bench/stats.json`](bench/stats.json).

### Manifest

[`bench/aracuisine_bench_manifest.tsv`](bench/aracuisine_bench_manifest.tsv) has one row
per item, in the same order as the five version files:

| Column               | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `item_id`            | Stable identifier (`aracuisine-0000` to `aracuisine-1531`)                     |
| `doc_id`             | Position of the item in every version file, as logged by lm-evaluation-harness |
| `original_category`  | Fine-grained label, same as `category` in the data                              |
| `reasoning_category` | One of the 7 reasoning categories used in the paper                             |
| `cultural_subset`    | `true` for the 302 Geographic, Religious, and Dietary/Health/Lifestyle items    |

### Evaluation

Treat each item as a single-answer MCQ: present `question` and `options`, score
each option, and compare the selected option against `correct_answer`. Accuracy is
the primary metric. Run all five versions and report the **mean ± std**, ideally
broken down by `reasoning_category` from the manifest.

```python
versions = [f"bench_v{i}" for i in range(1, 6)]
data = {v: load_dataset("MBZUAI/AraCuisine", v, split="test") for v in versions}

manifest = load_dataset(
    "csv",
    data_files="hf://datasets/MBZUAI/AraCuisine/bench/aracuisine_bench_manifest.tsv",
    delimiter="\t",
    split="train",
)
```

In the paper, options are scored by log-likelihood with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## AraCuisine-IFT

> **Coming soon to Hugging Face.** The instruction-tuning data will be added to this
> repository shortly. Until then, it is available in the
> [GitHub repository](https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/data/ift)
> as 10 xz-compressed JSONL parts. Unpack them with `xz -dk data/ift/*.jsonl.xz`.

Each pair is generated from structured recipe evidence, then rewritten for natural,
varied Arabic while keeping its label and evidence fixed. Tasks cover culinary
knowledge and execution (ingredients, steps, techniques, times, temperatures) and
cultural alignment (provenance, dietary and health suitability, religious compatibility).

```json
{
  "instruction": "ما هي المكونات اللازمة لإعداد البقسماط؟",
  "input": "",
  "output": "تتألف مكونات إعداد البقسماط، بكل بساطة، من العناصر التالية: ...",
  "language": "ar"
}
```

| Field         | Description                                    |
| ------------- | ---------------------------------------------- |
| `instruction` | User request in Arabic                         |
| `input`       | Optional extra context (empty in this release) |
| `output`      | Target response in Arabic, often in Markdown   |
| `language`    | Always `ar`                                    |

Instructions average about 19 words and responses about 197 words. See
[`ift/stats.json`](ift/stats.json).

### How it's built

| Stage                    | What happens                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Recipe cleaning**      | Extract clean titles, ingredients, steps, and metadata from crawled recipes                       |
| **Normalization**        | Map ~33K ingredient phrases to ~1K canonical classes, expert-validated (Cohen's κ = 0.90)        |
| **Symbolic constraints** | Label diabetic, vegetarian, Islamic, Christian-fasting, and kosher suitability from ingredients |
| **IFT synthesis**        | Generate pairs for ingredients, procedures, provenance, suitability, and recipe similarity       |
| **Enrichment**           | Rewrite instructions and responses for variety while keeping labels and evidence fixed           |

### Fine-tuning results

Five models were fine-tuned with LoRA (rank 16, 7,437 steps, effective batch 32)
using each model's chat template. Accuracy on AraCuisine-Bench, as mean ± std over
the five option orders:

| Model               | Base         | IFT          | Δ     |
| ------------------- | ------------ | ------------ | ----- |
| ALLaM-7B-Instruct   | 74.31 ± 0.64 | 76.74 ± 0.53 | +2.43 |
| Gemma-3-12B-it      | 69.92 ± 0.59 | 72.36 ± 0.62 | +2.44 |
| Fanar-1-9B-Instruct | 69.52 ± 0.53 | 72.28 ± 0.57 | +2.77 |
| Llama-3-8B-Instruct | 59.53 ± 0.24 | 62.98 ± 0.44 | +3.45 |
| Qwen3-8B            | 57.39 ± 0.60 | 66.61 ± 0.73 | +9.22 |

Training code and the full configuration are in the
[GitHub repository](https://github.com/mbzuai-nlp/arabic-cuisine/tree/main/IFT).

## Repository layout

```
bench/
├── aracuisine_bench_option_order_v{1..5}.json   # 5 option orders
├── aracuisine_bench_manifest.tsv                # item IDs, categories, cultural subset
└── stats.json
ift/                                             # coming soon
├── aracuisine_ift.parquet                       # 475,654 pairs (~170 MB, zstd)
└── stats.json
```

## Sources

Recipes were collected from five public Arabic culinary platforms: Atyab Tabkha
(~30K), Just Food (~16K), Mawdoo3 (~11.5K), Sayidaty Kitchen (~6K), and WebTeb
(~1.5K), for 64,987 recipes in total. Only derived data is released, not raw web pages.

## Limitations

- Dietary, health, and religious labels follow operational ingredient-level rules
  defined in the paper. They are dataset criteria, not medical or religious advice.
  Models trained on this data should not be used for such advice without expert review.
- Coverage reflects the five source platforms. Rare dishes, local names, and
  community-specific practices may be underrepresented.
- IFT responses were rewritten with a language model under consistency checks.
  Rare wording errors may remain.
- Several benchmark questions share the same stem with different options (for
  example, "which of these recipes suits diabetics?"), so the stem alone is not a
  unique key. Use `item_id` instead.

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
