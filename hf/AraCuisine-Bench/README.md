---
license: cc-by-nc-4.0
language:
  - ar
pretty_name: AraCuisine-Bench
size_categories:
  - 1K<n<10K
task_categories:
  - question-answering
  - multiple-choice
tags:
  - arabic
  - cuisine
  - culture
  - multiple-choice
  - benchmark
configs:
  - config_name: v1
    default: true
    data_files:
      - split: test
        path: aracuisine_bench_option_order_v1.json
  - config_name: v2
    data_files:
      - split: test
        path: aracuisine_bench_option_order_v2.json
  - config_name: v3
    data_files:
      - split: test
        path: aracuisine_bench_option_order_v3.json
  - config_name: v4
    data_files:
      - split: test
        path: aracuisine_bench_option_order_v4.json
  - config_name: v5
    data_files:
      - split: test
        path: aracuisine_bench_option_order_v5.json
---

<h1 align="center">AraCuisine-Bench: A Benchmark for Arabic Culinary Reasoning</h1>

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
  <a href="https://huggingface.co/datasets/MBZUAI/AraCuisine-IFT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Companion-AraCuisine--IFT-FFD21E" alt="AraCuisine-IFT"></a>
</p>

---

**AraCuisine-Bench** is a human-validated benchmark of **1,532** five-choice questions
on Arabic cuisine. Questions test factual, procedural, regional, and dietary or
religious reasoning, and each one is grounded in a source recipe.

Every item was checked by **three native Arabic annotators** (average pairwise
Cohen's κ = 0.87). Distractors are plausible by design, drawn from related
ingredients, steps, regions, or constraints. The benchmark is released in **five
option orders** to remove gold-answer position bias.

## Dataset at a glance

|            |                                                       |
| ---------- | ----------------------------------------------------- |
| Questions  | 1,532                                                 |
| Language   | Arabic (`ar`)                                       |
| Format     | Single-answer multiple choice (5 options)             |
| Versions   | 5 option orders (`v1` to `v5`)                    |
| Categories | 7 reasoning categories (15 fine-grained labels)       |
| Files      | `aracuisine_bench_option_order_v{1..5}.json`, `aracuisine_bench_manifest.tsv`, `stats.json` |

## Schema

Each record has exactly these fields:

```json
{
  "question": "نعتبر وصفة الحريرة من الوصفات الشهيرة في",
  "options": ["A. المطبخ الكويتي", "B. المطبخ المغربي", "C. المطبخ المصري", "D. المطبخ اللبناني", "E. المطبخ الأردني"],
  "correct_answer": "B. المطبخ المغربي",
  "category": "country"
}
```

| Field            | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `question`       | Question stem in Arabic                              |
| `options`        | Five answer strings, each prefixed with `A.` to `E.` |
| `correct_answer` | The gold option, identical to one entry in `options` |
| `category`       | Fine-grained label (see below)                       |

## Option-order versions

Models can favour an answer position, for example always picking `A`. To remove
this **gold-answer position bias**, the benchmark ships in five versions that differ
only in the order of the options:

- `v1` is the base order. `v2` to `v5` rotate the five options by one to four
  positions, so every option, including the gold answer, appears once in each position.
- Questions, option texts, gold answers, and item order are identical across versions.
- Within each version, gold answers are spread evenly over `A` to `E` (about 306 each).

All results in the paper are the **mean ± std accuracy over `v1` to `v5`**. The
files are released exactly as used, so the reported numbers can be reproduced.

## Category distribution

| Reasoning category                  | Fine-grained labels                                   | Questions |
| ----------------------------------- | ----------------------------------------------------- | --------- |
| Procedural Execution                | `steps`                                               | 682       |
| Ingredient Analysis                 | `ingredients`                                         | 356       |
| Culinary Explanation                | `explanation`                                         | 145       |
| Geographic Provenance               | `country`, `region`                                   | 119       |
| Religious Compatibility             | `(not_)suitable_christians`, `not_suitable_moslems`, `not_suitable_jews` | 102       |
| Dietary/Health/Lifestyle Compliance | `(not_)suitable_diabetics`, `(not_)suitable_vegetarians` | 81        |
| Technical Parameters                | `time`, `temperature`                                 | 47        |

Per-label counts and the gold-answer position counts for each version are in
[`stats.json`](stats.json).

## Manifest

[`aracuisine_bench_manifest.tsv`](aracuisine_bench_manifest.tsv) has one row per item, in
the same order as the five version files:

| Column               | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `item_id`            | Stable identifier (`aracuisine-0000` to `aracuisine-1531`)                       |
| `doc_id`             | Position of the item in every version file, as logged by lm-evaluation-harness   |
| `original_category`  | Fine-grained label, same as `category` in the data                                |
| `reasoning_category` | One of the 7 reasoning categories used in the paper                               |
| `cultural_subset`    | `true` for the 302 Geographic, Religious, and Dietary/Health/Lifestyle items      |

Use it to join predictions across versions and models, and to compute the
per-category and cultural-subset scores reported in the paper.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("MBZUAI/AraCuisine-Bench", "v1", split="test")
ex = ds[0]
print(ex["question"])
print(ex["options"], "→", ex["correct_answer"])
```

### Evaluation

Treat each item as a single-answer MCQ: present `question` and `options`, score
each option, and compare the selected option against `correct_answer`. Accuracy is
the primary metric. Run all five versions and report the **mean ± std**, ideally
broken down by category.

```python
versions = ["v1", "v2", "v3", "v4", "v5"]
data = {v: load_dataset("MBZUAI/AraCuisine-Bench", v, split="test") for v in versions}

# Item IDs, reasoning categories, and the cultural-subset flag
manifest = load_dataset(
    "csv",
    data_files="hf://datasets/MBZUAI/AraCuisine-Bench/aracuisine_bench_manifest.tsv",
    delimiter="\t",
    split="train",
)
```

In the paper, options are scored by log-likelihood with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Fine-tuning
on [AraCuisine-IFT](https://huggingface.co/datasets/MBZUAI/AraCuisine-IFT) improves accuracy
for all five tested models, by 2.43 to 9.22 points.

## Sources

Recipes were collected from five public Arabic culinary platforms: Atyab Tabkha
(~30K), Just Food (~16K), Mawdoo3 (~11.5K), Sayidaty Kitchen (~6K), and WebTeb
(~1.5K), for 64,987 recipes in total. Only derived data is released, not raw web pages.

Benchmark recipes and their near-duplicates were removed from AraCuisine-IFT before
any training pair was generated.

## Limitations

- Dietary, health, and religious labels follow operational ingredient-level rules
  defined in the paper. They are dataset criteria, not medical or religious advice.
- Coverage reflects the five source platforms. Rare dishes, local names, and
  community-specific practices may be underrepresented.
- Several questions share the same stem with different options (for example,
  "which of these recipes suits diabetics?"), so the stem alone is not a unique key.

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
