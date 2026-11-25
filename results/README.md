# Results

Everything needed to reproduce the AraCuisine-Bench numbers reported in the paper.

```
results/
├── eval_benchmark/        # the exact 5 benchmark files the paper was evaluated on
├── predictions/           # 50 lm-evaluation-harness per-sample logs (xz-compressed)
├── score.py               # rebuilds the tables below from the logs
├── accuracy.tsv           # accuracy per version, mean ± std (paper Tables 3 and 15)
├── category_error_rates.tsv  # error rate per reasoning category (paper Tables 5 and 17)
└── cultural_subset.tsv    # accuracy on the 302-item cultural subset
```

## Prediction logs

`predictions/` holds one log per model, setup, and option-order version:
5 models × 2 setups (`base`, `ift`) × 5 versions = 50 files, named
`<model>_<setup>_v<k>.jsonl.xz`.

| Model ID in file names | Model                 |
| ---------------------- | --------------------- |
| `allam-7b-instruct`    | ALLaM-7B-Instruct     |
| `gemma-3-12b-it`       | Gemma-3-12B-it        |
| `fanar-1-9b-instruct`  | Fanar-1-9B-Instruct   |
| `llama-3-8b-instruct`  | Llama-3-8B-Instruct   |
| `qwen3-8b`             | Qwen3-8B              |

Each line is one benchmark item as logged by lm-evaluation-harness (`--log_samples`):
`doc_id`, the scored `doc`, the full prompt, the log-likelihood of every option
(`resps`), and `acc` (1 if the highest-scoring option is the gold answer).

Read a log without unpacking it:

```python
import json, lzma

with lzma.open("results/predictions/qwen3-8b_ift_v1.jsonl.xz", "rt") as f:
    rows = [json.loads(line) for line in f]
print(sum(r["acc"] for r in rows) / len(rows))  # 0.6664
```

## Reproducing the tables

```bash
python results/score.py
```

This reads the logs, joins them with `data/benchmark/aracuisine_bench_manifest.tsv`
on `doc_id`, and rewrites the three TSV files. Standard deviations are sample
standard deviations over the five versions. The output matches the paper exactly.

## Evaluation snapshot vs. release

`eval_benchmark/` contains the benchmark files exactly as they were scored. The
released benchmark in [`data/benchmark/`](../data/benchmark/) was finalized afterwards:

- 131 of 1,532 items had text corrections. Gold answers are unchanged.
  - 119 question stems: the typo "نعتبر" was corrected to "تعتبر".
  - 34 distractors: "المطبخ الشمال إفريقيا" was reworded (33 to "مطبخ شمال أفريقيا",
    1 to "المطبخ المغاربي").
  - 10 options: a stray second letter prefix was removed (e.g. "E. C.العدس والفول المنقوع").
  - 3 items: a duplicated distractor was replaced with a new one.
- An `item_id` field was added.
- The five versions were regenerated as cyclic rotations of `v1`, so every option
  appears once in each position. In the evaluation snapshot, the five versions are
  independent reorderings of the same five options.

Item order (`doc_id`) is the same in both, so the manifest applies to both. Use
`eval_benchmark/` to reproduce the paper, and `data/benchmark/` for new evaluations.
