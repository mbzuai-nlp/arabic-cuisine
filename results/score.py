"""Rebuild the AraCuisine-Bench result tables from the lm-eval prediction logs.

Usage (from the repository root):
    python results/score.py

Reads results/predictions/*.jsonl.xz and data/benchmark/aracuisine_bench_manifest.tsv,
then writes three TSV files to results/:
    accuracy.tsv             accuracy per option-order version, mean and sample std (paper Table 3 and 15)
    category_error_rates.tsv error rate per reasoning category, mean and sample std (paper Tables 5 and 17)
    cultural_subset.tsv      accuracy on the 302-item cultural subset (paper cultural-subset table)
"""

import csv
import glob
import json
import lzma
import os
import re
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results", "predictions")
MANIFEST = os.path.join(ROOT, "data", "benchmark", "aracuisine_bench_manifest.tsv")
OUT = os.path.join(ROOT, "results")

MODELS = ["allam-7b-instruct", "gemma-3-12b-it", "fanar-1-9b-instruct", "llama-3-8b-instruct", "qwen3-8b"]
SETUPS = ["base", "ift"]
VERSIONS = [1, 2, 3, 4, 5]
CATEGORIES = [
    "Religious Compatibility", "Geographic Provenance", "Dietary/Health/Lifestyle Compliance",
    "Technical Parameters", "Ingredient Analysis", "Procedural Execution", "Culinary Explanation",
]

with open(MANIFEST, encoding="utf-8") as f:
    manifest = {int(r["doc_id"]): r for r in csv.DictReader(f, delimiter="\t")}

# correct[(model, setup, version)][doc_id] = 0 or 1
correct = {}
for path in glob.glob(os.path.join(PRED, "*.jsonl.xz")):
    model, setup, version = re.match(r"(.+)_(base|ift)_v(\d)\.jsonl\.xz$", os.path.basename(path)).groups()
    with lzma.open(path, "rt", encoding="utf-8") as f:
        correct[(model, setup, int(version))] = {r["doc_id"]: r["acc"] for r in map(json.loads, f)}


def pct(values):
    return 100.0 * sum(values) / len(values)


def mean_std(xs):
    return statistics.mean(xs), statistics.stdev(xs)


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


rows = []
for m in MODELS:
    for s in SETUPS:
        accs = [pct(list(correct[(m, s, v)].values())) for v in VERSIONS]
        mu, sd = mean_std(accs)
        rows.append([m, s, *[f"{a:.2f}" for a in accs], f"{mu:.2f}", f"{sd:.2f}"])
write("accuracy.tsv", ["model", "setup", "v1", "v2", "v3", "v4", "v5", "mean", "std"], rows)

rows = []
for m in MODELS:
    for s in SETUPS:
        for c in CATEGORIES:
            ids = [d for d, r in manifest.items() if r["reasoning_category"] == c]
            errs = [100.0 - pct([correct[(m, s, v)][d] for d in ids]) for v in VERSIONS]
            mu, sd = mean_std(errs)
            rows.append([m, s, c, len(ids), f"{mu:.2f}", f"{sd:.2f}"])
write("category_error_rates.tsv", ["model", "setup", "reasoning_category", "items", "error_mean", "error_std"], rows)

rows = []
cultural = [d for d, r in manifest.items() if r["cultural_subset"] == "true"]
for m in MODELS:
    base = statistics.mean(pct([correct[(m, "base", v)][d] for d in cultural]) for v in VERSIONS)
    ift = statistics.mean(pct([correct[(m, "ift", v)][d] for d in cultural]) for v in VERSIONS)
    rows.append([m, len(cultural), f"{base:.2f}", f"{ift:.2f}", f"{ift - base:+.2f}"])
write("cultural_subset.tsv", ["model", "items", "base_acc", "ift_acc", "gain_pp"], rows)

print("Wrote accuracy.tsv, category_error_rates.tsv, cultural_subset.tsv")
