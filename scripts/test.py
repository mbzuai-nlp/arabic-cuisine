#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT Model Testing Script

Features:
- Base model + LoRA adapter loading
- Robust system-prompt detection (Gemma-safe)
- Chat-template–consistent prompting
- Safe resume
- Atomic save after each sample
"""

import os
import json
import argparse
import time
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ------------------------------------------------
# System prompt
# ------------------------------------------------
SYSTEM_PROMPT = (
    "You are a professional Arabic translation system specialized in "
    "Modern Standard Arabic (MSA) and regional Arabic dialects.\n"
    "Rules:\n"
    "1. Output only the translated text in Arabic.\n"
    "2. Do not add explanations, comments, labels, or quotation marks.\n"
    "3. Preserve meaning, tone, and level of formality.\n"
    "4. Do not change names, numbers, or entities.\n"
    "5. If the input already matches the target variety, return it unchanged."
)


# ------------------------------------------------
# Model loading (Base + LoRA)
# ------------------------------------------------
def load_sft_lora_model(adapter_path: str):
    meta_path = os.path.join(adapter_path, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing run_meta.json in {adapter_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    base_hf_id = meta["hf_id"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return tokenizer, model


# ------------------------------------------------
# System-role capability probe (robust)
# ------------------------------------------------
def supports_system_role(tokenizer) -> bool:
    """
    Gemma tokenizer does NOT support system role.
    Most others do.
    """
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


# ------------------------------------------------
# Message builder (authoritative)
# ------------------------------------------------
def build_messages(
    record: Dict[str, Any],
    direction: str,
    tokenizer,
) -> List[Dict[str, str]]:
    region = record["region"]
    country = record["country"]

    if direction == "msa_to_dialect":
        user_prompt = (
            f"Translate the following Modern Standard Arabic (MSA) text into the "
            f"{region} dialect of {country}. Do not include any explanation, "
            f"just return the Arabic translation.\n\n"
            f"MSA Text: {record['dialogue_msa']}\n\n"
            f"Dialect Translation:"
        )
    else:
        user_prompt = (
            f"Translate the following Arabic text from the {region} dialect of "
            f"{country} into Modern Standard Arabic (MSA). Do not include any "
            f"explanation, just return the Arabic translation.\n\n"
            f"Dialect Text: {record['dialogue_dialect']}\n\n"
            f"MSA Translation:"
        )

    if supports_system_role(tokenizer):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    else:
        return [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            }
        ]


# ------------------------------------------------
# Generation
# ------------------------------------------------
@torch.no_grad()
def generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 4096,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip()


# ------------------------------------------------
# Safe JSON helpers
# ------------------------------------------------
def load_existing_results(path: str) -> Dict[int, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def atomic_write_json(data: List[Dict], path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------
# Data loading
# ------------------------------------------------
def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser("SFT model testing (safe + resume)")
    ap.add_argument("--data_path", default="dialect_translation/data/test.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", default="dialect_translation/output/sft/")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_path)
    if args.limit:
        data = data[: args.limit]

    model_name = os.path.basename(args.model_path)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    existing = load_existing_results(output_path)
    done_ids = set(existing.keys())

    print(f"[INFO] Loaded {len(data)} samples")
    print(f"[INFO] Resuming from {len(done_ids)} completed samples")

    tokenizer, model = load_sft_lora_model(args.model_path)

    results = list(existing.values())
    start = time.time()

    for idx, record in enumerate(data, 1):
        rid = record["id"]
        if rid in done_ids:
            continue

        print(f"[{idx}/{len(data)}] Processing ID={rid}")

        try:
            msa_messages = build_messages(record, "msa_to_dialect", tokenizer)
            msa_out = generate(model, tokenizer, msa_messages)

            dia_messages = build_messages(record, "dialect_to_msa", tokenizer)
            dia_out = generate(model, tokenizer, dia_messages)

            item = {
                "id": rid,
                "region": record["region"],
                "country": record["country"],
                "original_msa": record["dialogue_msa"],
                "original_dialect": record["dialogue_dialect"],
                "msa_to_dialect_prediction": msa_out,
                "dialect_to_msa_prediction": dia_out,
                "model": model_name,
            }

        except Exception as e:
            print(f"[ERROR] ID={rid}: {e}")
            item = {"id": rid, "error": str(e), "model": model_name}

        results.append(item)
        done_ids.add(rid)
        atomic_write_json(results, output_path)

    elapsed = time.time() - start
    print(f"[DONE] Saved → {output_path}")
    print(f"[TIME] {elapsed:.2f}s total | {elapsed / max(1, len(results)):.2f}s/sample")


if __name__ == "__main__":
    main()
