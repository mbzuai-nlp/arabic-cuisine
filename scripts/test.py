#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instruction-tuned model testing script (raw generation only).
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import (
    format_instruction_prompt,
    load_instruction_datasets,
    select_dataset_split,
)


# ------------------------------------------------
# Model loading (Base + LoRA)
# ------------------------------------------------
def load_sft_lora_model(adapter_path: str):
    meta_path = os.path.join(adapter_path, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing run_meta.json in {adapter_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_hf_id = meta["hf_id"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_hf_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return tokenizer, model


# ------------------------------------------------
# Prompt construction (MCQ → instruction)
# ------------------------------------------------
def build_prompt(record: Dict[str, Any]) -> str:
    question = str(record["question"]).strip()
    options = record.get("options", [])

    options_text = "\n".join(str(opt).strip() for opt in options)

    instruction = (
        "اختر الإجابة الصحيحة من بين الخيارات التالية. "
        "أجب بصيغة: الحرف ثم نص الخيار فقط."
    )

    input_text = f"السؤال:\n{question}\n\n" f"الخيارات:\n{options_text}"

    return format_instruction_prompt(
        instruction=instruction,
        input_text=input_text,
        output_text=None,
    )


# ------------------------------------------------
# Generation
# ------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ------------------------------------------------
# Safe JSON helpers
# ------------------------------------------------
def load_existing_results(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]): item for item in data}


def atomic_write_json(data: List[Dict[str, Any]], path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Instruction-tuned model testing")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", default="outputs/sft")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_dict = load_instruction_datasets(args.data_path, eval_split=0.0)
    eval_dataset, used_split = select_dataset_split(dataset_dict, args.split)

    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    existing = load_existing_results(output_path)
    done_ids = set(existing.keys())

    total_samples = len(eval_dataset)
    if args.limit is not None:
        total_samples = min(total_samples, args.limit)

    print(f"[INFO] Loaded {total_samples} samples from split '{used_split}'")
    print(f"[INFO] Resuming from {len(done_ids)} completed samples")

    tokenizer, model = load_sft_lora_model(args.model_path)

    results = list(existing.values())
    start = time.time()

    for idx, record in enumerate(eval_dataset, 1):
        if args.limit is not None and idx > args.limit:
            break

        rid = str(record.get("id", idx - 1))
        if rid in done_ids:
            continue

        print(f"[{idx}/{total_samples}] Processing ID={rid}")

        try:
            prompt = build_prompt(record)
            prediction = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )

            item = {
                "id": rid,
                "question": record.get("question", ""),
                "options": record.get("options", []),
                "correct_answer": record.get("correct_answer", ""),
                "prediction": prediction,
                "category": record.get("category", ""),
                "model": model_name,
                "prompt": prompt,
            }

            if "category" in record:
                item["category"] = record["category"]

        except Exception as e:
            print(f"[ERROR] ID={rid}: {e}")
            item = {
                "id": rid,
                "question": record.get("question", ""),
                "error": str(e),
                "model": model_name,
            }

        results.append(item)
        done_ids.add(rid)
        atomic_write_json(results, output_path)

    elapsed = time.time() - start
    print(f"[DONE] Saved → {output_path}")
    print(
        f"[TIME] {elapsed:.2f}s total | {elapsed / max(1, total_samples):.2f}s/sample"
    )


if __name__ == "__main__":
    main()
