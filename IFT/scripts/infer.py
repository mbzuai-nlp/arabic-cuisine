import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    read_prompt_format,
    render_prompt,
    load_instruction_datasets,
    select_dataset_split,
)


def load_sft_lora_model(adapter_path: str):
    meta_path = os.path.join(adapter_path, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing run_meta.json in {adapter_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(meta["hf_id"], **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, add_special_tokens: bool = True) -> str:
    model_device = next(model.parameters()).device
    # Chat-template prompts already contain BOS and role tokens.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(model_device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def load_prompt_from_dataset(data_path: str, split: str, sample_index: int):
    dataset_dict = load_instruction_datasets(data_path, eval_split=0.0)
    dataset, used_split = select_dataset_split(dataset_dict, split)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range for split '{used_split}' with {len(dataset)} samples"
        )
    record = dataset[sample_index]
    return record, used_split


def main():
    ap = argparse.ArgumentParser(
        "Single-sample inference for instruction-tuned Arabic cuisine models"
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--input_text", default="")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    if not args.instruction and not args.data_path:
        raise ValueError(
            "Provide either --instruction or --data_path for inference input"
        )

    if args.data_path:
        record, used_split = load_prompt_from_dataset(
            args.data_path,
            args.split,
            args.sample_index,
        )
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        print(f"[INFO] Loaded sample {args.sample_index} from split '{used_split}'")
    else:
        instruction = args.instruction or ""
        input_text = args.input_text

    tokenizer, model = load_sft_lora_model(args.model_path)
    prompt_format = read_prompt_format(args.model_path)
    prompt = render_prompt(tokenizer, instruction, input_text, prompt_format)
    response = generate(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        add_special_tokens=(prompt_format != "chat_template"),
    )

    print("Prompt:\n")
    print(prompt)
    print("\nResponse:\n")
    print(response)


if __name__ == "__main__":
    main()
