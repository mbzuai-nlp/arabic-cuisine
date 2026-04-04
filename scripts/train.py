import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse, json
from typing import List
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from scripts.utils import load_yaml, ensure_dir, set_all_seeds


# ------------------------------------------------
# Model loading (supports QLoRA)
# ------------------------------------------------
def load_model(hf_id: str, four_bit: bool):
    quant = None
    if four_bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant,
        attn_implementation="eager",
    )

    if quant is not None:
        model = prepare_model_for_kbit_training(model)

    return tokenizer, model


# ------------------------------------------------
# Prompt formatting
# ------------------------------------------------
def formatting_func_builder(tokenizer):
    def format_example(example):
        messages = [
            {
                "role": "user",
                "content": f"Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}",
            },
            {
                "role": "assistant",
                "content": example["output"],
            },
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return format_example


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    reg = load_yaml(args.models)

    set_all_seeds(cfg.get("seed", 42))
    has_cuda = torch.cuda.is_available()

    # Load dataset once
    ds = load_from_disk(args.dataset)
    train_dataset = ds["train"]
    train_dataset = Dataset.from_list(list(train_dataset))

    for alias in reg["train_targets"]:
        mcfg = reg["base_models"][alias]
        ovr = (reg.get("lora_overrides") or {}).get(alias, {})

        lora_cfg = LoraConfig(
            r=ovr.get("r", cfg["train"]["lora_defaults"]["r"]),
            lora_alpha=ovr.get("alpha", cfg["train"]["lora_defaults"]["alpha"]),
            lora_dropout=ovr.get("dropout", cfg["train"]["lora_defaults"]["dropout"]),
            target_modules=ovr.get(
                "target_modules", cfg["train"]["lora_defaults"]["target_modules"]
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )

        four_bit = bool(ovr.get("four_bit", cfg["train"]["lora_defaults"]["four_bit"]))
        if four_bit and not has_cuda:
            four_bit = False

        tokenizer, model = load_model(mcfg["hf_id"], four_bit=four_bit)

        out_dir = os.path.join(cfg["paths"]["lora_out"], alias)
        ensure_dir(out_dir)

        sft_args = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=cfg["train"]["num_train_epochs"],
            learning_rate=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
            per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
            max_grad_norm=cfg["train"]["max_grad_norm"],
            warmup_ratio=cfg["train"]["warmup_ratio"],
            logging_steps=cfg["train"]["logging_steps"],
            save_steps=cfg["train"]["save_steps"],
            fp16=cfg["train"]["fp16"] and has_cuda,
            bf16=cfg["train"]["bf16"] and has_cuda,
            gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
            dataloader_num_workers=0,
            dataloader_pin_memory=has_cuda,
            report_to=[],
            packing=False,
            max_length=mcfg["max_len"],
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            peft_config=lora_cfg,
            formatting_func=formatting_func_builder(tokenizer),
            processing_class=tokenizer,
        )

        trainer.train()

        # Save adapters only
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump({"alias": alias, "hf_id": mcfg["hf_id"], "four_bit": four_bit}, f)

        print(f"[OK] Saved LoRA adapters → {out_dir}")


if __name__ == "__main__":
    main()
