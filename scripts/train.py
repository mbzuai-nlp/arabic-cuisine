import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from utils import (
    ensure_dir,
    format_instruction_prompt,
    load_instruction_datasets,
    load_yaml,
    select_dataset_split,
    set_all_seeds,
)


# ------------------------------------------------
# Model loading (supports QLoRA)
# ------------------------------------------------
def load_model(hf_id: str, four_bit: bool, lora_cfg: LoraConfig):
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

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "eager",
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if quant is not None:
        model_kwargs["quantization_config"] = quant

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.config.use_cache = False

    if quant is not None:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_cfg)

    return tokenizer, model


# ------------------------------------------------
# Prompt formatting
# ------------------------------------------------
def formatting_func_builder():
    def format_example(example):
        return format_instruction_prompt(
            example["instruction"],
            example.get("input", ""),
            example["output"],
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

    dataset_dict = load_instruction_datasets(
        args.dataset,
        eval_split=cfg["train"].get("eval_split", 0.0),
        seed=cfg.get("seed", 42),
    )
    train_dataset, train_split_name = select_dataset_split(dataset_dict, "train")

    eval_dataset = None
    eval_split_name = None
    if "test" in dataset_dict:
        eval_dataset, eval_split_name = select_dataset_split(dataset_dict, "test")
    elif "validation" in dataset_dict:
        eval_dataset, eval_split_name = select_dataset_split(dataset_dict, "validation")

    print(
        f"[INFO] Loaded {len(train_dataset)} training samples from split '{train_split_name}'"
    )
    if eval_dataset is not None:
        print(
            f"[INFO] Loaded {len(eval_dataset)} evaluation samples from split '{eval_split_name}'"
        )

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

        tokenizer, model = load_model(
            mcfg["hf_id"], four_bit=four_bit, lora_cfg=lora_cfg
        )

        out_dir = os.path.join(cfg["paths"]["lora_out"], alias)
        ensure_dir(out_dir)
        log_dir = os.path.join(cfg["paths"]["logs_out"], alias)
        ensure_dir(log_dir)

        sft_kwargs = dict(
            output_dir=out_dir,
            logging_dir=log_dir,
            num_train_epochs=cfg["train"]["num_train_epochs"],
            learning_rate=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
            per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
            max_grad_norm=cfg["train"]["max_grad_norm"],
            warmup_ratio=cfg["train"]["warmup_ratio"],
            logging_steps=cfg["train"]["logging_steps"],
            save_strategy=cfg["train"].get("save_strategy", "epoch"),
            fp16=cfg["train"]["fp16"] and has_cuda,
            bf16=cfg["train"]["bf16"] and has_cuda,
            gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
            dataloader_num_workers=0,
            dataloader_pin_memory=has_cuda,
            report_to="none",
            packing=False,
            max_length=mcfg["max_len"],
        )

        if eval_dataset is not None:
            sft_kwargs["per_device_eval_batch_size"] = cfg["train"].get(
                "per_device_eval_batch_size",
                cfg["train"]["per_device_train_batch_size"],
            )
            sft_kwargs["eval_strategy"] = cfg["train"].get("eval_strategy", "epoch")
            sft_kwargs["load_best_model_at_end"] = True
            sft_kwargs["metric_for_best_model"] = "eval_loss"
            sft_kwargs["greater_is_better"] = False
        else:
            sft_kwargs["eval_strategy"] = "no"

        sft_args = SFTConfig(**sft_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func_builder(),
            processing_class=tokenizer,
        )

        model.print_trainable_parameters()
        trainer.train()

        # Save adapters only
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(
                {
                    "alias": alias,
                    "hf_id": mcfg["hf_id"],
                    "four_bit": four_bit,
                    "train_size": len(train_dataset),
                    "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
                    "dataset_path": args.dataset,
                },
                f,
                indent=2,
            )

        print(f"[OK] Saved LoRA adapters → {out_dir}")


if __name__ == "__main__":
    main()
