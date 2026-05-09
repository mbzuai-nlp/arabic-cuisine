import os
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
def load_model(hf_id: str, four_bit: bool, lora_cfg: LoraConfig, local_rank: int = 0):
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
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "flash_attention_2", #eager
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"" : local_rank}
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

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

    if is_main:
        print(
            f"[INFO] Loaded {len(train_dataset)} training samples from split '{train_split_name}'"
        )
        if eval_dataset is not None:
            print(
                f"[INFO] Loaded {len(eval_dataset)} evaluation samples from split '{eval_split_name}'"
            )

    total = len(reg["train_targets"])
    for idx, alias in enumerate(reg["train_targets"], start=1):
        mcfg = reg["base_models"][alias]
        ovr = (reg.get("lora_overrides") or {}).get(alias, {})

        if is_main:
            print(f"\n{'='*60}")
            print(f"[SFT] Model {idx}/{total}: {alias}  ({mcfg['hf_id']})")
            print(f"{'='*60}\n")

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
            mcfg["hf_id"], four_bit=four_bit, lora_cfg=lora_cfg, local_rank=local_rank
        )

        out_dir = os.path.join(cfg["paths"]["lora_out"], alias)
        ensure_dir(out_dir)
        log_dir = os.path.join(cfg["paths"]["logs_out"], alias)
        ensure_dir(log_dir)

        save_strategy = cfg["train"].get("save_strategy", "epoch")
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
            save_strategy=save_strategy,
            fp16=cfg["train"]["fp16"] and has_cuda,
            bf16=cfg["train"]["bf16"] and has_cuda,
            gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=has_cuda,
            report_to="none",
            packing=False,
            max_length=mcfg["max_len"],
            ddp_find_unused_parameters=True,
        )

        # Step-based saving requires save_steps; cap retained checkpoints if set.
        if save_strategy == "steps" and "save_steps" in cfg["train"]:
            sft_kwargs["save_steps"] = cfg["train"]["save_steps"]
        if "save_total_limit" in cfg["train"]:
            sft_kwargs["save_total_limit"] = cfg["train"]["save_total_limit"]

        if eval_dataset is not None:
            sft_kwargs["per_device_eval_batch_size"] = cfg["train"].get(
                "per_device_eval_batch_size",
                cfg["train"]["per_device_train_batch_size"],
            )
            eval_strategy = cfg["train"].get("eval_strategy", "epoch")
            sft_kwargs["eval_strategy"] = eval_strategy
            if eval_strategy == "steps" and "eval_steps" in cfg["train"]:
                sft_kwargs["eval_steps"] = cfg["train"]["eval_steps"]
            # load_best_model_at_end is incompatible with DDP + QLoRA (Trainer tries
            # to reload with device_map='auto', conflicting with the per-rank device map).
            # The final checkpoint is saved explicitly below instead.
            sft_kwargs["load_best_model_at_end"] = False
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

        if is_main:
            model.print_trainable_parameters()
        trainer.train()

        # Save adapters only (main process only)
        if is_main:
            trainer.model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        if is_main:
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
