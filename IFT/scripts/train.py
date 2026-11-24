import os
import argparse, json
import importlib.util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from utils import (
    PROMPT_FORMATS,
    build_chat_messages,
    ensure_dir,
    format_instruction_prompt,
    load_instruction_datasets,
    load_yaml,
    select_dataset_split,
    set_all_seeds,
)


def _resolve_attn_implementation(requested: str, is_main: bool = True) -> str:
    """Pick a working attention implementation, falling back gracefully.

    - "auto": prefer flash_attention_2 if installed, else sdpa.
    - "flash_attention_2": fall back to sdpa if flash_attn isn't importable.
    - "sdpa" / "eager": pass through.
    """
    has_flash = importlib.util.find_spec("flash_attn") is not None

    if requested in (None, "", "auto"):
        chosen = "flash_attention_2" if has_flash else "sdpa"
    elif requested == "flash_attention_2" and not has_flash:
        chosen = "sdpa"
        if is_main:
            print(
                "[WARN] attn_implementation='flash_attention_2' requested but flash_attn "
                "is not installed. Falling back to 'sdpa'. To enable flash-attn run: "
                "pip install flash-attn --no-build-isolation"
            )
    else:
        chosen = requested

    if is_main:
        print(f"[INFO] Using attn_implementation: {chosen}")
    return chosen


# ------------------------------------------------
# Model loading (supports QLoRA)
# ------------------------------------------------
def load_model(
    hf_id: str,
    four_bit: bool,
    lora_cfg: LoraConfig,
    local_rank: int = 0,
    attn_implementation: str = "auto",
):
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

    chosen_attn = _resolve_attn_implementation(
        attn_implementation, is_main=(local_rank == 0)
    )

    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": chosen_attn,
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


def to_chat_dataset(dataset):
    """Convert instruction records to conversational records so TRL applies the
    model-specific chat template (the setup used in the paper)."""

    def convert(example):
        return {
            "messages": build_chat_messages(
                example["instruction"],
                example.get("input", ""),
                example["output"],
            )
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


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

    prompt_format = cfg["train"].get("prompt_format", "chat_template")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"train.prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}")

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

        # Resolution order: per-model override > global train.attn_implementation > "auto"
        attn_impl = ovr.get(
            "attn_implementation",
            cfg["train"].get("attn_implementation", "auto"),
        )

        tokenizer, model = load_model(
            mcfg["hf_id"],
            four_bit=four_bit,
            lora_cfg=lora_cfg,
            local_rank=local_rank,
            attn_implementation=attn_impl,
        )

        out_dir = os.path.join(cfg["paths"]["lora_out"], alias)
        ensure_dir(out_dir)
        log_dir = os.path.join(cfg["paths"]["logs_out"], alias)
        ensure_dir(log_dir)

        if prompt_format == "chat_template" and not getattr(tokenizer, "chat_template", None):
            raise ValueError(f"{mcfg['hf_id']} has no chat template; set train.prompt_format: alpaca")

        save_strategy = cfg["train"].get("save_strategy", "epoch")
        sft_kwargs = dict(
            output_dir=out_dir,
            max_steps=cfg["train"].get("max_steps", -1),
            num_train_epochs=cfg["train"].get("num_train_epochs", 1),
            learning_rate=cfg["train"]["learning_rate"],
            lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "linear"),
            optim=cfg["train"].get("optim", "adamw_torch"),
            seed=cfg.get("seed", 42),
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
            dataloader_num_workers=cfg["train"].get("dataloader_num_workers", 2),
            dataloader_pin_memory=cfg["train"].get("dataloader_pin_memory", True) and has_cuda,
            report_to="none",
            packing=cfg["train"].get("packing", False),
            max_length=mcfg["max_len"],
            ddp_find_unused_parameters=True,
        )

        # Step-based saving requires save_steps; cap retained checkpoints if set.
        if save_strategy == "steps" and "save_steps" in cfg["train"]:
            sft_kwargs["save_steps"] = cfg["train"]["save_steps"]
        if cfg["train"].get("save_total_limit") is not None:
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

        # Older TRL/transformers versions accept logging_dir; newer ones removed it.
        if "logging_dir" in SFTConfig.__dataclass_fields__:
            sft_kwargs["logging_dir"] = log_dir

        sft_args = SFTConfig(**sft_kwargs)

        if prompt_format == "chat_template":
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=to_chat_dataset(train_dataset),
                eval_dataset=to_chat_dataset(eval_dataset) if eval_dataset is not None else None,
                processing_class=tokenizer,
            )
        else:
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
                        "prompt_format": prompt_format,
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
