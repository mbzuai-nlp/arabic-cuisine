import os
import glob
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file and return the parsed object (defaults to an empty dict)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into memory."""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_jsonl_paths(path: str) -> List[str]:
    """Resolve a JSONL file or a directory of JSONL files into a sorted list of paths."""
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        jsonl_paths = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if jsonl_paths:
            return jsonl_paths

    raise FileNotFoundError(f"No JSONL data found at: {path}")


def normalize_instruction_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all instruction-tuning records have normalized string fields."""
    normalized = dict(record)
    normalized["instruction"] = str(record.get("instruction") or "").strip()
    normalized["input"] = str(record.get("input") or "").strip()
    normalized["output"] = str(record.get("output") or "").strip()
    return normalized


def load_instruction_datasets(
    path: str,
    eval_split: float = 0.0,
    seed: int = 42,
) -> DatasetDict:
    """Load an instruction dataset from a HF dataset directory, a JSONL file, or a JSONL folder."""
    dataset_obj = None

    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_dict.json")):
        dataset_obj = load_from_disk(path)
    elif os.path.isdir(path) and os.path.exists(os.path.join(path, "state.json")):
        dataset_obj = load_from_disk(path)
    else:
        data_files = {"train": resolve_jsonl_paths(path)}
        dataset_obj = load_dataset("json", data_files=data_files)

    if isinstance(dataset_obj, DatasetDict):
        dataset_dict = dataset_obj
    elif isinstance(dataset_obj, Dataset):
        dataset_dict = DatasetDict({"train": dataset_obj})
    else:
        raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)!r}")

    normalized_splits = {}
    for split_name, split_dataset in dataset_dict.items():
        normalized_splits[split_name] = split_dataset.map(normalize_instruction_record)

    dataset_dict = DatasetDict(normalized_splits)

    if (
        "train" in dataset_dict
        and "test" not in dataset_dict
        and "validation" not in dataset_dict
        and eval_split > 0.0
    ):
        split_dict = dataset_dict["train"].train_test_split(
            test_size=eval_split,
            seed=seed,
            shuffle=True,
        )
        dataset_dict = DatasetDict(
            {"train": split_dict["train"], "test": split_dict["test"]}
        )

    return dataset_dict


def select_dataset_split(
    dataset_dict: DatasetDict, preferred_split: str
) -> Tuple[Dataset, str]:
    """Select the requested split, falling back to validation or the first available split."""
    if preferred_split in dataset_dict:
        return dataset_dict[preferred_split], preferred_split

    if preferred_split == "test" and "validation" in dataset_dict:
        return dataset_dict["validation"], "validation"

    first_split = next(iter(dataset_dict.keys()))
    return dataset_dict[first_split], first_split


def format_instruction_prompt(
    instruction: str,
    input_text: str = "",
    output_text: Optional[str] = None,
) -> str:
    lines = [f"Instruction:\n{instruction.strip()}"]

    if input_text.strip():
        lines.append(f"Input:\n{input_text.strip()}")

    if output_text is None:
        lines.append("Response:\n")
    else:
        lines.append(f"Response:\n{output_text.strip()}")

    return "\n\n".join(lines)


def ensure_dir(path: Optional[str]) -> Optional[str]:
    """Create the directory if it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def set_all_seeds(seed: int) -> None:
    """Set random seeds across common libraries."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.manual_seed_all(seed)  # type: ignore
    except ImportError:
        pass
