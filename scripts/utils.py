import os
import random
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file and return the parsed object (defaults to an empty dict)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


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
