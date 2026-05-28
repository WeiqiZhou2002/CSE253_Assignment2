from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """Create all configured output directories."""
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir"):
            ensure_dir(PROJECT_ROOT / value)
    ensure_dir(PROJECT_ROOT / "outputs" / "processed")
    ensure_dir(PROJECT_ROOT / "notebooks")
    ensure_dir(PROJECT_ROOT / "submission")


def resolve_device(name: str) -> torch.device:
    """Resolve a configured torch device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def save_json(obj: Any, path: str | Path) -> None:
    """Write JSON with stable indentation."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a list into contiguous batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

