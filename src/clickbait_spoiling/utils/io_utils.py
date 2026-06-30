import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dictionaries."""
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write a list of dictionaries to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists and return its Path representation."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a standard YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def directory_size_gb(path: Union[str, Path]) -> float:
    """Calculate the cumulative file size of a directory in Gigabytes."""
    path = Path(path)
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)
