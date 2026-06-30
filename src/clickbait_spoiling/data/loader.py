import json
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets

from clickbait_spoiling.schema import ClickbaitPost


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of raw dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required data file: {path}. "
            f"Please ensure you place the SemEval dataset files in data/raw/."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_split(path: Union[str, Path]) -> List[ClickbaitPost]:
    """Load JSONL and parse into ClickbaitPost objects."""
    records = load_jsonl(path)
    return [ClickbaitPost.from_dict(r) for r in records]


def to_hf_dataset(posts: List[ClickbaitPost]) -> datasets.Dataset:
    """Convert a list of ClickbaitPost into a Hugging Face Dataset of plain dicts."""
    records = []
    for p in posts:
        record = {
            "uuid": p.uuid,
            "postText": p.post_text,
            "targetParagraphs": p.target_paragraphs,
            "targetTitle": p.target_title,
            "targetUrl": p.target_url,
            "humanSpoiler": p.human_spoiler,
            "spoiler": p.spoiler,
            "spoilerPositions": p.spoiler_positions,
            "tags": p.tags,
        }
        records.append(record)
    return datasets.Dataset.from_list(records)
