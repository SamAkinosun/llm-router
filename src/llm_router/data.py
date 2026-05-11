"""Data loading for training and evaluation sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} is not valid JSON: {e.msg}") from e
            if "prompt" not in row or "label" not in row:
                raise ValueError(f"{path}:{i} missing required fields 'prompt' and 'label'")
            rows.append(row)
    return rows


def split_prompts_labels(rows: List[dict]) -> Tuple[List[str], List[str]]:
    prompts = [r["prompt"] for r in rows]
    labels = [r["label"] for r in rows]
    return prompts, labels
