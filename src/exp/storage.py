"""Append-only JSONL storage. No database, matches the pilot's throwaway scale."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: Any) -> None:
    if not is_dataclass(record):
        raise TypeError(f"expected a dataclass instance, got {type(record)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
