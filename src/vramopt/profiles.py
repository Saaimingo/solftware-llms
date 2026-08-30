"""Atomic JSON profile persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def save_profile(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a profile after flushing it to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile(path: Path) -> dict[str, object]:
    """Load a profile and reject non-object JSON roots."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("profile root must be a JSON object")
    return cast(dict[str, object], payload)
