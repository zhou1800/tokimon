"""Lightweight file logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def log_to_file(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a") as handle:
        handle.write(f"[{timestamp}] {message}\n")
