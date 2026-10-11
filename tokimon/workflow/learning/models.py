"""Small data helpers for Learning workflow artifacts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def timestamp_text(now: datetime | None = None) -> str:
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_timestamp(now: datetime | None = None) -> str:
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str | None, *, fallback: str = "learning-run") -> str:
    if not value:
        return fallback
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or fallback


def normalize_token(value: object) -> str:
    return slugify(str(value or ""), fallback="").replace("-", "_")


def require_mapping(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload
