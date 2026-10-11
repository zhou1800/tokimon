"""Source templates for AI Harness generated package content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TEMPLATE_ROOT = Path(__file__).with_name("templates")


def render_text_template(name: str, values: Mapping[str, str]) -> str:
    template = _read_template(name)
    for key, value in values.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template.rstrip() + "\n"


def load_json_template(name: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_template(name))
    except json.JSONDecodeError as exc:
        raise ValueError(f"template {name} must contain a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"template {name} must contain a JSON object")
    return payload


def _read_template(name: str) -> str:
    path = TEMPLATE_ROOT / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read template {name}: {exc}") from exc
