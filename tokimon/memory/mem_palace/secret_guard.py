"""Secret-boundary checks for Mem Palace write paths."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SECRET_KEY_RE = re.compile(
    r"(token|api[_-]?key|password|passwd|secret|cookie|authorization|auth[_-]?code)",
    re.IGNORECASE,
)
TOKEN_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GitHub token", re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}\b")),
)
KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(?P<key>token|api[_-]?key|password|passwd|secret|cookie|authorization|auth[_-]?code)"
    r"(?P<sep>\s*[:=]\s*)['\"]?(?P<value>[^'\"\s,;]+)"
)
PLACEHOLDER_WORDS = (
    "example",
    "placeholder",
    "redacted",
    "dummy",
    "sample",
    "changeme",
    "your_",
    "your-",
)


def find_secret_indicators(value: Any, *, path: str = "value") -> list[str]:
    """Return non-secret evidence labels for high-confidence secret-like input."""

    indicators: list[str] = []
    _collect_secret_indicators(value, path=path, indicators=indicators)
    return indicators


def reject_secret_material(value: Any, *, context: str) -> None:
    indicators = find_secret_indicators(value, path=context)
    if not indicators:
        return
    shown = ", ".join(indicators[:3])
    if len(indicators) > 3:
        shown += f", +{len(indicators) - 3} more"
    raise ValueError(f"{context} contains secret-like material: {shown}")


def _collect_secret_indicators(value: Any, *, path: str, indicators: list[str]) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if SECRET_KEY_RE.search(key) and _value_looks_sensitive(raw_value):
                indicators.append(f"{child_path} is a sensitive field")
            _collect_secret_indicators(raw_value, path=child_path, indicators=indicators)
        return

    if isinstance(value, str):
        indicators.extend(_text_secret_indicators(value, path=path))
        return

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _collect_secret_indicators(item, path=f"{path}[{index}]", indicators=indicators)


def _text_secret_indicators(text: str, *, path: str) -> list[str]:
    indicators: list[str] = []
    for label, pattern in TOKEN_TEXT_PATTERNS:
        if pattern.search(text):
            indicators.append(f"{path} matches {label}")
    for match in KEY_VALUE_SECRET_RE.finditer(text):
        key = match.group("key")
        if _value_looks_sensitive(match.group("value")):
            indicators.append(f"{path} contains secret-like {key} assignment")
    return indicators


def _value_looks_sensitive(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().strip("'\"")
    if len(stripped) < 8:
        return False

    lowered = stripped.lower()
    if any(word in lowered for word in PLACEHOLDER_WORDS):
        return False
    if stripped.startswith(("<", "{", "$")) or stripped.endswith((">", "}")):
        return False
    if set(stripped) <= {"x", "X", "*", "-"}:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", stripped):
        return False
    return True
