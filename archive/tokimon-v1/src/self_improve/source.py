"""Helpers for reading optional self-improvement inputs (URL, file, or inline text)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools.web_tool import WebTool


@dataclass(frozen=True)
class InputPayload:
    kind: str  # "none" | "url" | "file" | "text"
    ref: str | None
    content: str


def read_goal_input(goal: str, input_ref: str | None, max_bytes: int = 512_000) -> InputPayload:
    """Resolve the input payload for a self-improve run.

    Precedence:
    1) explicit --input value (URL/file/text)
    2) first URL detected in the goal (fetched)
    3) none
    """

    payload = read_optional_input(input_ref, max_bytes=max_bytes)
    if payload.kind != "none":
        return payload

    urls = extract_urls(goal, max_urls=3)
    if not urls:
        return payload
    return _read_url(urls[0], max_bytes=max_bytes)


def read_optional_input(ref: str | None, max_bytes: int = 512_000) -> InputPayload:
    if not ref:
        return InputPayload(kind="none", ref=None, content="")

    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        return _read_url(ref, max_bytes=max_bytes)

    path = Path(ref)
    if path.exists() and path.is_file():
        content = path.read_text(errors="replace")
        if len(content.encode()) > max_bytes:
            content = content.encode()[:max_bytes].decode(errors="replace")
        return InputPayload(kind="file", ref=str(path), content=content)

    return InputPayload(kind="text", ref=None, content=ref)


def _read_url(url: str, max_bytes: int) -> InputPayload:
    tool = WebTool()
    result = tool.fetch(url, max_bytes=max_bytes, timeout_s=15)
    if not result.ok:
        raise ValueError(result.error or result.summary)
    payload = result.data or {}
    if not isinstance(payload, dict):
        raise ValueError("web tool returned invalid payload")
    ref = payload.get("url")
    content = payload.get("content")
    if not isinstance(ref, str) or not isinstance(content, str):
        raise ValueError("web tool returned invalid payload")
    return InputPayload(kind="url", ref=ref, content=content)


def extract_urls(text: str, max_urls: int = 5) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    max_urls = int(max_urls)
    if max_urls <= 0:
        return []
    candidates = re.findall(r"https?://[^\s)\]>\"]+", text)
    urls: list[str] = []
    seen = set()
    for candidate in candidates:
        if len(urls) >= max_urls:
            break
        cleaned = candidate.strip().rstrip(".,;:!?'\"")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        urls.append(cleaned)
    return urls

