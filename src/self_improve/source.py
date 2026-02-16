"""Helpers for reading optional self-improvement inputs (URL, file, or inline text)."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


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
    parsed = _validate_url(url)
    req = urllib.request.Request(parsed.geturl(), headers={"User-Agent": "tokimon-self-improve"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
    content = data.decode(errors="replace")
    return InputPayload(kind="url", ref=parsed.geturl(), content=content)


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


def _validate_url(url: str) -> urllib.parse.ParseResult:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http/https urls are allowed")
    if not parsed.hostname:
        raise ValueError("url hostname missing")
    host = parsed.hostname.strip().lower()
    if host in {"localhost"}:
        raise ValueError("localhost is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("private or local IPs are not allowed")
    return parsed
