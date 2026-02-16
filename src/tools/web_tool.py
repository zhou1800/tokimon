"""WebTool fetches web content for research with basic safety limits."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.parse
import urllib.request
from typing import Any

from .base import ToolResult


class WebTool:
    name = "web"

    def fetch(self, url: str, max_bytes: int = 512_000, timeout_s: float = 15.0) -> ToolResult:
        start = time.perf_counter()
        try:
            parsed = _validate_url(url)
            req = urllib.request.Request(parsed.geturl(), headers={"User-Agent": "tokimon-web-tool"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read(max_bytes + 1)
                content_type = resp.headers.get("Content-Type")
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]
            content = raw.decode(errors="replace")
            return ToolResult(
                ok=True,
                summary="fetched",
                data={
                    "url": parsed.geturl(),
                    "content_type": content_type,
                    "truncated": truncated,
                    "content": content,
                },
                elapsed_ms=_elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                summary="fetch failed",
                data={"url": url},
                elapsed_ms=_elapsed_ms(start),
                error=str(exc),
            )

    def search(self, query: str, max_results: int = 5, timeout_s: float = 15.0) -> ToolResult:
        start = time.perf_counter()
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            max_results = int(max_results)
            if max_results <= 0:
                return ToolResult(ok=True, summary="no results requested", data={"query": query, "results": []}, elapsed_ms=_elapsed_ms(start))
            api_url = _duckduckgo_api_url(query.strip())
            parsed = _validate_url(api_url)
            req = urllib.request.Request(parsed.geturl(), headers={"User-Agent": "tokimon-web-tool"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read(2_000_000)
            payload = json.loads(raw.decode(errors="replace"))
            results = _extract_duckduckgo_results(payload, limit=max_results)
            return ToolResult(
                ok=True,
                summary="search ok",
                data={"query": query, "results": results},
                elapsed_ms=_elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                summary="search failed",
                data={"query": query},
                elapsed_ms=_elapsed_ms(start),
                error=str(exc),
            )


def _duckduckgo_api_url(query: str) -> str:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
        }
    )
    return f"https://api.duckduckgo.com/?{params}"


def _extract_duckduckgo_results(payload: dict[str, Any], *, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    def add_result(url: Any, text: Any) -> None:
        if len(results) >= limit:
            return
        if not isinstance(url, str) or not isinstance(text, str):
            return
        url = url.strip()
        text = text.strip()
        if not url or not text:
            return
        results.append({"url": url, "text": text})

    related = payload.get("RelatedTopics", [])
    if isinstance(related, list):
        stack: list[Any] = list(related)
        while stack and len(results) < limit:
            item = stack.pop(0)
            if not isinstance(item, dict):
                continue
            if "FirstURL" in item and "Text" in item:
                add_result(item.get("FirstURL"), item.get("Text"))
                continue
            topics = item.get("Topics")
            if isinstance(topics, list):
                stack.extend(topics)
    return results


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


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
