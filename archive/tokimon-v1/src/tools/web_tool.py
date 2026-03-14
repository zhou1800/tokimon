"""WebTool fetches web content for research with basic safety limits."""

from __future__ import annotations

from collections import deque
import ipaddress
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from .base import ToolResult, elapsed_ms


class WebTool:
    name = "web"

    def __init__(
        self,
        *,
        org_allowlist: list[str] | None = None,
        request_allowlist: list[str] | None = None,
        domain_secrets: dict[str, dict[str, str]] | None = None,
    ) -> None:
        env_org_allowlist = _read_env_domain_csv("TOKIMON_WEB_ORG_ALLOWLIST")
        env_request_allowlist = _read_env_domain_csv("TOKIMON_WEB_REQUEST_ALLOWLIST")
        env_domain_secrets, env_domain_secrets_error = _read_env_domain_secrets("TOKIMON_WEB_DOMAIN_SECRETS_JSON")

        self._org_allowlist = _normalize_allowlist(org_allowlist if org_allowlist is not None else env_org_allowlist)
        self._request_allowlist = _normalize_allowlist(
            request_allowlist if request_allowlist is not None else env_request_allowlist
        )
        self._domain_secrets = _normalize_domain_secrets(domain_secrets if domain_secrets is not None else env_domain_secrets)
        self._config_error = env_domain_secrets_error or _validate_network_policy(
            org_allowlist=self._org_allowlist,
            request_allowlist=self._request_allowlist,
        )

    def fetch(self, url: str, max_bytes: int = 512_000, timeout_s: float = 15.0) -> ToolResult:
        start = time.perf_counter()
        try:
            if self._config_error:
                raise ValueError(self._config_error)
            parsed = _validate_url(url)
            host = parsed.hostname or ""
            _enforce_allowlist(host, _effective_allowlist(self._org_allowlist, self._request_allowlist))
            headers = {"User-Agent": "tokimon-web-tool"}
            secret_headers = _secret_headers_for_host(host, self._domain_secrets)
            if secret_headers:
                headers.update(secret_headers)
            req = urllib.request.Request(parsed.geturl(), headers=headers)
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
                elapsed_ms=elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                summary="fetch failed",
                data={"url": url},
                elapsed_ms=elapsed_ms(start),
                error=str(exc),
            )

    def search(self, query: str, max_results: int = 5, timeout_s: float = 15.0) -> ToolResult:
        start = time.perf_counter()
        try:
            if self._config_error:
                raise ValueError(self._config_error)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            max_results = int(max_results)
            if max_results <= 0:
                return ToolResult(ok=True, summary="no results requested", data={"query": query, "results": []}, elapsed_ms=elapsed_ms(start))
            api_url = _duckduckgo_api_url(query.strip())
            parsed = _validate_url(api_url)
            host = parsed.hostname or ""
            _enforce_allowlist(host, _effective_allowlist(self._org_allowlist, self._request_allowlist))
            headers = {"User-Agent": "tokimon-web-tool"}
            secret_headers = _secret_headers_for_host(host, self._domain_secrets)
            if secret_headers:
                headers.update(secret_headers)
            req = urllib.request.Request(parsed.geturl(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read(2_000_000)
            payload = json.loads(raw.decode(errors="replace"))
            results = _extract_duckduckgo_results(payload, limit=max_results)
            return ToolResult(
                ok=True,
                summary="search ok",
                data={"query": query, "results": results},
                elapsed_ms=elapsed_ms(start),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                summary="search failed",
                data={"query": query},
                elapsed_ms=elapsed_ms(start),
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
        queue: deque[Any] = deque(related)
        while queue and len(results) < limit:
            item = queue.popleft()
            if not isinstance(item, dict):
                continue
            if "FirstURL" in item and "Text" in item:
                add_result(item.get("FirstURL"), item.get("Text"))
                continue
            topics = item.get("Topics")
            if isinstance(topics, list):
                queue.extend(topics)
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


_ENV_VAR_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _read_env_domain_csv(var_name: str) -> list[str] | None:
    raw = os.environ.get(var_name)
    if not raw or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _read_env_domain_secrets(var_name: str) -> tuple[dict[str, dict[str, str]] | None, str | None]:
    raw = os.environ.get(var_name)
    if not raw or not raw.strip():
        return None, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid domain secrets json: {exc}"
    if not isinstance(parsed, dict):
        return None, "invalid domain secrets json: expected object"
    secrets: dict[str, dict[str, str]] = {}
    for domain, headers in parsed.items():
        if not isinstance(domain, str) or not isinstance(headers, dict):
            continue
        header_map: dict[str, str] = {}
        for header_name, template in headers.items():
            if not isinstance(header_name, str) or not isinstance(template, str):
                continue
            if header_name.strip():
                header_map[header_name.strip()] = template
        if header_map:
            secrets[domain] = header_map
    return secrets, None


def _normalize_allowlist(domains: list[str] | None) -> tuple[str, ...] | None:
    if not domains:
        return None
    normalized: list[str] = []
    for entry in domains:
        domain = _normalize_domain_entry(entry)
        if domain and domain not in normalized:
            normalized.append(domain)
    return tuple(normalized) if normalized else None


def _normalize_domain_entry(value: str) -> str | None:
    raw = (value or "").strip().lower()
    if not raw:
        return None
    if "://" in raw:
        parsed = urllib.parse.urlparse(raw)
        if parsed.hostname:
            raw = parsed.hostname.strip().lower()
        else:
            return None
    raw = raw.strip().lstrip(".").rstrip(".")
    return raw or None


def _validate_network_policy(
    *,
    org_allowlist: tuple[str, ...] | None,
    request_allowlist: tuple[str, ...] | None,
) -> str | None:
    if org_allowlist and request_allowlist:
        for requested in request_allowlist:
            if not any(_domain_matches(requested, allowed) for allowed in org_allowlist):
                return "request allowlist must be a subset of org allowlist"
    return None


def _effective_allowlist(
    org_allowlist: tuple[str, ...] | None,
    request_allowlist: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    return request_allowlist or org_allowlist


def _enforce_allowlist(host: str, allowlist: tuple[str, ...] | None) -> None:
    if not allowlist:
        return
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("url hostname missing")
    if not any(_domain_matches(host, allowed) for allowed in allowlist):
        raise ValueError("domain not in allowlist")


def _domain_matches(host: str, domain: str) -> bool:
    host = (host or "").strip().lower().rstrip(".")
    domain = (domain or "").strip().lower().lstrip(".").rstrip(".")
    if not host or not domain:
        return False
    if host == domain:
        return True
    return host.endswith(f".{domain}")


def _normalize_domain_secrets(raw: dict[str, dict[str, str]] | None) -> dict[str, dict[str, str]]:
    if not raw:
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for domain, headers in raw.items():
        if not isinstance(domain, str) or not isinstance(headers, dict):
            continue
        norm_domain = _normalize_domain_entry(domain)
        if not norm_domain:
            continue
        header_map: dict[str, str] = {}
        for header_name, template in headers.items():
            if not isinstance(header_name, str) or not isinstance(template, str):
                continue
            if header_name.strip():
                header_map[header_name.strip()] = template
        if header_map:
            normalized[norm_domain] = header_map
    return normalized


def _secret_headers_for_host(host: str, domain_secrets: dict[str, dict[str, str]]) -> dict[str, str]:
    if not domain_secrets:
        return {}
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return {}

    best_domain = ""
    best_headers: dict[str, str] | None = None
    for domain, headers in domain_secrets.items():
        if not isinstance(domain, str) or not isinstance(headers, dict):
            continue
        if _domain_matches(host, domain) and len(domain) > len(best_domain):
            best_domain = domain
            best_headers = {str(k): str(v) for k, v in headers.items()}

    if not best_headers:
        return {}

    resolved: dict[str, str] = {}
    for name, template in best_headers.items():
        resolved[name] = _expand_env_template(template)
    return resolved


def _expand_env_template(template: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"missing environment variable: {var_name}")
        return value

    return _ENV_VAR_PATTERN.sub(replace, template or "")
