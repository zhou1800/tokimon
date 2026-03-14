"""HTTP client."""

from __future__ import annotations


def send_request(url: str, payload: dict) -> dict:
    return {"url": url, "payload": payload}
