"""Intermediate scoring for http-client-refactor."""

from __future__ import annotations

try:
    from client import request
    from service import fetch_user
except Exception:
    request = None
    fetch_user = None


def score() -> float:
    score = 0.0
    if request is not None:
        result = request("/health", {"ok": True})
        if result.get("timeout") == 5:
            score += 0.5
    if fetch_user is not None:
        result = fetch_user(7)
        if result.get("endpoint") == "/users/7":
            score += 0.5
    return score
