"""Intermediate scoring for lru-cache."""

from __future__ import annotations

from lru import LRUCache


def score() -> float:
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    return 1.0 if cache.get("b") is None else 0.0
