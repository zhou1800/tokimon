"""Intermediate scoring for dedupe-sorted."""

from __future__ import annotations

from dedupe import dedupe_sorted


def score() -> float:
    result = dedupe_sorted([1, 1, 2, 2, 3])
    return 1.0 if result == [1, 2, 3] else 0.0
