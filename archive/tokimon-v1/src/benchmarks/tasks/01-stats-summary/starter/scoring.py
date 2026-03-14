"""Intermediate scoring for stats-summary."""

from __future__ import annotations

from stats import mean, variance


def score() -> float:
    score = 0.0
    if mean([1.0, 2.0, 3.0]) == 2.0:
        score += 0.5
    if round(variance([1.0, 2.0, 3.0]), 2) == 1.0:
        score += 0.5
    return score
