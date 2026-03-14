"""Intermediate scoring for expression-eval."""

from __future__ import annotations

from evaluator import evaluate


def score() -> float:
    score = 0.0
    if evaluate("1 + 2") == 3:
        score += 0.5
    if evaluate("1 + 2 * 3") == 7:
        score += 0.5
    return score
