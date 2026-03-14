"""Intermediate scoring for interval-merge."""

from __future__ import annotations

from intervals import merge_intervals


def score() -> float:
    score = 0.0
    if merge_intervals([(1, 3), (2, 4)]) == [(1, 4)]:
        score += 0.5
    if merge_intervals([(5, 6), (1, 2), (2, 4)]) == [(1, 4), (5, 6)]:
        score += 0.5
    return score
