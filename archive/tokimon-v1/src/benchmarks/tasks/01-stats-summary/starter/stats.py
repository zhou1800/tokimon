"""Statistics helpers."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)
