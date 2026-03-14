"""Summary statistics."""

from __future__ import annotations

from stats import mean, variance


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "variance": variance(values)}
