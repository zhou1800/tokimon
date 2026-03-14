"""Intermediate scoring for config-precedence."""

from __future__ import annotations

from config import load_config


def score() -> float:
    defaults = {"a": 1, "b": 2}
    file_config = {"b": 3, "c": 4}
    env_config = {"c": 5}
    result = load_config(defaults, file_config, env_config)
    return 1.0 if result == {"a": 1, "b": 3, "c": 5} else 0.0
