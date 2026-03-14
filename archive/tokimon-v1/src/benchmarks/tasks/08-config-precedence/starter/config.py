"""Configuration loader."""

from __future__ import annotations


def load_config(defaults: dict, file_config: dict, env_config: dict) -> dict:
    merged = defaults.copy()
    merged.update(env_config)
    merged.update(file_config)
    return merged
