"""Mem Palace package."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ("MemPalace",)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".mem_palace", __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
