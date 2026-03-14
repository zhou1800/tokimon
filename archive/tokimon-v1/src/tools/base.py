"""Base tool abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any]
    elapsed_ms: float
    error: str | None = None


class ToolError(Exception):
    pass


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
