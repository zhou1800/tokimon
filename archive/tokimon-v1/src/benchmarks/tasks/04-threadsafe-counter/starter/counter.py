"""Counter implementation."""

from __future__ import annotations

import time


class Counter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        current = self.value
        time.sleep(0)
        self.value = current + 1

    def get(self) -> int:
        return self.value
