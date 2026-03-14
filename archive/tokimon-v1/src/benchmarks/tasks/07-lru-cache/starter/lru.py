"""LRU cache."""

from __future__ import annotations


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.store: dict[str, int] = {}

    def get(self, key: str) -> int | None:
        return self.store.get(key)

    def put(self, key: str, value: int) -> None:
        if len(self.store) >= self.capacity:
            self.store.pop(next(iter(self.store)))
        self.store[key] = value
