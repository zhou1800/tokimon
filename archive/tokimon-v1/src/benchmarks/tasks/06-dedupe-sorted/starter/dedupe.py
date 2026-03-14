"""Remove duplicates from a sorted list."""

from __future__ import annotations


def dedupe_sorted(items: list[int], ops=None) -> list[int]:
    result: list[int] = []
    for item in items:
        duplicate = False
        for existing in result:
            if ops:
                ops()
            if existing == item:
                duplicate = True
                break
        if not duplicate:
            result.append(item)
    return result
