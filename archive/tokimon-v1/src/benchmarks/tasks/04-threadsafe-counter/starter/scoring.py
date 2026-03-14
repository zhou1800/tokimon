"""Intermediate scoring for threadsafe-counter."""

from __future__ import annotations

import threading

from counter import Counter


def score() -> float:
    counter = Counter()
    threads = [threading.Thread(target=lambda: [counter.increment() for _ in range(100)]) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 1.0 if counter.get() == 500 else 0.0
