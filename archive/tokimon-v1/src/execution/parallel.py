"""Async parallel execution with concurrency control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


@dataclass
class ConcurrencyConfig:
    max_concurrency: int = 8
    timeout_seconds: float | None = None


class AsyncExecutor:
    def __init__(self, config: ConcurrencyConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def run(self, tasks: list[Callable[[], Awaitable[T]]]) -> list[T]:
        results: list[T] = []
        awaitables = [self._wrap(task) for task in tasks]
        for future in asyncio.as_completed(awaitables):
            result = await future
            results.append(result)
        return results

    async def _wrap(self, task: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            if self.config.timeout_seconds:
                return await asyncio.wait_for(task(), timeout=self.config.timeout_seconds)
            return await task()
