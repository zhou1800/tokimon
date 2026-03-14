import asyncio

from execution.parallel import AsyncExecutor, ConcurrencyConfig


def test_parallel_executor_runs_tasks():
    async def task(value: int):
        await asyncio.sleep(0.01)
        return value * 2

    async def run():
        executor = AsyncExecutor(ConcurrencyConfig(max_concurrency=2))
        results = await executor.run([lambda v=value: task(v) for value in range(4)])
        return sorted(results)

    results = asyncio.run(run())
    assert results == [0, 2, 4, 6]
