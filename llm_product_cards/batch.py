import time
import asyncio
from typing import TypeVar
from collections.abc import Awaitable, Callable, Iterable

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrent(
    items: Iterable[T],
    concurrency: int,
    process_one: Callable[[T], Awaitable[R]],
) -> tuple[list[R], float]:
    """Run process_one over all items with at most `concurrency` in flight.

    Returns the results in input order and the wall time of the whole batch.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await process_one(item)

    started = time.perf_counter()
    results = await asyncio.gather(*(guarded(item) for item in items))
    return list(results), time.perf_counter() - started