from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Sequence, TypeVar


TBatch = TypeVar("TBatch")
TResult = TypeVar("TResult")


def build_fixed_batches(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


async def run_parallel_batches(
    *,
    batches: Sequence[TBatch],
    parallel_requests: int,
    process_batch: Callable[[int, TBatch], Awaitable[TResult]],
    on_batch_completed: Callable[[int, TBatch, TResult], Any],
) -> None:
    if parallel_requests <= 0:
        raise ValueError("parallel_requests must be greater than 0")

    sem = asyncio.Semaphore(parallel_requests)

    async def wrapped_process_batch(batch_index: int, batch: TBatch) -> tuple[int, TBatch, TResult]:
        async with sem:
            result = await process_batch(batch_index, batch)
            return batch_index, batch, result

    tasks = [
        asyncio.create_task(wrapped_process_batch(batch_index, batch))
        for batch_index, batch in enumerate(batches)
    ]

    for finished in asyncio.as_completed(tasks):
        batch_index, batch, result = await finished
        callback_result = on_batch_completed(batch_index, batch, result)
        if inspect.isawaitable(callback_result):
            await callback_result


__all__ = [
    "build_fixed_batches",
    "run_parallel_batches",
]
