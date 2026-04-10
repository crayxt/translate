from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar

from core.providers import TranslationProvider


TBatch = TypeVar("TBatch")
TResult = TypeVar("TResult")
TItem = TypeVar("TItem")
TValue = TypeVar("TValue")


def build_fixed_batches(items: Sequence[TItem], batch_size: int) -> list[list[TItem]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def build_indexed_batch_map(
    batch: Sequence[TItem],
    item_builder: Callable[[TItem], TValue],
    *,
    key_builder: Callable[[int, TItem], str] | None = None,
) -> dict[str, TValue]:
    build_key = key_builder or (lambda index, _item: str(index))
    return {
        build_key(index, item): item_builder(item)
        for index, item in enumerate(batch)
    }


def find_missing_index_keys(total_items: int, results_by_id: Mapping[str, object]) -> list[int]:
    return [
        index
        for index in range(total_items)
        if str(index) not in results_by_id
    ]


def merge_mapping_results(
    base: Mapping[str, TValue],
    extra: Mapping[str, TValue],
) -> dict[str, TValue]:
    merged = dict(base)
    merged.update(extra)
    return merged


async def _maybe_await(value: TValue | Awaitable[TValue]) -> TValue:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_parallel_batches(
    *,
    batches: Sequence[TBatch],
    parallel_requests: int,
    process_batch: Callable[[int, TBatch], Awaitable[TResult]],
    on_batch_completed: Callable[[int, TBatch, TResult], object | Awaitable[object]],
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
        await _maybe_await(callback_result)


async def run_model_batches(
    *,
    batches: Sequence[TBatch],
    parallel_requests: int,
    provider: TranslationProvider,
    client: Any,
    model: str,
    config: Any,
    max_attempts: int,
    build_contents: Callable[[int, TBatch], Any],
    parse_response: Callable[[Any], TResult],
    on_batch_completed: Callable[[int, TBatch, TResult], object | Awaitable[object]],
    build_batch_label: Callable[[int], str],
    find_missing_indices: Callable[[TBatch, TResult], list[int]] | None = None,
    build_retry_contents: Callable[[int, TBatch, list[int]], Any] | None = None,
    build_retry_label: Callable[[int], str] | None = None,
    retry_max_attempts: int | Callable[[int], int] = 3,
    merge_retry_result: Callable[[TResult, TResult], TResult] | None = None,
    on_missing_indices: Callable[[int, TBatch, list[int]], object | Awaitable[object]] | None = None,
    on_retry_error: Callable[[int, TBatch, list[int], Exception], object | Awaitable[object]] | None = None,
) -> None:
    if find_missing_indices is not None:
        if build_retry_contents is None or build_retry_label is None or merge_retry_result is None:
            raise ValueError(
                "Retry support requires build_retry_contents, build_retry_label, and merge_retry_result."
            )

    async def process_batch(batch_index: int, batch: TBatch) -> TResult:
        response = await provider.generate_with_retry(
            client=client,
            model=model,
            contents=build_contents(batch_index, batch),
            batch_label=build_batch_label(batch_index),
            max_attempts=max_attempts,
            config=config,
        )
        result = parse_response(response)

        if find_missing_indices is None:
            return result

        missing_indices = find_missing_indices(batch, result)
        if not missing_indices:
            return result

        if on_missing_indices is not None:
            await _maybe_await(on_missing_indices(batch_index, batch, missing_indices))

        retry_attempts = (
            retry_max_attempts(batch_index)
            if callable(retry_max_attempts)
            else retry_max_attempts
        )
        try:
            retry_response = await provider.generate_with_retry(
                client=client,
                model=model,
                contents=build_retry_contents(batch_index, batch, missing_indices),
                batch_label=build_retry_label(batch_index),
                max_attempts=retry_attempts,
                config=config,
            )
            retry_result = parse_response(retry_response)
            return merge_retry_result(result, retry_result)
        except Exception as exc:
            if on_retry_error is not None:
                await _maybe_await(on_retry_error(batch_index, batch, missing_indices, exc))
            return result

    await run_parallel_batches(
        batches=batches,
        parallel_requests=parallel_requests,
        process_batch=process_batch,
        on_batch_completed=on_batch_completed,
    )


__all__ = [
    "build_fixed_batches",
    "build_indexed_batch_map",
    "find_missing_index_keys",
    "merge_mapping_results",
    "run_model_batches",
    "run_parallel_batches",
]
