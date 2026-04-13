from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Mapping, Sequence, Tuple, TypeVar

from core.providers import TranslationProvider
from core.formats import EntryStatus
from core.runtime import resolve_runtime_limits
from core.task_batches import (
    build_fixed_batches,
    build_indexed_batch_map,
    find_missing_index_keys,
    merge_mapping_results,
    run_model_batches,
)
from core.task_runtime import (
    TaskRuntimeContext,
    build_task_runtime_context,
    print_startup_configuration,
)


TItem = TypeVar("TItem")
TResult = TypeVar("TResult")
TValue = TypeVar("TValue")


@dataclass(slots=True, frozen=True)
class ReviewBatchPlan(Generic[TItem]):
    """Batch list plus stable global start offsets for review items."""

    batches: list[list[TItem]]
    batch_start_indices: dict[int, int]


@dataclass(slots=True, frozen=True)
class PreparedReviewRun(Generic[TItem]):
    """Resolved runtime state for one review-style task run."""

    runtime_context: TaskRuntimeContext
    items: list[TItem]
    batch_size: int
    parallel_requests: int
    limits_mode: str
    batch_plan: ReviewBatchPlan[TItem]

    @property
    def batch_count(self) -> int:
        """Return the number of fixed batches in this prepared run."""
        return len(self.batch_plan.batches)


@dataclass(slots=True, frozen=True)
class ReviewBatchRunnerSpec(Generic[TItem, TResult]):
    """Task-specific hooks needed to execute a review-style batch runner."""

    build_contents: Callable[[int, Sequence[TItem]], Any]
    parse_response: Callable[[Any], TResult]
    on_batch_completed: Callable[[int, Sequence[TItem], TResult], object | Awaitable[object]]
    build_batch_label: Callable[[int], str]
    retry_spec: ReviewRetryRunnerSpec[TItem, TResult] | None = None


@dataclass(slots=True, frozen=True)
class ReviewRetryRunnerSpec(Generic[TItem, TResult]):
    """Optional retry hooks for review tasks that must recover missing item ids."""

    find_missing_indices: Callable[[Sequence[TItem], TResult], list[int]]
    build_retry_contents: Callable[[int, Sequence[TItem], list[int]], Any]
    build_retry_label: Callable[[int], str]
    merge_retry_result: Callable[[TResult, TResult], TResult]
    retry_max_attempts: int | Callable[[int], int] = 3
    on_missing_indices: Callable[[int, Sequence[TItem], list[int]], object | Awaitable[object]] | None = None
    on_retry_error: Callable[[int, Sequence[TItem], list[int], Exception], object | Awaitable[object]] | None = None


def normalize_limits(
    *,
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
    default_batch_size: int,
    default_parallel: int,
    label: str,
) -> Tuple[int, int, str]:
    """Resolve batch limits, applying task defaults when the user omitted both knobs."""
    if batch_size_arg is None and parallel_arg is None:
        batch_size, parallel, _ = resolve_runtime_limits(
            total_items=total_items,
            batch_size_arg=default_batch_size,
            parallel_arg=default_parallel,
        )
        return batch_size, parallel, f"{label} defaults"

    return resolve_runtime_limits(
        total_items=total_items,
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
    )


def has_reviewable_translation(
    entry: Any,
    *,
    plural_texts: list[str] | None = None,
    allow_context_only: bool = False,
) -> bool:
    """Return whether an entry has enough translated content to review."""
    if bool(getattr(entry, "obsolete", False)):
        return False
    if getattr(entry, "status", None) == EntryStatus.SKIPPED:
        return False

    msgid = str(getattr(entry, "msgid", "") or "").strip()
    msgctxt = str(getattr(entry, "msgctxt", "") or "").strip()
    if allow_context_only:
        if not msgid and not msgctxt:
            return False
    elif not msgid:
        return False

    if str(getattr(entry, "msgstr", "") or "").strip():
        return True

    plural_values = plural_texts
    if plural_values is None:
        plural_map = getattr(entry, "msgstr_plural", None)
        if isinstance(plural_map, dict):
            plural_values = [str(value or "") for value in plural_map.values()]
        else:
            plural_values = []
    return any(str(text or "").strip() for text in plural_values)


def limit_items(items: list[Any], num_messages: int | None) -> list[Any]:
    """Apply an optional positive prefix limit to a review item list."""
    if num_messages is None:
        return items
    if num_messages <= 0:
        raise ValueError("--probe/--num-messages must be greater than 0")
    return items[:num_messages]


def prepare_review_batches(items: Sequence[TItem], batch_size: int) -> ReviewBatchPlan[TItem]:
    """Split review items into fixed batches and precompute global start offsets."""
    batches = build_fixed_batches(items, batch_size)
    batch_start_indices: dict[int, int] = {}
    start_index = 0
    for batch_index, batch in enumerate(batches):
        batch_start_indices[batch_index] = start_index
        start_index += len(batch)
    return ReviewBatchPlan(batches=batches, batch_start_indices=batch_start_indices)


def prepare_review_run(
    *,
    items: Sequence[TItem],
    provider_name: str | None,
    target_lang: str,
    flex_mode: bool = False,
    explicit_vocab_path: str | None = None,
    explicit_rules_path: str | None = None,
    inline_rules: str | None = None,
    batch_size_arg: int | None = None,
    parallel_arg: int | None = None,
    default_batch_size: int,
    default_parallel: int,
    label: str,
    build_task_runtime_context_fn: Callable[..., TaskRuntimeContext] = build_task_runtime_context,
    normalize_limits_fn: Callable[..., Tuple[int, int, str]] = normalize_limits,
    prepare_review_batches_fn: Callable[[Sequence[TItem], int], ReviewBatchPlan[TItem]] = prepare_review_batches,
) -> PreparedReviewRun[TItem]:
    """Resolve runtime resources, limits, and batches for a review-style task."""
    item_list = list(items)
    batch_size, parallel_requests, limits_mode = normalize_limits_fn(
        total_items=len(item_list),
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
        default_batch_size=default_batch_size,
        default_parallel=default_parallel,
        label=label,
    )
    runtime_context = build_task_runtime_context_fn(
        provider_name=provider_name,
        target_lang=target_lang,
        flex_mode=flex_mode,
        explicit_vocab_path=explicit_vocab_path,
        explicit_rules_path=explicit_rules_path,
        inline_rules=inline_rules,
    )
    batch_plan = prepare_review_batches_fn(item_list, batch_size)
    return PreparedReviewRun(
        runtime_context=runtime_context,
        items=item_list,
        batch_size=batch_size,
        parallel_requests=parallel_requests,
        limits_mode=limits_mode,
        batch_plan=batch_plan,
    )


def build_review_startup_entries(
    *,
    provider: TranslationProvider,
    model_name: str,
    flex_mode: bool,
    thinking_level: str | None,
    parallel_requests: int,
    batch_size: int,
    limits_mode: str,
    resource_context: Any,
    item_label: str,
    item_count: int,
    extra_entries: Sequence[tuple[str, Any]] = (),
) -> tuple[tuple[str, Any], ...]:
    """Build the common startup configuration rows for review-style tasks."""
    entries = [
        ("Provider", provider.name),
        ("Model", model_name),
        ("Flex mode", "yes" if flex_mode and getattr(provider, "supports_flex_mode", False) else "no"),
        ("Thinking level", thinking_level or "provider default"),
        ("Parallel requests", parallel_requests),
        ("Batch size", batch_size),
        ("Limits mode", limits_mode),
        ("Vocabulary source", getattr(resource_context, "vocabulary_source", "none")),
        ("Rules source", getattr(resource_context, "rules_source", None) or "none"),
        *list(extra_entries),
        (item_label, item_count),
    ]
    return tuple(entries)


def print_review_startup(
    *,
    provider: TranslationProvider,
    model_name: str,
    flex_mode: bool,
    thinking_level: str | None,
    parallel_requests: int,
    batch_size: int,
    limits_mode: str,
    resource_context: Any,
    item_label: str,
    item_count: int,
    extra_entries: Sequence[tuple[str, Any]] = (),
) -> None:
    """Print the common startup configuration for a review-style task."""
    print_startup_configuration(
        *build_review_startup_entries(
            provider=provider,
            model_name=model_name,
            flex_mode=flex_mode,
            thinking_level=thinking_level,
            parallel_requests=parallel_requests,
            batch_size=batch_size,
            limits_mode=limits_mode,
            resource_context=resource_context,
            item_label=item_label,
            item_count=item_count,
            extra_entries=extra_entries,
        ),
    )


def build_review_batch_messages(
    batch: Sequence[TItem],
    item_builder: Callable[[TItem], TValue],
) -> dict[str, TValue]:
    """Build the standard string-indexed payload map for one review batch."""
    return build_indexed_batch_map(batch, item_builder)


def build_retry_review_batch_messages(
    batch: Sequence[TItem],
    missing_indices: Sequence[int],
    item_builder: Callable[[TItem], TValue],
) -> dict[str, TValue]:
    """Build a retry payload map keyed by original item index strings."""
    return build_indexed_batch_map(
        missing_indices,
        lambda item_index: item_builder(batch[item_index]),
        key_builder=lambda _retry_index, item_index: str(item_index),
    )


def find_missing_mapping_indices(
    batch: Sequence[Any],
    results_by_id: Mapping[str, object],
) -> list[int]:
    """Find missing stringified item indices in a mapping-shaped model response."""
    return find_missing_index_keys(len(batch), results_by_id)


def merge_mapping_review_results(
    base: Mapping[str, TValue],
    extra: Mapping[str, TValue],
) -> dict[str, TValue]:
    """Merge initial and retry mapping results for review tasks."""
    return merge_mapping_results(base, extra)


async def run_review_batches(
    *,
    batches: Sequence[Sequence[TItem]],
    parallel_requests: int,
    provider: TranslationProvider,
    client: Any,
    model: str,
    config: Any,
    max_attempts: int,
    runner_spec: ReviewBatchRunnerSpec[TItem, TResult],
) -> None:
    """Run a review-style model task over batches, with optional missing-item retry support."""
    retry_spec = runner_spec.retry_spec
    await run_model_batches(
        batches=batches,
        parallel_requests=parallel_requests,
        provider=provider,
        client=client,
        model=model,
        config=config,
        max_attempts=max_attempts,
        build_contents=runner_spec.build_contents,
        parse_response=runner_spec.parse_response,
        on_batch_completed=runner_spec.on_batch_completed,
        build_batch_label=runner_spec.build_batch_label,
        find_missing_indices=retry_spec.find_missing_indices if retry_spec is not None else None,
        build_retry_contents=retry_spec.build_retry_contents if retry_spec is not None else None,
        build_retry_label=retry_spec.build_retry_label if retry_spec is not None else None,
        retry_max_attempts=retry_spec.retry_max_attempts if retry_spec is not None else 3,
        merge_retry_result=retry_spec.merge_retry_result if retry_spec is not None else None,
        on_missing_indices=retry_spec.on_missing_indices if retry_spec is not None else None,
        on_retry_error=retry_spec.on_retry_error if retry_spec is not None else None,
    )
