from __future__ import annotations

from typing import Any, Tuple

from core.formats import EntryStatus
from core.runtime import resolve_runtime_limits


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
