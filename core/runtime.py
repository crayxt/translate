from __future__ import annotations

import argparse
import asyncio
import math
from typing import Any, Tuple

from google import genai
from google.genai import types as genai_types

DEFAULT_BATCH_SIZE = 1000
DEFAULT_PARALLEL_REQUESTS = 10
MIN_ITEMS_PER_WORKER = 50
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")


def add_thinking_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Gemini thinking level (default: provider/model default)",
    )


def build_thinking_config(thinking_level: str | None) -> genai_types.ThinkingConfig | None:
    if thinking_level is None:
        return None

    normalized = str(thinking_level).strip().lower()
    thinking_level_map = {
        "minimal": genai_types.ThinkingLevel.MINIMAL,
        "low": genai_types.ThinkingLevel.LOW,
        "medium": genai_types.ThinkingLevel.MEDIUM,
        "high": genai_types.ThinkingLevel.HIGH,
    }
    resolved = thinking_level_map.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported thinking level: {thinking_level!r}. "
            f"Expected one of: {', '.join(THINKING_LEVEL_CHOICES)}"
        )
    return genai_types.ThinkingConfig(thinking_level=resolved)


def resolve_runtime_limits(
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
) -> Tuple[int, int, str]:
    if batch_size_arg is not None and batch_size_arg <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if parallel_arg is not None and parallel_arg <= 0:
        raise ValueError("--parallel-requests must be greater than 0")

    if batch_size_arg is None and parallel_arg is None:
        return DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL_REQUESTS, "defaults"
    if batch_size_arg is not None and parallel_arg is not None:
        return batch_size_arg, parallel_arg, "explicit"
    if batch_size_arg is not None:
        derived_parallel = max(
            1,
            min(DEFAULT_PARALLEL_REQUESTS, math.ceil(total_items / batch_size_arg)),
        )
        return batch_size_arg, derived_parallel, "auto parallel"

    assert parallel_arg is not None
    derived_batch = max(MIN_ITEMS_PER_WORKER, math.ceil(total_items / parallel_arg))
    return derived_batch, parallel_arg, "auto batch"


async def generate_content_async(
    client: genai.Client,
    model: str,
    contents: Any,
    config: genai_types.GenerateContentConfig | None = None,
) -> Any:
    if hasattr(client, "aio") and client.aio:
        return await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=config,
    )


async def generate_with_retry(
    client: genai.Client,
    model: str,
    contents: Any,
    batch_label: str,
    max_attempts: int = 5,
    config: genai_types.GenerateContentConfig | None = None,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return await generate_content_async(client, model, contents, config=config)
        except Exception as exc:
            print(f"\nAPI Error [{batch_label}] (Attempt {attempt}/{max_attempts}): {exc}")
            if attempt == max_attempts:
                raise RuntimeError(f"Aborting [{batch_label}] due to repeated API errors.") from exc
            wait_time = 2 ** attempt
            print(f"Retrying [{batch_label}] in {wait_time}s...")
            await asyncio.sleep(wait_time)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PARALLEL_REQUESTS",
    "MIN_ITEMS_PER_WORKER",
    "THINKING_LEVEL_CHOICES",
    "add_thinking_level_argument",
    "build_thinking_config",
    "generate_content_async",
    "generate_with_retry",
    "resolve_runtime_limits",
]
