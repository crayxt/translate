from __future__ import annotations

import argparse
import math
from typing import Tuple

DEFAULT_BATCH_SIZE = 50
DEFAULT_PARALLEL_REQUESTS = 1
MIN_ITEMS_PER_WORKER = 50
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")


def add_thinking_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Provider reasoning/thinking level (default: provider/model default)",
    )


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


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PARALLEL_REQUESTS",
    "MIN_ITEMS_PER_WORKER",
    "THINKING_LEVEL_CHOICES",
    "add_thinking_level_argument",
    "resolve_runtime_limits",
]
