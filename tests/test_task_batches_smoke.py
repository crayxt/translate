import asyncio
import unittest

from core.task_batches import (
    build_fixed_batches,
    build_indexed_batch_map,
    find_missing_index_keys,
    merge_mapping_results,
    run_model_batches,
    run_parallel_batches,
)


class _DummyProvider:
    def __init__(self, responses):
        self.responses = responses

    async def generate_with_retry(self, *, client, model, contents, batch_label, max_attempts, config):
        return self.responses[batch_label]


class TaskBatchesSmokeTests(unittest.TestCase):
    def test_build_fixed_batches_splits_items(self):
        batches = build_fixed_batches([1, 2, 3, 4, 5], 2)
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    def test_build_indexed_batch_map_supports_custom_keys(self):
        payload = build_indexed_batch_map(
            ["a", "b"],
            lambda value: value.upper(),
            key_builder=lambda index, value: f"{index}:{value}",
        )
        self.assertEqual(payload, {"0:a": "A", "1:b": "B"})

    def test_find_missing_index_keys_detects_missing_string_ids(self):
        self.assertEqual(find_missing_index_keys(4, {"0": "x", "2": "y"}), [1, 3])

    def test_merge_mapping_results_updates_existing_keys(self):
        merged = merge_mapping_results({"0": "A", "1": "B"}, {"1": "B2", "2": "C"})
        self.assertEqual(merged, {"0": "A", "1": "B2", "2": "C"})

    def test_run_parallel_batches_processes_all_batches(self):
        observed: list[tuple[int, list[int], list[int]]] = []

        async def process_batch(batch_index: int, batch: list[int]) -> list[int]:
            if batch_index == 0:
                await asyncio.sleep(0.01)
            return [item * 2 for item in batch]

        async def on_batch_completed(
            batch_index: int,
            batch: list[int],
            result: list[int],
        ) -> None:
            observed.append((batch_index, batch, result))

        asyncio.run(
            run_parallel_batches(
                batches=[[1, 2], [3], [4, 5]],
                parallel_requests=2,
                process_batch=process_batch,
                on_batch_completed=on_batch_completed,
            )
        )

        observed.sort(key=lambda item: item[0])
        self.assertEqual(
            observed,
            [
                (0, [1, 2], [2, 4]),
                (1, [3], [6]),
                (2, [4, 5], [8, 10]),
            ],
        )

    def test_run_model_batches_retries_missing_items_and_merges_results(self):
        observed: list[dict[str, str]] = []
        missing_events: list[list[int]] = []
        retry_errors: list[str] = []
        provider = _DummyProvider(
            {
                "batch 1/1": {"0": "A"},
                "batch 1/1 missing-items": {"1": "B"},
            }
        )

        async def on_batch_completed(
            batch_index: int,
            batch: list[str],
            result: dict[str, str],
        ) -> None:
            observed.append(result)

        asyncio.run(
            run_model_batches(
                batches=[["first", "second"]],
                parallel_requests=1,
                provider=provider,
                client=object(),
                model="test-model",
                config=object(),
                max_attempts=5,
                build_contents=lambda _batch_index, batch: {"items": batch},
                parse_response=lambda response: response,
                on_batch_completed=on_batch_completed,
                build_batch_label=lambda _batch_index: "batch 1/1",
                find_missing_indices=lambda batch, result: find_missing_index_keys(len(batch), result),
                build_retry_contents=lambda _batch_index, batch, missing_indices: {
                    "missing": [batch[index] for index in missing_indices]
                },
                build_retry_label=lambda _batch_index: "batch 1/1 missing-items",
                retry_max_attempts=3,
                merge_retry_result=merge_mapping_results,
                on_missing_indices=lambda _batch_index, _batch, missing_indices: missing_events.append(missing_indices),
                on_retry_error=lambda _batch_index, _batch, _missing_indices, exc: retry_errors.append(str(exc)),
            )
        )

        self.assertEqual(missing_events, [[1]])
        self.assertEqual(retry_errors, [])
        self.assertEqual(observed, [{"0": "A", "1": "B"}])


if __name__ == "__main__":
    unittest.main()
