import asyncio
import unittest

from core.task_batches import build_fixed_batches, run_parallel_batches


class TaskBatchesSmokeTests(unittest.TestCase):
    def test_build_fixed_batches_splits_items(self):
        batches = build_fixed_batches([1, 2, 3, 4, 5], 2)
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

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


if __name__ == "__main__":
    unittest.main()
