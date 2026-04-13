import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.review_flow import (
    PreparedReviewRun,
    ReviewBatchRunnerSpec,
    ReviewRetryRunnerSpec,
    build_retry_review_batch_messages,
    build_review_batch_messages,
    build_review_startup_entries,
    find_missing_mapping_indices,
    limit_items,
    merge_mapping_review_results,
    prepare_review_run,
    prepare_review_batches,
    print_review_startup,
)


class ReviewFlowSmokeTests(unittest.TestCase):
    def test_build_review_startup_entries_includes_common_fields(self):
        provider = SimpleNamespace(name="gemini", supports_flex_mode=True)
        resource_context = SimpleNamespace(vocabulary_source="file:vocab.txt", rules_source="file:rules.md")

        entries = build_review_startup_entries(
            provider=provider,
            model_name="gemini-test",
            flex_mode=True,
            thinking_level="medium",
            parallel_requests=2,
            batch_size=50,
            limits_mode="check defaults",
            resource_context=resource_context,
            item_label="Review items",
            item_count=12,
            extra_entries=(("Probe limit", "none"),),
        )

        self.assertEqual(
            entries,
            (
                ("Provider", "gemini"),
                ("Model", "gemini-test"),
                ("Flex mode", "yes"),
                ("Thinking level", "medium"),
                ("Parallel requests", 2),
                ("Batch size", 50),
                ("Limits mode", "check defaults"),
                ("Vocabulary source", "file:vocab.txt"),
                ("Rules source", "file:rules.md"),
                ("Probe limit", "none"),
                ("Review items", 12),
            ),
        )

    def test_prepare_review_run_resolves_runtime_limits_and_batches(self):
        runtime_context = SimpleNamespace(provider="provider", client="client", resources="resources")

        prepared = prepare_review_run(
            items=["a", "b", "c"],
            provider_name="gemini",
            target_lang="kk",
            flex_mode=True,
            explicit_vocab_path="vocab.txt",
            explicit_rules_path="rules.md",
            inline_rules="Rule A",
            batch_size_arg=10,
            parallel_arg=2,
            default_batch_size=50,
            default_parallel=1,
            label="check",
            build_task_runtime_context_fn=lambda **kwargs: runtime_context,
        )

        self.assertIsInstance(prepared, PreparedReviewRun)
        self.assertIs(prepared.runtime_context, runtime_context)
        self.assertEqual(prepared.items, ["a", "b", "c"])
        self.assertEqual(prepared.batch_size, 10)
        self.assertEqual(prepared.parallel_requests, 2)
        self.assertEqual(prepared.limits_mode, "explicit")
        self.assertEqual(prepared.batch_plan.batches, [["a", "b", "c"]])
        self.assertEqual(prepared.batch_plan.batch_start_indices, {0: 0})
        self.assertEqual(prepared.batch_count, 1)

    def test_print_review_startup_delegates_to_task_runtime_formatter(self):
        provider = SimpleNamespace(name="gemini", supports_flex_mode=True)
        resource_context = SimpleNamespace(vocabulary_source="file:vocab.txt", rules_source="file:rules.md")

        with patch("core.review_flow.print_startup_configuration") as print_mock:
            print_review_startup(
                provider=provider,
                model_name="gemini-test",
                flex_mode=True,
                thinking_level="medium",
                parallel_requests=2,
                batch_size=50,
                limits_mode="check defaults",
                resource_context=resource_context,
                item_label="Review items",
                item_count=12,
                extra_entries=(("Probe limit", "none"),),
            )

        print_mock.assert_called_once_with(
            ("Provider", "gemini"),
            ("Model", "gemini-test"),
            ("Flex mode", "yes"),
            ("Thinking level", "medium"),
            ("Parallel requests", 2),
            ("Batch size", 50),
            ("Limits mode", "check defaults"),
            ("Vocabulary source", "file:vocab.txt"),
            ("Rules source", "file:rules.md"),
            ("Probe limit", "none"),
            ("Review items", 12),
        )

    def test_prepare_review_batches_tracks_global_start_indices(self):
        plan = prepare_review_batches(["a", "b", "c", "d", "e"], 2)

        self.assertEqual(plan.batches, [["a", "b"], ["c", "d"], ["e"]])
        self.assertEqual(plan.batch_start_indices, {0: 0, 1: 2, 2: 4})

    def test_build_review_batch_messages_uses_string_indices(self):
        payload = build_review_batch_messages(
            ["open", "save"],
            lambda item: {"source": item},
        )

        self.assertEqual(
            payload,
            {
                "0": {"source": "open"},
                "1": {"source": "save"},
            },
        )

    def test_build_retry_review_batch_messages_keeps_original_indices(self):
        payload = build_retry_review_batch_messages(
            ["open", "save", "close"],
            [0, 2],
            lambda item: {"source": item},
        )

        self.assertEqual(
            payload,
            {
                "0": {"source": "open"},
                "2": {"source": "close"},
            },
        )

    def test_find_missing_mapping_indices_uses_batch_length(self):
        missing = find_missing_mapping_indices(["a", "b", "c"], {"0": object(), "2": object()})

        self.assertEqual(missing, [1])

    def test_merge_mapping_review_results_updates_existing_keys(self):
        merged = merge_mapping_review_results(
            {"0": "open", "1": "save"},
            {"1": "save-new", "2": "close"},
        )

        self.assertEqual(
            merged,
            {
                "0": "open",
                "1": "save-new",
                "2": "close",
            },
        )

    def test_limit_items_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            limit_items(["a"], 0)

    def test_review_batch_runner_spec_stores_callbacks(self):
        spec = ReviewBatchRunnerSpec(
            build_contents=lambda *_args: {"messages": {}},
            parse_response=lambda response: response,
            on_batch_completed=lambda *_args: None,
            build_batch_label=lambda index: f"batch {index}",
        )

        self.assertEqual(spec.build_contents(0, []), {"messages": {}})
        self.assertEqual(spec.parse_response({"ok": True}), {"ok": True})
        self.assertEqual(spec.build_batch_label(1), "batch 1")
        self.assertIsNone(spec.retry_spec)

    def test_review_retry_runner_spec_groups_retry_hooks(self):
        retry_spec = ReviewRetryRunnerSpec(
            find_missing_indices=lambda _batch, _result: [1],
            build_retry_contents=lambda *_args: {"messages": {"1": {"source": "Save"}}},
            build_retry_label=lambda index: f"batch {index} missing",
            merge_retry_result=lambda base, extra: {**base, **extra},
        )

        self.assertEqual(retry_spec.find_missing_indices([], {}), [1])
        self.assertEqual(retry_spec.build_retry_contents(0, [], [1]), {"messages": {"1": {"source": "Save"}}})
        self.assertEqual(retry_spec.build_retry_label(1), "batch 1 missing")


if __name__ == "__main__":
    unittest.main()
