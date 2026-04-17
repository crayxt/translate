import unittest
from unittest.mock import patch

from core import runtime
from core.task_resources import TaskResourceContext
from core.task_runtime import build_task_runtime_context, print_startup_configuration


class _DummyProvider:
    def __init__(self):
        self.name = "dummy"
        self.last_flex_mode = None

    def create_client_from_env(self, *, flex_mode: bool = False):
        self.last_flex_mode = flex_mode
        return object()


class TaskRuntimeSmokeTests(unittest.TestCase):
    def test_resolve_runtime_limits_uses_global_defaults(self):
        batch_size, parallel_requests, mode = runtime.resolve_runtime_limits(
            total_items=500,
            batch_size_arg=None,
            parallel_arg=None,
        )

        self.assertEqual(batch_size, 50)
        self.assertEqual(parallel_requests, 1)
        self.assertEqual(mode, "defaults")

    def test_build_task_runtime_context_builds_provider_client_and_resources(self):
        resources = TaskResourceContext(vocabulary_source="file:glossary.po", rules_source="file:rules.md")
        provider = _DummyProvider()

        context = build_task_runtime_context(
            provider_name="dummy",
            target_lang="kk",
            flex_mode=True,
            explicit_vocab_path="glossary.po",
            explicit_rules_path="rules.md",
            inline_rules="Rule A",
            get_translation_provider_fn=lambda name: provider,
            load_task_resource_context_fn=lambda **kwargs: resources,
        )

        self.assertEqual(context.provider.name, "dummy")
        self.assertIs(context.resources, resources)
        self.assertIsNotNone(context.client)
        self.assertTrue(provider.last_flex_mode)

    def test_print_startup_configuration_prints_label_value_pairs(self):
        with patch("builtins.print") as mocked_print:
            print_startup_configuration(
                ("Provider", "gemini"),
                ("Model", "gemini-2.5-flash"),
            )

        mocked_print.assert_any_call("Startup configuration:")
        mocked_print.assert_any_call("  Provider: gemini")
        mocked_print.assert_any_call("  Model: gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
