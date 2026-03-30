import unittest
from unittest.mock import patch

from core.task_resources import TaskResourceContext
from core.task_runtime import build_task_runtime_context, print_startup_configuration


class _DummyProvider:
    def __init__(self):
        self.name = "dummy"

    def create_client_from_env(self):
        return object()


class TaskRuntimeSmokeTests(unittest.TestCase):
    def test_build_task_runtime_context_builds_provider_client_and_resources(self):
        resources = TaskResourceContext(vocabulary_source="file:vocab.txt", rules_source="file:rules.md")

        context = build_task_runtime_context(
            provider_name="dummy",
            target_lang="kk",
            explicit_vocab_path="vocab.txt",
            explicit_rules_path="rules.md",
            inline_rules="Rule A",
            get_translation_provider_fn=lambda name: _DummyProvider(),
            load_task_resource_context_fn=lambda **kwargs: resources,
        )

        self.assertEqual(context.provider.name, "dummy")
        self.assertIs(context.resources, resources)
        self.assertIsNotNone(context.client)

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
