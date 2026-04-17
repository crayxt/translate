import argparse
import unittest
from unittest.mock import patch

from core.task_cli import (
    apply_provider_environment_from_args,
    add_glossary_argument,
    add_language_arguments,
    add_max_attempts_argument,
    add_probe_argument,
    add_provider_arguments,
    add_rules_arguments,
    add_runtime_limit_arguments,
    build_task_parser,
    resolve_provider_model,
    run_task_main,
)


class TaskCliSmokeTests(unittest.TestCase):
    def test_provider_argument_defaults_to_gemini_vertex_backend(self):
        parser = argparse.ArgumentParser()
        add_provider_arguments(
            parser,
            default_provider_name="gemini",
            default_model="gemini-test",
            include_thinking=False,
        )

        args = parser.parse_args([])

        self.assertEqual(args.gemini_backend, "vertex")

    def test_common_argument_helpers_populate_expected_fields(self):
        parser = argparse.ArgumentParser()
        add_language_arguments(parser)
        add_provider_arguments(
            parser,
            default_provider_name="gemini",
            default_model="gemini-test",
            include_thinking=False,
        )
        add_runtime_limit_arguments(parser)
        add_glossary_argument(parser)
        add_rules_arguments(
            parser,
            rules_help="Rules file",
            rules_str_help="Inline rules",
        )
        add_probe_argument(parser, help_text="Probe count")
        add_max_attempts_argument(parser)

        args = parser.parse_args(
            [
                "--source-lang",
                "ru",
                "--target-lang",
                "kk",
                "--provider",
                "gemini",
                "--gemini-backend",
                "vertex",
                "--google-cloud-location",
                "global",
                "--model",
                "gemini-2.5-flash",
                "--flex",
                "--batch-size",
                "10",
                "--parallel-requests",
                "2",
                "--glossary",
                "glossary.po",
                "--rules",
                "rules.md",
                "--rules-str",
                "Rule A",
                "--probe",
                "5",
                "--max-attempts",
                "7",
            ]
        )

        self.assertEqual(args.source_lang, "ru")
        self.assertEqual(args.target_lang, "kk")
        self.assertEqual(args.provider, "gemini")
        self.assertEqual(args.gemini_backend, "vertex")
        self.assertEqual(args.google_cloud_location, "global")
        self.assertEqual(args.model, "gemini-2.5-flash")
        self.assertTrue(args.flex_mode)
        self.assertEqual(args.batch_size, 10)
        self.assertEqual(args.parallel_requests, 2)
        self.assertEqual(args.glossary, "glossary.po")
        self.assertEqual(args.rules, "rules.md")
        self.assertEqual(args.rules_str, "Rule A")
        self.assertEqual(args.num_messages, 5)
        self.assertEqual(args.max_attempts, 7)

    def test_build_task_parser_uses_configure_function(self):
        parser = build_task_parser(lambda parser: parser)
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_run_task_main_parses_args_and_calls_runner(self):
        def configure(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser.add_argument("file")
            return parser

        with patch("builtins.print"):
            captured = []

            def run_from_args(args):
                captured.append(args.file)

            run_task_main(
                configure_parser_fn=configure,
                run_from_args_fn=run_from_args,
                argv=["sample.po"],
            )

        self.assertEqual(captured, ["sample.po"])

    def test_resolve_provider_model_uses_selected_provider_default(self):
        model = resolve_provider_model("openai", None)
        self.assertEqual(model, "gpt-5-mini")

    def test_apply_provider_environment_from_args_sets_gemini_vertex_env(self):
        args = argparse.Namespace(
            provider="gemini",
            gemini_backend="vertex",
            google_cloud_location="global",
        )
        env = {}

        apply_provider_environment_from_args(args, environ=env)

        self.assertEqual(env["GOOGLE_GENAI_USE_VERTEXAI"], "true")
        self.assertEqual(env["GOOGLE_CLOUD_LOCATION"], "global")


if __name__ == "__main__":
    unittest.main()
