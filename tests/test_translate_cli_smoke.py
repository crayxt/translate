import os
import unittest
from unittest.mock import patch

import translate_cli


class TranslateCliSmokeTests(unittest.TestCase):
    def test_translate_subcommand_dispatches_to_process(self):
        with patch("translate_cli.run_translate") as mocked_main:
            translate_cli.main(["translate", "input.po", "--target-lang", "fr"])
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "translate")
        self.assertEqual(mocked_main.call_args.args[0].files, ["input.po"])
        self.assertEqual(mocked_main.call_args.args[0].target_lang, "fr")

    def test_translate_subcommand_reports_invalid_provider_as_error(self):
        with self.assertRaises(SystemExit) as raised:
            translate_cli.main(["translate", "input.po", "--target-lang", "fr", "--provider", "nope"])

        self.assertEqual(
            str(raised.exception),
            "ERROR: Unsupported provider: 'nope'. Supported providers: anthropic, gemini, openai",
        )

    def test_translate_subcommand_reports_missing_input_file_as_error(self):
        with self.assertRaises(SystemExit) as raised:
            translate_cli.main(["translate", "_missing_translate_input.po", "--target-lang", "fr"])

        self.assertEqual(
            str(raised.exception),
            "ERROR: Input file does not exist: _missing_translate_input.po",
        )

    def test_translate_subcommand_accepts_multiple_files(self):
        with patch("translate_cli.run_translate") as mocked_main:
            translate_cli.main(["translate", "one.po", "two.po", "--target-lang", "fr"])
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "translate")
        self.assertEqual(mocked_main.call_args.args[0].files, ["one.po", "two.po"])

    def test_translate_subcommand_applies_gemini_environment_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("translate_cli.run_translate") as mocked_main:
                translate_cli.main(
                    [
                        "translate",
                        "input.po",
                        "--target-lang",
                        "fr",
                        "--provider",
                        "gemini",
                        "--gemini-backend",
                        "vertex",
                        "--google-cloud-location",
                        "global",
                    ]
                )
                mocked_main.assert_called_once()
                self.assertEqual(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"), "true")
                self.assertEqual(os.environ.get("GOOGLE_CLOUD_LOCATION"), "global")

    def test_extract_subcommand_dispatches_to_extract_terms(self):
        with patch("translate_cli.run_extract_terms") as mocked_main:
            translate_cli.main(["extract-terms", "input.po", "--target-lang", "fr"])
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "extract-terms")
        self.assertEqual(mocked_main.call_args.args[0].file, "input.po")

    def test_extract_local_subcommand_dispatches_to_extract_terms_local(self):
        with patch("translate_cli.run_extract_terms_local") as mocked_main:
            translate_cli.main(["extract-terms-local", "input.po"])
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "extract-terms-local")
        self.assertEqual(mocked_main.call_args.args[0].file, "input.po")

    def test_check_subcommand_dispatches_to_check_translations(self):
        with patch("translate_cli.run_check") as mocked_main:
            translate_cli.main(["check", "input.po", "--target-lang", "fr", "--probe", "5"])
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "check")
        self.assertEqual(mocked_main.call_args.args[0].file, "input.po")
        self.assertEqual(mocked_main.call_args.args[0].num_messages, 5)

    def test_revise_subcommand_dispatches_to_revise_translations(self):
        with patch("translate_cli.run_revise") as mocked_main:
            translate_cli.main(
                ["revise", "input.po", "--target-lang", "fr", "--instruction", "Use shorter term"]
            )
        mocked_main.assert_called_once()
        self.assertEqual(mocked_main.call_args.args[0].command, "revise")
        self.assertEqual(mocked_main.call_args.args[0].file, "input.po")
        self.assertEqual(mocked_main.call_args.args[0].instruction, "Use shorter term")


if __name__ == "__main__":
    unittest.main()
