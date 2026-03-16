import unittest
from unittest.mock import patch

import translate_cli


class TranslateCliSmokeTests(unittest.TestCase):
    def test_translate_subcommand_dispatches_to_process(self):
        with patch("translate_cli.run_translate") as mocked_main:
            translate_cli.main(["translate", "input.po", "--target-lang", "fr"])
        mocked_main.assert_called_once_with(["input.po", "--target-lang", "fr"])

    def test_extract_subcommand_dispatches_to_extract_terms(self):
        with patch("translate_cli.run_extract_terms") as mocked_main:
            translate_cli.main(["extract-terms", "input.po"])
        mocked_main.assert_called_once_with(["input.po"])

    def test_check_subcommand_dispatches_to_check_translations(self):
        with patch("translate_cli.run_check") as mocked_main:
            translate_cli.main(["check", "input.po", "--probe", "5"])
        mocked_main.assert_called_once_with(["input.po", "--probe", "5"])

    def test_revise_subcommand_dispatches_to_revise_translations(self):
        with patch("translate_cli.run_revise") as mocked_main:
            translate_cli.main(["revise", "input.po", "--instruction", "Use shorter term"])
        mocked_main.assert_called_once_with(["input.po", "--instruction", "Use shorter term"])


if __name__ == "__main__":
    unittest.main()
