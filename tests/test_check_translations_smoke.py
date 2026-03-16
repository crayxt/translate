import json
import os
import unittest
from unittest.mock import patch

import polib
from google.genai import types as genai_types

from tasks import check_translations
from tasks import translate as process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class CheckTranslationsSmokeTests(unittest.TestCase):
    def test_build_check_output_path(self):
        path = r"C:\work\file.po"
        out = check_translations.build_check_output_path(path)
        self.assertEqual(out, r"C:\work\file.translation-check.json")

    def test_build_check_generation_config_includes_thinking_level(self):
        config = check_translations.build_check_generation_config("low")
        self.assertEqual(
            config.thinking_config.thinking_level,
            genai_types.ThinkingLevel.LOW,
        )

    def test_build_check_prompt_includes_translation_rules_and_vocab(self):
        prompt = check_translations.build_check_prompt(
            messages={
                "0": {
                    "source": "Open <b>%s</b>",
                    "translation": "Ashu <b>%s</b>",
                }
            },
            source_lang="en",
            target_lang="kk",
            vocabulary="open - ashu",
            translation_rules="Use imperative tone.",
        )

        self.assertIn("Approved vocabulary/glossary", prompt)
        self.assertIn("Use imperative tone.", prompt)
        self.assertIn('"translation": "Ashu <b>%s</b>"', prompt)
        self.assertIn("Inflection and derivation are acceptable", prompt)
        self.assertIn("Do not flag a terminology issue solely because", prompt)
        self.assertIn("Suggested fixes must use the actual target-language alphabet/script", prompt)
        self.assertIn("real Kazakh Cyrillic alphabet", prompt)
        self.assertIn("ө, ү, ұ, қ, ң, ғ, ә, і, һ", prompt)

    def test_parse_check_response_uses_parsed_payload(self):
        payload = {
            "results": [
                {
                    "id": "0",
                    "issues": [
                        {
                            "category": "meaning",
                            "severity": "error",
                            "message": "Meaning changed.",
                        }
                    ],
                }
            ]
        }

        parsed = check_translations.parse_check_response(_DummyResponse(parsed=payload))

        self.assertIn("0", parsed)
        self.assertEqual(parsed["0"][0].origin, "model")
        self.assertEqual(parsed["0"][0].category, "meaning")

    def test_select_review_entries_skips_untranslated_and_obsolete(self):
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Open",
                msgstr="Ashu",
                status=process.EntryStatus.TRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Save",
                msgstr="",
                status=process.EntryStatus.UNTRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Delete",
                msgstr="Oshiru",
                status=process.EntryStatus.TRANSLATED,
                obsolete=True,
            ),
        ]

        selected = check_translations.select_review_entries(entries)

        self.assertEqual([entry.msgid for entry in selected], ["Open"])

    def test_limit_review_entries_applies_cap(self):
        entries = [
            process.UnifiedEntry(file_kind=process.FileKind.PO, msgid="One", msgstr="Bir"),
            process.UnifiedEntry(file_kind=process.FileKind.PO, msgid="Two", msgstr="Eki"),
            process.UnifiedEntry(file_kind=process.FileKind.PO, msgid="Three", msgstr="Ush"),
        ]

        limited = check_translations.limit_review_entries(entries, 2)

        self.assertEqual([entry.msgid for entry in limited], ["One", "Two"])

    def test_limit_review_entries_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            check_translations.limit_review_entries([], 0)

    def test_build_target_script_guidance_generic_non_kazakh(self):
        guidance = check_translations.build_target_script_guidance("fr")
        self.assertIn("real writing system", guidance)

    def test_build_target_script_guidance_mentions_kazakh_cyrillic(self):
        guidance = check_translations.build_target_script_guidance("kk")
        self.assertIn("Kazakh Cyrillic alphabet", guidance)
        self.assertIn("Latin transliteration", guidance)

    def test_main_writes_json_report(self):
        out_path = os.path.join(os.getcwd(), "_tmp_translation_check.json")
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Open addon",
                msgstr="Ashu",
                status=process.EntryStatus.TRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Save addon",
                msgstr="Saqtau qosymsha",
                status=process.EntryStatus.TRANSLATED,
            ),
        ]

        try:
            with (
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
                patch("tasks.check_translations.genai.Client"),
                patch("tasks.check_translations.detect_file_kind", return_value=process.FileKind.PO),
                patch("tasks.check_translations.resolve_resource_path", side_effect=[os.path.join("data", "kk", "vocab.txt"), os.path.join("data", "kk", "rules.md")]),
                patch("tasks.check_translations.read_optional_vocabulary_file", return_value="addon - qosymsha"),
                patch("tasks.check_translations.read_optional_text_file", return_value="Use imperative tone."),
                patch("tasks.check_translations.load_po", return_value=(entries, None, None)),
                patch(
                    "tasks.check_translations.generate_with_retry",
                    return_value=_DummyResponse(
                        parsed={
                            "results": [
                                {
                                    "id": "0",
                                    "issues": [
                                        {
                                            "category": "meaning",
                                            "severity": "warning",
                                            "message": "Translation may omit part of the source meaning.",
                                        }
                                    ],
                                }
                            ]
                        }
                    ),
                ),
                patch(
                    "tasks.check_translations.sys.argv",
                    ["check_translations.py", "input.po", "--out", out_path, "--probe", "1"],
                ),
                patch("builtins.print"),
            ):
                check_translations.main()

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["total_entries_checked"], 1)
            self.assertEqual(payload["probe_limit"], 1)
            self.assertEqual(payload["entries_with_issues"], 1)
            self.assertEqual(payload["issue_count"], 1)
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["verdict"], "issues")
            categories = [issue["category"] for issue in payload["results"][0]["issues"]]
            self.assertIn("meaning", categories)
            self.assertTrue(all(issue["origin"] == "model" for issue in payload["results"][0]["issues"]))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
