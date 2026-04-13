import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import polib
from google.genai import types as genai_types

from tasks import check_translations
from tasks import translate as process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _DummyProvider:
    name = "gemini"
    default_model = "gemini-test"
    api_key_env = "GOOGLE_API_KEY"
    supports_structured_json = True
    supports_thinking = True

    def __init__(self, response=None):
        self.response = response or _DummyResponse(parsed={"results": []})

    def create_client_from_env(self, *, flex_mode: bool = False):
        return object()

    def build_generation_config(self, *, thinking_level, json_schema, system_instruction, flex_mode=False):
        return object()

    async def generate_with_retry(self, *, client, model, contents, batch_label, max_attempts, config):
        return self.response


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
        self.assertIn("software localization QA reviewer", config.system_instruction)

    def test_build_check_system_instruction_includes_script_guidance(self):
        system_instruction = check_translations.build_check_system_instruction("kk")
        self.assertIn("software localization QA reviewer", system_instruction)
        self.assertIn("MANDATORY LOCALIZATION INVARIANTS", system_instruction)
        self.assertIn("Determine the intended sense of the source text", system_instruction)
        self.assertIn("Do not rely on source-token overlap alone", system_instruction)
        self.assertIn("Suggested fixes must use the actual target-language alphabet/script", system_instruction)
        self.assertIn("real Kazakh Cyrillic alphabet", system_instruction)
        self.assertIn("Latin transliteration", system_instruction)

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

        self.assertIn("approved vocabulary/glossary", prompt.lower())
        self.assertIn("Use imperative tone.", prompt)
        self.assertIn('"translation": "Ashu <b>%s</b>"', prompt)
        self.assertIn("Do not flag a terminology issue solely because", prompt)
        self.assertIn("suggested_translation", prompt)

    def test_build_check_message_payload_uses_structured_plural_source_fields(self):
        entry = polib.POEntry(
            msgid="File",
            msgid_plural="Files",
            msgstr_plural={0: "Файл", 1: "Файлдар"},
        )

        payload = check_translations.build_check_message_payload(entry)

        self.assertEqual(payload["source_singular"], "File")
        self.assertEqual(payload["source_plural"], "Files")
        self.assertEqual(payload["plural_forms"], 2)
        self.assertEqual(payload["plural_slots"], ["0", "1"])
        self.assertEqual(payload["translation"], "Файл")
        self.assertEqual(payload["translation_plural_forms"], ["Файл", "Файлдар"])
        self.assertNotIn("source", payload)

    def test_parse_check_response_uses_parsed_payload(self):
        payload = {
            "results": [
                {
                    "id": "0",
                    "issues": [
                        {
                            "code": "check.meaning",
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
        self.assertEqual(parsed["0"][0].code, "check.meaning")

    def test_parse_check_response_normalizes_legacy_category_to_code(self):
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

        self.assertEqual(parsed["0"][0].code, "check.meaning")

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

    def test_load_entries_for_check_supports_xliff(self):
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.XLIFF,
                msgid="Open",
                msgstr="Ашу",
                status=process.EntryStatus.TRANSLATED,
            )
        ]

        with patch("tasks.check_translations.load_xliff", return_value=(entries, None, None)) as mocked_load:
            loaded = check_translations.load_entries_for_check("input.xliff", process.FileKind.XLIFF)

        mocked_load.assert_called_once_with("input.xliff")
        self.assertEqual(loaded, entries)

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
            provider = _DummyProvider(
                response=_DummyResponse(
                    parsed={
                        "results": [
                            {
                                "id": "0",
                                "issues": [
                                    {
                                        "code": "check.meaning",
                                        "severity": "warning",
                                        "message": "Translation may omit part of the source meaning.",
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            runtime_context = SimpleNamespace(
                provider=provider,
                client=object(),
                resources=SimpleNamespace(
                    vocabulary_text="addon - qosymsha",
                    project_rules="Use imperative tone.",
                    vocabulary_source=os.path.join("data", "locales", "kk", "vocab.txt"),
                    rules_source=os.path.join("data", "locales", "kk", "rules.md"),
                ),
            )
            with (
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
                patch("tasks.check_translations.build_task_runtime_context", return_value=runtime_context),
                patch("tasks.check_translations.detect_file_kind", return_value=process.FileKind.PO),
                patch("tasks.check_translations.load_po", return_value=(entries, None, None)),
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
            codes = [issue["code"] for issue in payload["results"][0]["issues"]]
            self.assertIn("check.meaning", codes)
            self.assertTrue(all(issue["origin"] == "model" for issue in payload["results"][0]["issues"]))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_main_writes_json_report_for_ts(self):
        out_path = os.path.join(os.getcwd(), "_tmp_translation_check_ts.json")
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.TS,
                msgid="Open",
                msgstr="Ashu",
                msgctxt="MainWindow",
                status=process.EntryStatus.TRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.TS,
                msgid="Save",
                msgstr="Saqtau",
                status=process.EntryStatus.TRANSLATED,
            ),
        ]

        try:
            provider = _DummyProvider()
            runtime_context = SimpleNamespace(
                provider=provider,
                client=object(),
                resources=SimpleNamespace(
                    vocabulary_text="",
                    project_rules="",
                    vocabulary_source=os.path.join("data", "locales", "kk", "vocab.txt"),
                    rules_source=os.path.join("data", "locales", "kk", "rules.md"),
                ),
            )
            with (
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
                patch("tasks.check_translations.build_task_runtime_context", return_value=runtime_context),
                patch("tasks.check_translations.detect_file_kind", return_value=process.FileKind.TS),
                patch("tasks.check_translations.load_ts", return_value=(entries, None, None)),
                patch(
                    "tasks.check_translations.sys.argv",
                    ["check_translations.py", "input.ts", "--out", out_path],
                ),
                patch("builtins.print"),
            ):
                check_translations.main()

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["source_file"], "input.ts")
            self.assertEqual(payload["total_entries_checked"], 2)
            self.assertEqual(payload["entries_with_issues"], 0)
            self.assertEqual(payload["issue_count"], 0)
            self.assertEqual(len(payload["results"]), 0)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_main_writes_json_report_for_xliff(self):
        out_path = os.path.join(os.getcwd(), "_tmp_translation_check_xliff.json")
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.XLIFF,
                msgid="Open",
                msgstr="Ашу",
                msgctxt="openAction",
                status=process.EntryStatus.TRANSLATED,
            )
        ]

        try:
            provider = _DummyProvider()
            runtime_context = SimpleNamespace(
                provider=provider,
                client=object(),
                resources=SimpleNamespace(
                    vocabulary_text="",
                    project_rules="",
                    vocabulary_source=os.path.join("data", "locales", "kk", "vocab.txt"),
                    rules_source=os.path.join("data", "locales", "kk", "rules.md"),
                ),
            )
            with (
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
                patch("tasks.check_translations.build_task_runtime_context", return_value=runtime_context),
                patch("tasks.check_translations.detect_file_kind", return_value=process.FileKind.XLIFF),
                patch("tasks.check_translations.load_xliff", return_value=(entries, None, None)),
                patch(
                    "tasks.check_translations.sys.argv",
                    ["check_translations.py", "input.xliff", "--out", out_path],
                ),
                patch("builtins.print"),
            ):
                check_translations.main()

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["source_file"], "input.xliff")
            self.assertEqual(payload["total_entries_checked"], 1)
            self.assertEqual(payload["entries_with_issues"], 0)
            self.assertEqual(payload["issue_count"], 0)
            self.assertEqual(len(payload["results"]), 0)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
