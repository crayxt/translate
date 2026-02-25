import unittest
from unittest.mock import patch
import os

import polib

import process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class ProcessSmokeTests(unittest.TestCase):
    def test_system_instruction_is_language_neutral(self):
        self.assertNotIn("Kazakh", process.SYSTEM_INSTRUCTION)
        self.assertIn("target language", process.SYSTEM_INSTRUCTION)

    def test_merge_project_rules_combines_file_and_inline(self):
        merged = process.merge_project_rules("Rule A", "Rule B")
        self.assertEqual(merged, "Rule A\n\nRule B")

    def test_detect_rules_source_file_only(self):
        src = process.detect_rules_source("rules-kk.md", "Rule A", None)
        self.assertEqual(src, "file:rules-kk.md")

    def test_detect_rules_source_inline_only(self):
        src = process.detect_rules_source("rules-kk.md", None, "Rule B")
        self.assertEqual(src, "inline:--rules-str")

    def test_detect_rules_source_file_and_inline(self):
        src = process.detect_rules_source("rules-kk.md", "Rule A", "Rule B")
        self.assertEqual(src, "file:rules-kk.md, inline:--rules-str")

    def test_build_prompt_includes_rules_block(self):
        prompt = process.build_prompt(
            messages={"0": {"source": "Open file", "context": "menu action"}},
            source_lang="en",
            target_lang="kk",
            vocabulary="open - ashy",
            translation_rules="Use imperative tone.",
        )
        self.assertIn("Project translation rules/instructions", prompt)
        self.assertIn("Use imperative tone.", prompt)
        self.assertIn('"context": "menu action"', prompt)

    def test_build_entry_source_text_handles_plural(self):
        entry = polib.POEntry(msgid="File", msgid_plural="Files")
        text = process.build_entry_source_text(entry)
        self.assertEqual(text, "Singular: File\nPlural: Files")

    def test_build_prompt_message_payload_includes_context_and_notes(self):
        entry = polib.POEntry(msgid="Save", msgctxt="Menu|File")
        entry.tcomment = "Action label"
        entry.occurrences = [("ui/main.py", "42")]

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["source"], "Save")
        self.assertEqual(payload["context"], "Menu|File")
        self.assertIn("Action label", payload["note"])
        self.assertIn("ui/main.py:42", payload["note"])

    def test_parse_response_uses_parsed_payload(self):
        payload = {
            "translations": [
                {"id": "0", "text": "Alpha"},
                {"id": "1", "text": "Beta", "plural_texts": ["Beta one", "Beta many"]},
            ]
        }

        results = process.parse_response(_DummyResponse(parsed=payload))

        self.assertEqual(results["0"].text, "Alpha")
        self.assertEqual(results["1"].plural_texts, ["Beta one", "Beta many"])

    def test_parse_response_falls_back_to_json_text(self):
        text = '{"translations":[{"id":"7","text":"Gamma"}]}'
        results = process.parse_response(_DummyResponse(text=text))
        self.assertEqual(results["7"].text, "Gamma")

    def test_apply_translation_to_plural_entry_prefers_plural_texts(self):
        entry = polib.POEntry(msgid="File", msgid_plural="Files")
        entry.msgstr_plural = {0: "", 1: ""}

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(
                text="Fallback",
                plural_texts=["Single Form", "Plural Form"],
            ),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "Single Form")
        self.assertEqual(entry.msgstr_plural[1], "Plural Form")

    def test_apply_translation_to_plural_entry_falls_back_to_text(self):
        entry = polib.POEntry(msgid="Day", msgid_plural="Days")
        entry.msgstr_plural = {0: "", 1: "", 2: ""}

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(text="Kun"),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "Kun")
        self.assertEqual(entry.msgstr_plural[1], "Kun")
        self.assertEqual(entry.msgstr_plural[2], "Kun")

    def test_build_output_path_uses_splitext(self):
        path = r"C:\po.files\sample.po"
        output = process.build_output_path(path, process.FileKind.PO)
        self.assertEqual(output, r"C:\po.files\sample.ai-translated.po")

    def test_build_language_code_candidates_include_locale_and_base(self):
        candidates = process.build_language_code_candidates("fr_CA")
        self.assertIn("fr_CA", candidates)
        self.assertIn("fr", candidates)

    def test_detect_default_text_resource_prefers_exact_match(self):
        with patch("process.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "fr_CA", "rules.md"),
                os.path.join("data", "fr", "rules.md"),
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, os.path.join("data", "fr_CA", "rules.md"))

    def test_detect_default_text_resource_falls_back_to_base_language(self):
        with patch("process.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "fr", "rules.md")
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, os.path.join("data", "fr", "rules.md"))

    def test_detect_default_text_resource_uses_legacy_fallback(self):
        with patch("process.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path == "rules-fr.md"
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, "rules-fr.md")

    def test_resolve_resource_path_prefers_explicit_path(self):
        with patch("process.detect_default_text_resource") as mocked_detect:
            resolved = process.resolve_resource_path("custom-rules.md", "rules", "md", "fr")

        mocked_detect.assert_not_called()
        self.assertEqual(resolved, "custom-rules.md")


if __name__ == "__main__":
    unittest.main()
