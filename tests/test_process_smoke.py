import unittest
from unittest.mock import AsyncMock, Mock, patch
import os
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import polib
from google.genai import types as genai_types

from tasks import translate as process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _BatchProvider:
    def __init__(self, responses):
        self.responses = responses

    async def generate_with_retry(self, *, client, model, contents, batch_label, max_attempts, config):
        return self.responses[batch_label]


class ProcessSmokeTests(unittest.TestCase):
    def test_system_instruction_is_language_neutral(self):
        self.assertNotIn("Kazakh", process.SYSTEM_INSTRUCTION)
        self.assertIn("MANDATORY LOCALIZATION INVARIANTS", process.SYSTEM_INSTRUCTION)
        self.assertIn("line-wrapping markers", process.SYSTEM_INSTRUCTION)
        self.assertIn("Determine the intended sense of the source text", process.SYSTEM_INSTRUCTION)
        self.assertIn("Do not rely on source-token overlap alone", process.SYSTEM_INSTRUCTION)

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
        self.assertIn("project translation rules/instructions", prompt.lower())
        self.assertIn("Use imperative tone.", prompt)
        self.assertIn('"context": "menu action"', prompt)
        self.assertIn("Return only the corrected final JSON.", prompt)
        self.assertIn("consistently", prompt.lower())
        self.assertIn("relevant_vocabulary", prompt)

    def test_build_scoped_vocabulary_entries_parses_rich_glossary_text(self):
        entries = process.build_scoped_vocabulary_entries(
            "start|бастау|verb|Start playback\nstart|басталу|noun|Playback start"
        )

        self.assertEqual(
            [(item.source_term, item.target_term, item.part_of_speech) for item in entries],
            [("start", "бастау", "verb"), ("start", "басталу", "noun")],
        )

    def test_build_translation_message_payload_includes_relevant_vocabulary(self):
        entry = polib.POEntry(msgid="Start playback")
        payload = process.build_translation_message_payload(
            entry,
            process.build_scoped_vocabulary_entries(
                "start|бастау|verb|Start playback\nplayback|ойнату|noun|Media playback\nstop|тоқтату|verb|"
            ),
        )

        self.assertEqual(payload["source"], "Start playback")
        self.assertEqual(
            payload["relevant_vocabulary"],
            [
                {
                    "source_term": "playback",
                    "target_term": "ойнату",
                    "part_of_speech": "noun",
                    "context_note": "Media playback",
                },
                {
                    "source_term": "start",
                    "target_term": "бастау",
                    "part_of_speech": "verb",
                    "context_note": "Start playback",
                },
            ],
        )

    def test_build_translation_message_payload_plural_uses_structured_source_and_relevant_vocabulary(self):
        entry = polib.POEntry(msgid="File", msgid_plural="Files")
        entry.msgstr_plural = {0: "", 1: ""}

        payload = process.build_translation_message_payload(
            entry,
            process.build_scoped_vocabulary_entries(
                "file|файл|noun|\nfiles|файлдар|noun|plural form\nopen|ашу|verb|"
            ),
        )

        self.assertNotIn("source", payload)
        self.assertEqual(payload["source_singular"], "File")
        self.assertEqual(payload["source_plural"], "Files")
        self.assertEqual(payload["plural_slots"], ["0", "1"])
        self.assertEqual(
            payload["relevant_vocabulary"],
            [
                {
                    "source_term": "files",
                    "target_term": "файлдар",
                    "part_of_speech": "noun",
                    "context_note": "plural form",
                },
                {
                    "source_term": "file",
                    "target_term": "файл",
                    "part_of_speech": "noun",
                },
            ],
        )

    def test_build_thinking_config_maps_cli_value(self):
        config = process.build_thinking_config("high")
        self.assertEqual(config.thinking_level, genai_types.ThinkingLevel.HIGH)

    def test_build_translation_generation_config_includes_thinking_level(self):
        config = process.build_translation_generation_config("minimal")
        self.assertEqual(
            config.thinking_config.thinking_level,
            genai_types.ThinkingLevel.MINIMAL,
        )
        self.assertIn("professional software localization translator", config.system_instruction)

    def test_build_translation_generation_config_includes_flex_model_selection(self):
        config = process.build_translation_generation_config("minimal", flex_mode=True)
        self.assertEqual(
            config.model_selection_config.feature_selection_preference,
            genai_types.FeatureSelectionPreference.PRIORITIZE_COST,
        )

    def test_build_prompt_no_longer_embeds_system_instruction(self):
        prompt = process.build_prompt(
            messages={"0": {"source": "Open file"}},
            source_lang="en",
            target_lang="kk",
            vocabulary=None,
            translation_rules=None,
        )
        self.assertNotIn("You are a professional software localization translator.", prompt)

    def test_read_optional_glossary_file_supports_po_glossary(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_glossary.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="addon", msgstr="qosymsha"))
            po.append(polib.POEntry(msgid="save as", msgstr="qalaysha saqtau"))
            po.save(glossary_path)

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertEqual(vocabulary, "addon|qosymsha||\nsave as|qalaysha saqtau||")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_supports_tbx_glossary(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_glossary.tbx")
        try:
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<martif type="TBX" xml:lang="en">
  <text>
    <body>
      <termEntry>
        <langSet xml:lang="en"><tig><term>start</term></tig></langSet>
        <langSet xml:lang="kk"><tig><term>бастау</term></tig></langSet>
        <descrip>Start playback</descrip>
      </termEntry>
      <termEntry>
        <langSet xml:lang="en"><tig><term>skip me</term></tig></langSet>
        <langSet xml:lang="kk"><tig><term/></tig></langSet>
      </termEntry>
    </body>
  </text>
</martif>
"""
                )

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertEqual(vocabulary, "start|бастау||Start playback")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_supports_directory_bundle(self):
        glossary_dir = os.path.join(os.getcwd(), "_tmp_glossary_bundle")
        common_po_path = os.path.join(glossary_dir, "10-common.po")
        po_path = os.path.join(glossary_dir, "30-overrides.po")
        ignored_path = os.path.join(glossary_dir, "notes.md")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            common_po = polib.POFile()
            common_po.append(polib.POEntry(msgid="save", msgstr="saqtau", msgctxt="verb"))
            common_po.append(polib.POEntry(msgid="open", msgstr="ashu", msgctxt="verb"))
            common_po.save(common_po_path)
            with open(ignored_path, "w", encoding="utf-8") as handle:
                handle.write("not a glossary file\n")
            po = polib.POFile()
            po.append(polib.POEntry(msgid="save", msgstr="qoru", msgctxt="verb"))
            po.append(polib.POEntry(msgid="blue", msgstr="kok", msgctxt="adjective"))
            po.save(po_path)

            vocabulary = process.read_optional_glossary_file(glossary_dir)

            self.assertEqual(
                vocabulary,
                "save|qoru|verb|\nopen|ashu|verb|\nblue|kok|adjective|",
            )
        finally:
            for path in (common_po_path, po_path, ignored_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)

    def test_parse_glossary_line_parses_source_target_pairs(self):
        parsed = process.parse_glossary_line("save as|qalaysha saqtau|verb phrase|")
        self.assertEqual(parsed, ("save as", "qalaysha saqtau"))

    def test_parse_glossary_line_rejects_legacy_txt_schema(self):
        parsed = process.parse_glossary_line("save as - qalaysha saqtau")
        self.assertIsNone(parsed)

    def test_parse_glossary_line_supports_rich_schema(self):
        parsed = process.parse_glossary_line("start|бастау|verb|Start playback")
        self.assertEqual(parsed, ("start", "бастау"))

    def test_load_glossary_pairs_from_po_preserves_last_duplicate_wins(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_pairs.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="save", msgstr="saqtau", msgctxt="verb"))
            po.append(polib.POEntry(msgid="save", msgstr="qoru", msgctxt="verb"))
            po.append(polib.POEntry(msgid="open", msgstr="ashu", msgctxt="verb"))
            po.save(glossary_path)

            pairs = process.load_glossary_pairs(glossary_path)

            self.assertEqual(pairs, [("save", "qoru"), ("open", "ashu")])
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_load_glossary_pairs_preserves_same_source_with_different_rich_context(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_pairs_rich.po")
        try:
            po = polib.POFile()
            first = polib.POEntry(msgid="start", msgstr="бастау", msgctxt="verb")
            first.tcomment = "Start playback"
            po.append(first)
            second = polib.POEntry(msgid="start", msgstr="басталу", msgctxt="noun")
            second.tcomment = "Playback start"
            po.append(second)
            po.save(glossary_path)

            pairs = process.load_glossary_pairs(glossary_path)

            self.assertEqual(pairs, [("start", "бастау"), ("start", "басталу")])
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_load_glossary_pairs_from_directory_bundle_last_duplicate_wins(self):
        glossary_dir = os.path.join(os.getcwd(), "_tmp_glossary_pairs_bundle")
        first_path = os.path.join(glossary_dir, "10-common.po")
        second_path = os.path.join(glossary_dir, "20-colors.po")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            first_po = polib.POFile()
            first_po.append(polib.POEntry(msgid="save", msgstr="saqtau", msgctxt="verb"))
            first_po.append(polib.POEntry(msgid="open", msgstr="ashu", msgctxt="verb"))
            first_po.save(first_path)
            second_po = polib.POFile()
            second_po.append(polib.POEntry(msgid="save", msgstr="qoru", msgctxt="verb"))
            second_po.append(polib.POEntry(msgid="blue", msgstr="kok", msgctxt="adjective"))
            second_po.save(second_path)

            pairs = process.load_glossary_pairs(glossary_dir)

            self.assertEqual(
                pairs,
                [("save", "qoru"), ("open", "ashu"), ("blue", "kok")],
            )
        finally:
            for path in (first_path, second_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)

    def test_load_glossary_pairs_from_directory_bundle_supports_tbx(self):
        glossary_dir = os.path.join(os.getcwd(), "_tmp_glossary_tbx_bundle")
        tbx_path = os.path.join(glossary_dir, "10-common.tbx")
        po_path = os.path.join(glossary_dir, "20-overrides.po")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            with open(tbx_path, "w", encoding="utf-8") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<martif type="TBX" xml:lang="en">
  <text>
    <body>
      <termEntry>
        <langSet xml:lang="en"><tig><term>save</term></tig></langSet>
        <langSet xml:lang="kk"><tig><term>сақтау</term></tig></langSet>
      </termEntry>
      <termEntry>
        <langSet xml:lang="en"><tig><term>open</term></tig></langSet>
        <langSet xml:lang="kk"><tig><term>ашу</term></tig></langSet>
      </termEntry>
    </body>
  </text>
</martif>
"""
                )
            po = polib.POFile()
            po.append(polib.POEntry(msgid="save", msgstr="қору"))
            po.save(po_path)

            pairs = process.load_glossary_pairs(glossary_dir)

            self.assertEqual(pairs, [("save", "қору"), ("open", "ашу")])
        finally:
            for path in (tbx_path, po_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)

    def test_read_optional_glossary_file_rejects_txt_glossary(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_rich.txt")
        try:
            with open(glossary_path, "w", encoding="utf-8") as f:
                f.write("start|бастау|verb|Start playback\n")
                f.write("start|басталу|noun|Playback start\n")

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertIsNone(vocabulary)
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_returns_none_for_legacy_txt_schema(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_legacy.txt")
        try:
            with open(glossary_path, "w", encoding="utf-8") as f:
                f.write("save - saqtau\n")

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertIsNone(vocabulary)
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_skips_untranslated_fuzzy_and_obsolete_po_entries(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_glossary_filtered.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="addon", msgstr="qosymsha"))
            po.append(polib.POEntry(msgid="blank target", msgstr=""))
            po.append(polib.POEntry(msgid="needs review", msgstr="tekseru", flags=["fuzzy"]))
            obsolete = polib.POEntry(msgid="old term", msgstr="old target", obsolete=True)
            po.append(obsolete)
            po.save(glossary_path)

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertEqual(vocabulary, "addon|qosymsha||")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_includes_po_context_and_note_fields(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_glossary_context.po")
        try:
            po = polib.POFile()
            entry = polib.POEntry(msgid="start", msgstr="бастау", msgctxt="verb")
            entry.tcomment = "Start playback"
            po.append(entry)
            po.save(glossary_path)

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertEqual(vocabulary, "start|бастау|verb|Start playback")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_read_optional_glossary_file_supports_generated_glossary_catalog_po(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_glossary_glossary_catalog.po")
        try:
            po = polib.POFile()
            entry = polib.POEntry(
                msgid="line",
                msgstr="жол",
                msgctxt="line of text, text layout, editor text",
            )
            entry.comment = "ID: line.noun.text\nPOS: noun\nSense: text"
            po.append(entry)
            po.save(glossary_path)

            vocabulary = process.read_optional_glossary_file(glossary_path)

            self.assertEqual(
                vocabulary,
                "line|жол|noun|line of text, text layout, editor text",
            )
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_load_po_uses_wrapwidth_78(self):
        with patch("core.formats.po.polib.pofile", return_value=polib.POFile()) as mocked_pofile:
            process.load_po("sample.po")

        mocked_pofile.assert_called_once_with("sample.po", wrapwidth=process.PO_WRAP_WIDTH)

    def test_build_entry_source_text_handles_plural(self):
        entry = polib.POEntry(msgid="File", msgid_plural="Files")
        text = process.build_entry_source_text(entry)
        self.assertEqual(text, "Singular: File\nPlural: Files")

    def test_resolve_translation_input_paths_expands_directory_recursively(self):
        root_dir = os.path.join(os.getcwd(), "_tmp_translate_tree")
        top_po = os.path.join(root_dir, "top.po")
        nested_dir = os.path.join(root_dir, "nested")
        nested_po = os.path.join(nested_dir, "child.po")
        ignored = os.path.join(nested_dir, "notes.md")
        try:
            os.makedirs(nested_dir, exist_ok=True)
            with open(top_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "Top"\nmsgstr ""\n')
            with open(nested_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "Child"\nmsgstr ""\n')
            with open(ignored, "w", encoding="utf-8") as handle:
                handle.write("ignore me\n")

            resolved = process.resolve_translation_input_paths([root_dir])

            self.assertEqual(resolved, [top_po, nested_po])
        finally:
            for path in (top_po, nested_po, ignored):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(nested_dir):
                os.rmdir(nested_dir)
            if os.path.isdir(root_dir):
                os.rmdir(root_dir)

    def test_resolve_translation_input_paths_skips_generated_outputs_in_directory_scan(self):
        root_dir = os.path.join(os.getcwd(), "_tmp_translate_generated_filter")
        source_po = os.path.join(root_dir, "source.po")
        generated_po = os.path.join(root_dir, "source.ai-translated.po")
        glossary_po = os.path.join(root_dir, "source.glossary.po")
        missing_po = os.path.join(root_dir, "source.missing-terms.po")
        prototype_po = os.path.join(root_dir, "source.prototype-missing-terms.po")
        try:
            os.makedirs(root_dir, exist_ok=True)
            for path in (source_po, generated_po, glossary_po, missing_po, prototype_po):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write('msgid "One"\nmsgstr ""\n')

            resolved = process.resolve_translation_input_paths([root_dir])

            self.assertEqual(resolved, [source_po])
        finally:
            for path in (source_po, generated_po, glossary_po, missing_po, prototype_po):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(root_dir):
                os.rmdir(root_dir)

    def test_resolve_translation_input_paths_skips_toolkit_resource_dirs_when_scanning_repo_root(self):
        root_dir = os.path.join(os.getcwd(), "_tmp_translate_repo_root")
        source_dir = os.path.join(root_dir, "locale")
        source_po = os.path.join(source_dir, "source.po")
        glossary_dir = os.path.join(root_dir, "data", "locales", "kk")
        glossary_po = os.path.join(glossary_dir, "glossary.po")
        logs_dir = os.path.join(root_dir, "logs")
        log_txt = os.path.join(logs_dir, "run.txt")
        requirements_txt = os.path.join(root_dir, "requirements.txt")
        try:
            os.makedirs(source_dir, exist_ok=True)
            os.makedirs(glossary_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            with open(source_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "One"\nmsgstr ""\n')
            with open(glossary_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "save"\nmsgstr "сақтау"\n')
            with open(log_txt, "w", encoding="utf-8") as handle:
                handle.write("tool log\n")
            with open(requirements_txt, "w", encoding="utf-8") as handle:
                handle.write("polib>=1.2.0\n")

            with patch(
                "tasks.translate.TOOLKIT_PROJECT_ROOT",
                process._normalize_scan_path(root_dir),
            ):
                resolved = process.resolve_translation_input_paths([root_dir])

            self.assertEqual(resolved, [source_po])
        finally:
            for path in (source_po, glossary_po, log_txt, requirements_txt):
                if os.path.exists(path):
                    os.remove(path)
            for path in (source_dir, glossary_dir, logs_dir):
                if os.path.isdir(path):
                    os.rmdir(path)
            data_locales_dir = os.path.join(root_dir, "data", "locales")
            data_dir = os.path.join(root_dir, "data")
            if os.path.isdir(data_locales_dir):
                os.rmdir(data_locales_dir)
            if os.path.isdir(data_dir):
                os.rmdir(data_dir)
            locale_parent = os.path.join(root_dir, "locale")
            if os.path.isdir(locale_parent):
                os.rmdir(locale_parent)
            if os.path.isdir(root_dir):
                os.rmdir(root_dir)

    def test_run_translation_reports_malformed_ts_as_error(self):
        file_path = os.path.join(os.getcwd(), "_tmp_malformed_translate.ts")
        try:
            with open(file_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("<TS><context>")

            with self.assertRaises(SystemExit) as raised:
                process.run_translation(
                    process.TranslationRunConfig(
                        files=[file_path],
                        source_file=None,
                        source_lang="en",
                        target_lang="kk",
                        provider="gemini",
                        model="gemini-test",
                        thinking_level=None,
                        batch_size=None,
                        parallel_requests=None,
                        glossary=None,
                        rules=None,
                        rules_str=None,
                        retranslate_all=False,
                        flex_mode=False,
                        warnings_report=False,
                    )
                )

            self.assertIn("ERROR: Failed to load translation input", str(raised.exception))
            self.assertIn("_tmp_malformed_translate.ts", str(raised.exception))
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_validate_translation_files_accepts_directory_tree_of_same_format(self):
        root_dir = os.path.join(os.getcwd(), "_tmp_translate_tree_validate")
        first_po = os.path.join(root_dir, "one.po")
        second_po = os.path.join(root_dir, "sub", "two.po")
        try:
            os.makedirs(os.path.dirname(second_po), exist_ok=True)
            with open(first_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "One"\nmsgstr ""\n')
            with open(second_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "Two"\nmsgstr ""\n')

            kind = process.validate_translation_files([root_dir])

            self.assertEqual(kind, process.FileKind.PO)
        finally:
            for path in (first_po, second_po):
                if os.path.exists(path):
                    os.remove(path)
            subdir = os.path.dirname(second_po)
            if os.path.isdir(subdir):
                os.rmdir(subdir)
            if os.path.isdir(root_dir):
                os.rmdir(root_dir)

    def test_build_prompt_message_payload_includes_context_and_notes(self):
        entry = polib.POEntry(msgid="Save", msgctxt="Menu|File")
        entry.tcomment = "Action label"
        entry.occurrences = [("ui/main.py", "42")]

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["source"], "Save")
        self.assertEqual(payload["context"], "Menu|File")
        self.assertIn("Action label", payload["note"])
        self.assertIn("ui/main.py:42", payload["note"])

    def test_build_prompt_message_payload_plural_adds_required_form_count(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n file(s)</source>"
            "<translation><numerusform></numerusform><numerusform></numerusform></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message, context_name="Files")

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["source_singular"], "%n file(s)")
        self.assertEqual(payload["source_plural"], "%n file(s)")
        self.assertEqual(payload["plural_forms"], 2)
        self.assertEqual(payload["plural_slots"], ["0", "1"])
        self.assertIn("plural forms required: 2", payload["note"])
        self.assertNotIn("source", payload)

    def test_build_prompt_message_payload_plural_adds_plural_basis_note(self):
        entry = polib.POEntry(msgid="One file deleted", msgid_plural="%d files deleted")
        entry.msgstr_plural = {0: "", 1: ""}

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["source_singular"], "One file deleted")
        self.assertEqual(payload["source_plural"], "%d files deleted")
        self.assertEqual(payload["plural_forms"], 2)
        self.assertEqual(payload["plural_slots"], ["0", "1"])
        self.assertIn("translate source_singular and source_plural separately", payload["note"])
        self.assertIn("repeat the consistent wording in all required plural slots", payload["note"])

    def test_build_prompt_message_payload_plural_uses_existing_plural_keys_as_slots(self):
        entry = polib.POEntry(msgid="Day", msgid_plural="Days")
        entry.msgstr_plural = {0: "", 1: "", 5: ""}

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["plural_slots"], ["0", "1", "5"])

    def test_build_prompt_message_payload_plural_form_count_matches_existing_slots(self):
        entry = polib.POEntry(msgid="Day", msgid_plural="Days")
        entry.msgstr_plural = {0: ""}

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["plural_forms"], 1)
        self.assertEqual(payload["plural_slots"], ["0"])

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

    def test_parse_response_preserves_message_warnings(self):
        payload = {
            "translations": [
                {
                    "id": "0",
                    "text": "Ағынды бастау",
                    "warnings": [
                        {
                            "code": "translate.ambiguous_term",
                            "message": "Stream may be noun or verb; chose noun sense from context.",
                            "severity": "warning",
                        }
                    ],
                }
            ]
        }

        results = process.parse_response(_DummyResponse(parsed=payload))

        self.assertEqual(
            results["0"].warnings,
            [
                process.TranslationWarning(
                    code="translate.ambiguous_term",
                    message="Stream may be noun or verb; chose noun sense from context.",
                    severity="warning",
                    origin="model",
                )
            ],
        )

    def test_parse_response_normalizes_legacy_string_warning_to_default_code(self):
        payload = {
            "translations": [
                {
                    "id": "0",
                    "text": "Ағынды бастау",
                    "warnings": [
                        "Ambiguous term: stream may be noun or verb; chose noun sense from context."
                    ],
                }
            ]
        }

        results = process.parse_response(_DummyResponse(parsed=payload))

        self.assertEqual(
            results["0"].warnings,
            [
                process.TranslationWarning(
                    code="translate.unclear_source_meaning",
                    message="Ambiguous term: stream may be noun or verb; chose noun sense from context.",
                    severity="warning",
                    origin="model",
                )
            ],
        )

    def test_parse_response_preserves_info_level_warning(self):
        payload = {
            "translations": [
                {
                    "id": "0",
                    "text": "XML құрылымын сақтау",
                    "warnings": [
                        {
                            "code": "translate.placeholder_attention",
                            "message": "XML structure preserved exactly.",
                            "severity": "info",
                        }
                    ],
                }
            ]
        }

        results = process.parse_response(_DummyResponse(parsed=payload))

        self.assertEqual(
            results["0"].warnings,
            [
                process.TranslationWarning(
                    code="translate.placeholder_attention",
                    message="XML structure preserved exactly.",
                    severity="info",
                    origin="model",
                )
            ],
        )

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

    def test_apply_translation_to_plural_entry_rejects_labeled_plural_text_in_text(self):
        entry = polib.POEntry(msgid="Day", msgid_plural="Days")
        entry.msgstr_plural = {0: "", 1: "", 2: ""}

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(
                text="Singular: Kun\nPlural: Kunder"
            ),
        )

        self.assertFalse(applied)
        self.assertEqual(entry.msgstr_plural[0], "")
        self.assertEqual(entry.msgstr_plural[1], "")
        self.assertEqual(entry.msgstr_plural[2], "")

    def test_apply_translation_to_plural_entry_ignores_blank_plural_forms(self):
        entry = polib.POEntry(msgid="Day", msgid_plural="Days")
        entry.msgstr_plural = {0: "", 1: ""}

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(text="Kun", plural_texts=[" ", ""]),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "Kun")
        self.assertEqual(entry.msgstr_plural[1], "Kun")

    def test_translation_has_content_ignores_whitespace(self):
        self.assertFalse(
            process.translation_has_content(
                process.TranslationResult(text=" ", plural_texts=["", "  "])
            )
        )

    def test_apply_translation_to_ts_plural_entry_writes_numerusform(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n file(s)</source>"
            "<translation type='unfinished'><numerusform></numerusform><numerusform></numerusform></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message, context_name="Files")

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(
                text="Fallback",
                plural_texts=["%n file", "%n files"],
            ),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "%n file")
        self.assertEqual(entry.msgstr_plural[1], "%n files")
        forms = message.find("translation").findall("numerusform")
        self.assertEqual(forms[0].text, "%n file")
        self.assertEqual(forms[1].text, "%n files")

    def test_apply_translation_to_ts_plural_entry_single_form_repeats_for_all_slots(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n term</source>"
            "<translation type='unfinished'><numerusform></numerusform><numerusform></numerusform></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message, context_name="Animals")

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(
                text="Fallback",
                plural_texts=["%n TERM"],
            ),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "%n TERM")
        self.assertEqual(entry.msgstr_plural[1], "%n TERM")
        forms = message.find("translation").findall("numerusform")
        self.assertEqual(forms[0].text, "%n TERM")
        self.assertEqual(forms[1].text, "%n TERM")

    def test_apply_translation_to_ts_plural_entry_creates_missing_numerusform_nodes(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n file(s)</source>"
            "<translation type='unfinished'></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message, context_name="Files")

        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(
                text="Fallback",
                plural_texts=["%n file", "%n files"],
            ),
        )

        self.assertTrue(applied)
        self.assertEqual(entry.msgstr_plural[0], "%n file")
        self.assertEqual(entry.msgstr_plural[1], "%n files")
        forms = message.find("translation").findall("numerusform")
        self.assertEqual([node.text for node in forms], ["%n file", "%n files"])

    def test_ts_plural_entry_translated_requires_all_forms(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n file(s)</source>"
            "<translation><numerusform>one</numerusform><numerusform></numerusform></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message)
        self.assertFalse(entry.translated())

        message.find("translation").findall("numerusform")[1].text = "many"
        self.assertTrue(entry.translated())

    def test_build_output_path_uses_splitext(self):
        path = r"C:\po.files\sample.po"
        output = process.build_output_path(path, process.FileKind.PO)
        self.assertEqual(output, r"C:\po.files\sample.ai-translated.po")

    def test_detect_file_kind_supports_strings(self):
        self.assertEqual(
            process.detect_file_kind(r"C:\tmp\sample.strings"),
            process.FileKind.STRINGS,
        )

    def test_detect_file_kind_supports_txt(self):
        self.assertEqual(
            process.detect_file_kind(r"C:\tmp\sample.txt"),
            process.FileKind.TXT,
        )

    def test_validate_translation_files_rejects_mixed_file_types(self):
        po_path = os.path.join(os.getcwd(), "_tmp_validate_mixed.po")
        ts_path = os.path.join(os.getcwd(), "_tmp_validate_mixed.ts")
        try:
            with open(po_path, "w", encoding="utf-8", newline="") as handle:
                handle.write('msgid ""\nmsgstr ""\n')
            with open(ts_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("<TS></TS>")

            with self.assertRaisesRegex(ValueError, "same format"):
                process.validate_translation_files([po_path, ts_path])
        finally:
            if os.path.exists(po_path):
                os.remove(po_path)
            if os.path.exists(ts_path):
                os.remove(ts_path)

    def test_select_work_items_respects_retranslate_all(self):
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Untranslated",
                status=process.EntryStatus.UNTRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Fuzzy",
                status=process.EntryStatus.FUZZY,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Translated",
                status=process.EntryStatus.TRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="",
                status=process.EntryStatus.TRANSLATED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Skipped",
                status=process.EntryStatus.SKIPPED,
            ),
            process.UnifiedEntry(
                file_kind=process.FileKind.PO,
                msgid="Obsolete",
                status=process.EntryStatus.UNTRANSLATED,
                obsolete=True,
            ),
        ]

        default_items = process.select_work_items(entries, retranslate_all=False)
        forced_items = process.select_work_items(entries, retranslate_all=True)

        self.assertEqual([e.msgid for e in default_items], ["Untranslated", "Fuzzy"])
        self.assertEqual([e.msgid for e in forced_items], ["Untranslated", "Fuzzy", "Translated"])

    def test_run_translation_batches_routes_results_to_correct_files(self):
        saved: list[str] = []
        entry_a = process.UnifiedEntry(
            file_kind=process.FileKind.PO,
            msgid="Open",
            status=process.EntryStatus.UNTRANSLATED,
        )
        entry_b = process.UnifiedEntry(
            file_kind=process.FileKind.PO,
            msgid="Save",
            status=process.EntryStatus.UNTRANSLATED,
        )
        job_a = process.TranslationFileJob(
            file_path="one.po",
            file_kind=process.FileKind.PO,
            entries=[entry_a],
            save_callback=lambda: saved.append("one"),
            output_path="one.ai-translated.po",
        )
        job_b = process.TranslationFileJob(
            file_path="two.po",
            file_kind=process.FileKind.PO,
            entries=[entry_b],
            save_callback=lambda: saved.append("two"),
            output_path="two.ai-translated.po",
        )
        provider = _BatchProvider(
            {
                "batch 1/1": _DummyResponse(
                    parsed={
                        "translations": [
                            {"id": "0", "text": "Ashu"},
                            {"id": "1", "text": "Saqtau"},
                        ]
                    }
                )
            }
        )

        translated_count, touched_output_paths = process.asyncio.run(
            process.run_translation_batches(
                provider=provider,
                client=object(),
                model="test-model",
                translation_config=object(),
                all_batches=[
                    [
                        process.TranslationQueueItem(job=job_a, entry=entry_a),
                        process.TranslationQueueItem(job=job_b, entry=entry_b),
                    ]
                ],
                total=2,
                parallel_requests=1,
                source_lang="en",
                target_lang="kk",
                glossary_text=None,
                project_rules=None,
                scoped_vocabulary_entries=[],
            )
        )

        self.assertEqual(translated_count, 2)
        self.assertEqual(
            touched_output_paths,
            {"one.ai-translated.po", "two.ai-translated.po"},
        )
        self.assertEqual(entry_a.msgstr, "Ashu")
        self.assertEqual(entry_b.msgstr, "Saqtau")
        self.assertIn("fuzzy", entry_a.flags)
        self.assertIn("fuzzy", entry_b.flags)
        self.assertCountEqual(saved, ["one", "two"])

    def test_write_translation_warning_report_writes_sidecar_json(self):
        entry = polib.POEntry(msgid="Start stream", msgctxt="Button label")
        output_path = os.path.join(os.getcwd(), "_tmp_warning_report.ai-translated.po")
        report_path = process.build_translation_warnings_output_path(output_path)
        try:
            job = process.TranslationFileJob(
                file_path="input.po",
                file_kind=process.FileKind.PO,
                entries=[entry],
                save_callback=None,
                output_path=output_path,
            )

            written_path = process.write_translation_warning_report(
                job=job,
                warning_items=[
                    process.build_translation_warning_item(
                        entry,
                        process.TranslationResult(
                            text="Ағынды бастау",
                            warnings=[
                                process.TranslationWarning(
                                    code="translate.ambiguous_term",
                                    message="Stream may be noun or verb.",
                                    severity="warning",
                                )
                            ],
                        ),
                        process.build_scoped_vocabulary_entries(
                            "start|бастау|verb|Start playback\nstream|ағын|noun|Media stream"
                        ),
                    )
                ],
                provider_name="gemini",
                model="gemini-test",
                source_lang="en",
                target_lang="kk",
            )

            self.assertEqual(written_path, report_path)
            with open(report_path, "r", encoding="utf-8") as handle:
                payload = process.json.load(handle)

            self.assertEqual(payload["output_file"], output_path)
            self.assertEqual(payload["warning_message_count"], 1)
            self.assertEqual(payload["messages"][0]["source"], "Start stream")
            self.assertEqual(payload["messages"][0]["context"], "Button label")
            self.assertEqual(payload["messages"][0]["translation"], "Ағынды бастау")
            self.assertEqual(
                payload["messages"][0]["warnings"],
                [
                    {
                        "code": "translate.ambiguous_term",
                        "message": "Stream may be noun or verb.",
                        "severity": "warning",
                    }
                ],
            )
            self.assertEqual(
                payload["messages"][0]["relevant_vocabulary"][0]["source_term"],
                "stream",
            )
        finally:
            if os.path.exists(report_path):
                os.remove(report_path)

    def test_build_translation_warning_item_preserves_plural_source_fields(self):
        entry = polib.POEntry(msgid="File", msgid_plural="Files")
        entry.msgstr_plural = {0: "", 1: ""}

        item = process.build_translation_warning_item(
            entry,
            process.TranslationResult(
                text="",
                plural_texts=["Файл", "Файлдар"],
            ),
            [],
        )

        self.assertEqual(item["source"], "Singular: File\nPlural: Files")
        self.assertEqual(item["source_singular"], "File")
        self.assertEqual(item["source_plural"], "Files")
        self.assertEqual(item["translation"], "Файл")
        self.assertEqual(item["plural_texts"], ["Файл", "Файлдар"])
        self.assertEqual(item["plural_forms"], 2)
        self.assertEqual(item["plural_slots"], ["0", "1"])

    def test_unified_entry_model_exposes_status_and_string_type(self):
        in_path = os.path.join(os.getcwd(), "_tmp_unified.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_unified.ai-translated.strings")
        try:
            content = (
                '/* "app|Name" = "Document Viewer"; */\n'
                '"app|Comment" = "Already translated";\n'
            )
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, _, _ = process.load_strings(in_path)
            self.assertEqual(len(entries), 2)
            self.assertIsInstance(entries[0], process.UnifiedEntry)
            self.assertEqual(entries[0].string_type, "strings")
            self.assertEqual(entries[0].status, process.EntryStatus.UNTRANSLATED)
            self.assertEqual(entries[1].status, process.EntryStatus.TRANSLATED)

            applied = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="Hujat korsetkishi"),
            )
            self.assertTrue(applied)
            self.assertEqual(entries[0].status, process.EntryStatus.TRANSLATED)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_unified_entry_status_is_fuzzy_for_po_fuzzy_flag(self):
        entry = polib.POEntry(msgid="Save", msgstr="Saqtau")
        entry.flags = ["fuzzy"]
        unified = process._build_unified_entry(
            entry=entry,
            file_kind=process.FileKind.PO,
            commit_callback=lambda _: None,
        )
        self.assertEqual(unified.status, process.EntryStatus.FUZZY)

    def test_unified_entry_status_is_fuzzy_for_ts_unfinished(self):
        message = ET.fromstring(
            "<message>"
            "<source>Open</source>"
            "<translation type='unfinished'>Ashu</translation>"
            "</message>"
        )
        ts_entry = process.TSEntryAdapter(message, context_name="Main")
        unified = process._build_unified_entry(
            entry=ts_entry,
            file_kind=process.FileKind.TS,
            commit_callback=lambda _: None,
        )
        self.assertEqual(unified.status, process.EntryStatus.FUZZY)
        self.assertFalse(unified.translated())

    def test_apply_translation_marks_fuzzy_entry_as_translated_status(self):
        entry = process.UnifiedEntry(
            file_kind=process.FileKind.PO,
            msgid="Open",
            msgstr="",
            status=process.EntryStatus.FUZZY,
        )
        applied = process.apply_translation_to_entry(
            entry,
            process.TranslationResult(text="Ashu"),
        )
        self.assertTrue(applied)
        self.assertEqual(entry.status, process.EntryStatus.TRANSLATED)

    def test_loaders_return_unified_entries_for_po_ts_resx(self):
        po_path = os.path.join(os.getcwd(), "_tmp_unified.po")
        po_out = os.path.join(os.getcwd(), "_tmp_unified.ai-translated.po")
        ts_path = os.path.join(os.getcwd(), "_tmp_unified.ts")
        ts_out = os.path.join(os.getcwd(), "_tmp_unified.ai-translated.ts")
        resx_path = os.path.join(os.getcwd(), "_tmp_unified.resx")
        resx_out = os.path.join(os.getcwd(), "_tmp_unified.ai-translated.resx")
        txt_path = os.path.join(os.getcwd(), "_tmp_unified.txt")
        txt_out = os.path.join(os.getcwd(), "_tmp_unified.ai-translated.txt")
        try:
            po_content = (
                'msgid ""\n'
                'msgstr ""\n'
                '"Language: en\\n"\n\n'
                'msgid "Open file"\n'
                'msgstr ""\n'
            )
            with open(po_path, "w", encoding="utf-8", newline="") as f:
                f.write(po_content)

            ts_content = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<TS version=\"2.1\" language=\"kk\">\n"
                "  <context>\n"
                "    <name>Main</name>\n"
                "    <message>\n"
                "      <source>Open</source>\n"
                "      <translation type=\"unfinished\"></translation>\n"
                "    </message>\n"
                "  </context>\n"
                "</TS>\n"
            )
            with open(ts_path, "w", encoding="utf-8", newline="") as f:
                f.write(ts_content)

            resx_content = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<root>\n"
                '  <data name="Label.Text"><value>Open</value></data>\n'
                '  <data name="Icon" type="System.Drawing.Bitmap"><value>icon.bmp</value></data>\n'
                "</root>\n"
            )
            with open(resx_path, "w", encoding="utf-8", newline="") as f:
                f.write(resx_content)

            txt_content = "First line\n\nSecond line\n"
            with open(txt_path, "w", encoding="utf-8", newline="") as f:
                f.write(txt_content)

            po_entries, _, _ = process.load_po(po_path)
            ts_entries, _, _ = process.load_ts(ts_path)
            resx_entries, _, _ = process.load_resx(resx_path)
            txt_entries, _, _ = process.load_txt(txt_path)

            self.assertTrue(po_entries)
            self.assertTrue(ts_entries)
            self.assertTrue(resx_entries)
            self.assertTrue(txt_entries)

            self.assertTrue(all(isinstance(e, process.UnifiedEntry) for e in po_entries))
            self.assertTrue(all(isinstance(e, process.UnifiedEntry) for e in ts_entries))
            self.assertTrue(all(isinstance(e, process.UnifiedEntry) for e in resx_entries))
            self.assertTrue(all(isinstance(e, process.UnifiedEntry) for e in txt_entries))

            self.assertEqual(po_entries[0].string_type, "po")
            self.assertEqual(ts_entries[0].string_type, "ts")
            self.assertEqual(resx_entries[0].string_type, "resx")
            self.assertEqual(txt_entries[0].string_type, "txt")

            self.assertEqual(ts_entries[0].status, process.EntryStatus.FUZZY)
            self.assertEqual(resx_entries[0].status, process.EntryStatus.UNTRANSLATED)
            self.assertEqual(resx_entries[1].status, process.EntryStatus.SKIPPED)
            self.assertEqual(txt_entries[0].status, process.EntryStatus.UNTRANSLATED)
            self.assertEqual(txt_entries[1].status, process.EntryStatus.SKIPPED)
        finally:
            for path in (po_path, po_out, ts_path, ts_out, resx_path, resx_out, txt_path, txt_out):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_txt_translates_per_line_and_preserves_blank_lines(self):
        in_path = os.path.join(os.getcwd(), "_tmp_lines.txt")
        out_path = os.path.join(os.getcwd(), "_tmp_lines.ai-translated.txt")
        try:
            content = "Open file\n\nSave file\n"
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, save_callback, out_path = process.load_txt(in_path)
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[0].msgid, "Open file")
            self.assertEqual(entries[0].prompt_context, "line:1")
            self.assertEqual(entries[1].msgid, "")
            self.assertEqual(entries[1].status, process.EntryStatus.SKIPPED)
            self.assertEqual(entries[2].msgid, "Save file")
            self.assertEqual(entries[2].prompt_context, "line:3")

            applied_0 = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="Ashu fail"),
            )
            applied_2 = process.apply_translation_to_entry(
                entries[2],
                process.TranslationResult(text="Saqtau fail"),
            )
            self.assertTrue(applied_0)
            self.assertTrue(applied_2)
            save_callback()

            with open(out_path, "r", encoding="utf-8") as f:
                out_text = f.read()
            self.assertEqual(out_text, "Ashu fail\n\nSaqtau fail\n")
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_strings_treats_commented_entries_as_untranslated(self):
        in_path = os.path.join(os.getcwd(), "_tmp_sample.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_sample.ai-translated.strings")
        try:
            content = (
                "/* File or program name: sample-app\n"
                "   Entry type: Name */\n"
                '/* "app|Name" = "Document Viewer"; */\n'
                '"app|Comment" = "Already translated";\n'
            )
            with open(in_path, "w", encoding="utf-16", newline="") as f:
                f.write(content)

            entries, save_callback, out_path = process.load_strings(in_path)
            self.assertEqual(len(entries), 2)
            self.assertFalse(entries[0].translated())
            self.assertTrue(entries[1].translated())
            self.assertEqual(entries[0].msgid, "Document Viewer")
            self.assertEqual(entries[0].prompt_context, "app|Name")
            self.assertIn("File or program name: sample-app", entries[0].prompt_note)
            self.assertIn("Entry type: Name", entries[0].prompt_note)

            applied = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="Hujat korsetkishi"),
            )
            self.assertTrue(applied)
            save_callback()

            with open(out_path, "rb") as f:
                head = f.read(2)
            self.assertEqual(head, b"\xff\xfe")

            with open(out_path, "r", encoding="utf-16") as f:
                out_text = f.read()
            self.assertIn('"app|Name" = "Hujat korsetkishi";', out_text)
            self.assertIn('"app|Comment" = "Already translated";', out_text)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_strings_decodes_and_reencodes_escaped_text(self):
        in_path = os.path.join(os.getcwd(), "_tmp_escaped.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_escaped.ai-translated.strings")
        try:
            content = '/* "quote\\\\\\"key" = "Line\\\\nTwo"; */\n'
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, save_callback, out_path = process.load_strings(in_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].prompt_context, 'quote\\"key')
            self.assertEqual(entries[0].msgid, "Line\\nTwo")

            process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text='A "quoted" line'),
            )
            save_callback()

            with open(out_path, "r", encoding="utf-8") as f:
                out_text = f.read()
            self.assertIn('"quote\\\\\\"key" = "A \\"quoted\\" line";', out_text)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_strings_normalizes_literal_escapes_from_model(self):
        in_path = os.path.join(os.getcwd(), "_tmp_escape_norm.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_escape_norm.ai-translated.strings")
        try:
            content = '/* "a|newline" = "Line\\nTwo"; */\n'
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, save_callback, out_path = process.load_strings(in_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].msgid, "Line\nTwo")

            # Simulate model returning a literal escape sequence instead of newline char.
            applied = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="AAA\\nBBB"),
            )
            self.assertTrue(applied)
            save_callback()

            with open(out_path, "r", encoding="utf-8") as f:
                out_text = f.read()
            self.assertIn('"a|newline" = "AAA\\nBBB";', out_text)
            self.assertNotIn('"a|newline" = "AAA\\\\nBBB";', out_text)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_strings_normalizes_literal_bell_escape_from_model(self):
        in_path = os.path.join(os.getcwd(), "_tmp_escape_bell.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_escape_bell.ai-translated.strings")
        try:
            content = '/* "a|bell" = "Beep\\aDone"; */\n'
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, save_callback, out_path = process.load_strings(in_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].msgid, "Beep\aDone")

            applied = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="AAA\\aBBB"),
            )
            self.assertTrue(applied)
            save_callback()

            with open(out_path, "r", encoding="utf-8") as f:
                out_text = f.read()
            self.assertIn('"a|bell" = "AAA\\aBBB";', out_text)
            self.assertNotIn('"a|bell" = "AAA\\\\aBBB";', out_text)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_strings_attaches_leading_comments_to_active_entry(self):
        in_path = os.path.join(os.getcwd(), "_tmp_note_active.strings")
        out_path = os.path.join(os.getcwd(), "_tmp_note_active.ai-translated.strings")
        try:
            content = (
                "/* Section within .desktop file: Desktop Entry\n"
                "   Entry type: Comment */\n"
                '"app|Comment" = "Already translated";\n'
            )
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            entries, _, _ = process.load_strings(in_path)
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].translated())
            self.assertIn("Section within .desktop file: Desktop Entry", entries[0].prompt_note)
            self.assertIn("Entry type: Comment", entries[0].prompt_note)
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_write_text_with_encoding_fallback_uses_utf8_sig_when_needed(self):
        out_path = os.path.join(os.getcwd(), "_tmp_encoding_fallback.txt")
        try:
            used_encoding = process._write_text_with_encoding_fallback(
                out_path,
                "Қазақша мәтін",
                "ascii",
                newline="",
            )

            self.assertEqual(used_encoding, "utf-8-sig")
            with open(out_path, "rb") as f:
                self.assertTrue(f.read().startswith(b"\xef\xbb\xbf"))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_build_language_code_candidates_include_exact_locale_variants_only(self):
        candidates = process.build_language_code_candidates("fr_CA")
        self.assertIn("fr_CA", candidates)
        self.assertIn("fr-CA", candidates)
        self.assertNotIn("fr", candidates)

    def test_detect_default_text_resource_prefers_exact_match(self):
        with patch("core.resources.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "locales", "fr_CA", "rules.md"),
                os.path.join("data", "locales", "fr", "rules.md"),
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, os.path.join("data", "locales", "fr_CA", "rules.md"))

    def test_detect_default_text_resource_does_not_fall_back_to_base_language(self):
        with patch("core.resources.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "locales", "fr", "rules.md")
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertIsNone(resolved)

    def test_detect_default_text_resource_uses_exact_legacy_path(self):
        with patch("core.resources.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path == "rules-fr_CA.md"
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, "rules-fr_CA.md")

    def test_detect_default_text_resource_supports_glossary_directory(self):
        glossary_dir = os.path.join("data", "locales", "fr", "glossary")
        with (
            patch("core.resources.os.path.isfile", return_value=False),
            patch("core.resources.os.path.isdir") as mocked_isdir,
        ):
            mocked_isdir.side_effect = lambda path: path == glossary_dir
            resolved = process.detect_default_text_resource(
                "glossary",
                "po",
                "fr",
                allow_directory=True,
            )

        self.assertEqual(resolved, glossary_dir)

    def test_run_translation_skips_outputs_for_jobs_with_no_work_items(self):
        active_job = process.TranslationFileJob(
            file_path="active.po",
            file_kind=process.FileKind.PO,
            entries=[],
            save_callback=Mock(),
            output_path="active.ai-translated.po",
        )
        idle_job = process.TranslationFileJob(
            file_path="idle.po",
            file_kind=process.FileKind.PO,
            entries=[],
            save_callback=Mock(),
            output_path="idle.ai-translated.po",
        )
        runtime_context = SimpleNamespace(
            provider=SimpleNamespace(name="gemini", supports_flex_mode=False),
            client=object(),
            resources=SimpleNamespace(
                glossary_text=None,
                project_rules=None,
                glossary_source="none",
                rules_source=None,
            ),
        )
        active_item = process.TranslationQueueItem(job=active_job, entry=object())

        with (
            patch("tasks.translate.resolve_translation_input_paths", return_value=["active.po", "idle.po"]),
            patch("tasks.translate.validate_translation_files", return_value=process.FileKind.PO),
            patch("tasks.translate.load_translation_jobs", return_value=[active_job, idle_job]),
            patch("tasks.translate.build_task_runtime_context", return_value=runtime_context),
            patch("tasks.translate.build_translation_queue", return_value=[active_item]),
            patch("tasks.translate.resolve_runtime_limits", return_value=(10, 1, "manual")),
            patch("tasks.translate.build_translation_generation_config", return_value=object()),
            patch("tasks.translate.build_scoped_vocabulary_entries", return_value=[]),
            patch("tasks.translate.build_batches", return_value=[[active_item]]),
            patch(
                "tasks.translate.run_translation_batches",
                new=AsyncMock(return_value=(1, {"active.ai-translated.po"})),
            ),
            patch(
                "tasks.translate.write_translation_warning_report",
                return_value="active.translation-warnings.json",
            ) as report_mock,
            patch("builtins.print"),
        ):
            process.run_translation(
                process.TranslationRunConfig(
                    files=["selected-root"],
                    source_file=None,
                    source_lang="en",
                    target_lang="kk",
                    provider="gemini",
                    model="gemini-test",
                    thinking_level=None,
                    batch_size=None,
                    parallel_requests=None,
                    glossary=None,
                    rules=None,
                    rules_str=None,
                    retranslate_all=False,
                    flex_mode=False,
                    warnings_report=True,
                )
            )

        active_job.save_callback.assert_called_once()
        idle_job.save_callback.assert_not_called()
        report_mock.assert_called_once()
        self.assertEqual(report_mock.call_args.kwargs["job"], active_job)

    def test_run_translation_with_no_work_items_writes_no_outputs(self):
        idle_job = process.TranslationFileJob(
            file_path="idle.po",
            file_kind=process.FileKind.PO,
            entries=[],
            save_callback=Mock(),
            output_path="idle.ai-translated.po",
        )
        runtime_context = SimpleNamespace(
            provider=SimpleNamespace(name="gemini", supports_flex_mode=False),
            client=object(),
            resources=SimpleNamespace(
                glossary_text=None,
                project_rules=None,
                glossary_source="none",
                rules_source=None,
            ),
        )

        with (
            patch("tasks.translate.resolve_translation_input_paths", return_value=["idle.po"]),
            patch("tasks.translate.validate_translation_files", return_value=process.FileKind.PO),
            patch("tasks.translate.load_translation_jobs", return_value=[idle_job]),
            patch("tasks.translate.build_task_runtime_context", return_value=runtime_context),
            patch("tasks.translate.build_translation_queue", return_value=[]),
            patch("tasks.translate.write_translation_warning_report") as report_mock,
            patch("builtins.print"),
        ):
            process.run_translation(
                process.TranslationRunConfig(
                    files=["selected-root"],
                    source_file=None,
                    source_lang="en",
                    target_lang="kk",
                    provider="gemini",
                    model="gemini-test",
                    thinking_level=None,
                    batch_size=None,
                    parallel_requests=None,
                    glossary=None,
                    rules=None,
                    rules_str=None,
                    retranslate_all=False,
                    flex_mode=False,
                    warnings_report=True,
                )
            )

        idle_job.save_callback.assert_not_called()
        report_mock.assert_not_called()

    def test_resolve_resource_path_prefers_explicit_path(self):
        with patch("core.resources.detect_default_text_resource") as mocked_detect:
            resolved = process.resolve_resource_path("custom-rules.md", "rules", "md", "fr")

        mocked_detect.assert_not_called()
        self.assertEqual(resolved, "custom-rules.md")

    def test_resolve_translation_input_paths_includes_xliff_files(self):
        root_dir = os.path.join(os.getcwd(), "_tmp_translate_xliff_tree")
        xliff_path = os.path.join(root_dir, "messages.xliff")
        xlf_path = os.path.join(root_dir, "messages.xlf")
        ignored = os.path.join(root_dir, "notes.md")
        try:
            os.makedirs(root_dir, exist_ok=True)
            with open(xliff_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    '<?xml version="1.0" encoding="utf-8"?>\n'
                    '<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">\n'
                    "  <file><body/></file>\n"
                    "</xliff>\n"
                )
            with open(xlf_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    '<?xml version="1.0" encoding="utf-8"?>\n'
                    '<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">\n'
                    "  <file><body/></file>\n"
                    "</xliff>\n"
                )
            with open(ignored, "w", encoding="utf-8") as handle:
                handle.write("ignore me\n")

            resolved = process.resolve_translation_input_paths([root_dir])

            self.assertEqual(resolved, [xlf_path, xliff_path])
        finally:
            for path in (xliff_path, xlf_path, ignored):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(root_dir):
                os.rmdir(root_dir)

    def test_load_entries_for_translation_supports_xliff(self):
        entries = [
            process.UnifiedEntry(
                file_kind=process.FileKind.XLIFF,
                msgid="Open",
                msgstr="",
                status=process.EntryStatus.UNTRANSLATED,
            )
        ]

        with patch("tasks.translate.load_xliff", return_value=(entries, Mock(), "out.xliff")) as mocked_load:
            file_kind, loaded_entries, save_callback, output_path, warnings = process.load_entries_for_translation(
                "messages.xliff"
            )

        mocked_load.assert_called_once_with("messages.xliff")
        self.assertEqual(file_kind, process.FileKind.XLIFF)
        self.assertEqual(loaded_entries, entries)
        self.assertEqual(output_path, "out.xliff")
        self.assertEqual(warnings, [])
        self.assertTrue(callable(save_callback))


if __name__ == "__main__":
    unittest.main()

