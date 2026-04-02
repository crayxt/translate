import unittest
from unittest.mock import patch
import os
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
        self.assertIn("MUST:", process.SYSTEM_INSTRUCTION)
        self.assertIn("line-wrapping markers", process.SYSTEM_INSTRUCTION)

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
        self.assertIn("mandatory, not advisory", prompt.lower())
        self.assertIn("run a silent vocabulary audit", prompt.lower())
        self.assertIn("Return only the corrected final JSON.", prompt)
        self.assertIn("prefer the source plural form as the basis for translation", prompt)
        self.assertIn("numeric placeholder", prompt.lower())

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

    def test_build_prompt_no_longer_embeds_system_instruction(self):
        prompt = process.build_prompt(
            messages={"0": {"source": "Open file"}},
            source_lang="en",
            target_lang="kk",
            vocabulary=None,
            translation_rules=None,
        )
        self.assertNotIn("You are a professional software localization translator.", prompt)

    def test_read_optional_vocabulary_file_supports_po_glossary(self):
        vocab_path = os.path.join(os.getcwd(), "_tmp_vocab_glossary.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="addon", msgstr="qosymsha"))
            po.append(polib.POEntry(msgid="save as", msgstr="qalaysha saqtau"))
            po.save(vocab_path)

            vocabulary = process.read_optional_vocabulary_file(vocab_path)

            self.assertEqual(vocabulary, "addon - qosymsha\nsave as - qalaysha saqtau")
        finally:
            if os.path.exists(vocab_path):
                os.remove(vocab_path)

    def test_parse_vocabulary_line_parses_source_target_pairs(self):
        parsed = process.parse_vocabulary_line("save as - qalaysha saqtau")
        self.assertEqual(parsed, ("save as", "qalaysha saqtau"))

    def test_load_vocabulary_pairs_from_txt_ignores_comments_and_last_duplicate_wins(self):
        vocab_path = os.path.join(os.getcwd(), "_tmp_vocab_pairs.txt")
        try:
            with open(vocab_path, "w", encoding="utf-8") as f:
                f.write("# comment\n")
                f.write("save - saqtau\n")
                f.write("save - qoru\n")
                f.write("open - ashu\n")

            pairs = process.load_vocabulary_pairs(vocab_path)

            self.assertEqual(pairs, [("save", "qoru"), ("open", "ashu")])
        finally:
            if os.path.exists(vocab_path):
                os.remove(vocab_path)

    def test_read_optional_vocabulary_file_skips_untranslated_fuzzy_and_obsolete_po_entries(self):
        vocab_path = os.path.join(os.getcwd(), "_tmp_vocab_glossary_filtered.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="addon", msgstr="qosymsha"))
            po.append(polib.POEntry(msgid="blank target", msgstr=""))
            po.append(polib.POEntry(msgid="needs review", msgstr="tekseru", flags=["fuzzy"]))
            obsolete = polib.POEntry(msgid="old term", msgstr="old target", obsolete=True)
            po.append(obsolete)
            po.save(vocab_path)

            vocabulary = process.read_optional_vocabulary_file(vocab_path)

            self.assertEqual(vocabulary, "addon - qosymsha")
        finally:
            if os.path.exists(vocab_path):
                os.remove(vocab_path)

    def test_load_po_uses_wrapwidth_78(self):
        with patch("tasks.translate.polib.pofile", return_value=polib.POFile()) as mocked_pofile:
            process.load_po("sample.po")

        mocked_pofile.assert_called_once_with("sample.po", wrapwidth=process.PO_WRAP_WIDTH)

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

    def test_build_prompt_message_payload_plural_adds_required_form_count(self):
        message = ET.fromstring(
            "<message numerus='yes'>"
            "<source>%n file(s)</source>"
            "<translation><numerusform></numerusform><numerusform></numerusform></translation>"
            "</message>"
        )
        entry = process.TSEntryAdapter(message, context_name="Files")

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["plural_forms"], 2)
        self.assertIn("plural forms required: 2", payload["note"])

    def test_build_prompt_message_payload_plural_adds_plural_basis_note(self):
        entry = polib.POEntry(msgid="One file deleted", msgid_plural="%d files deleted")
        entry.msgstr_plural = {0: "", 1: ""}

        payload = process.build_prompt_message_payload(entry)

        self.assertEqual(payload["plural_forms"], 2)
        self.assertIn("no plural difference", payload["note"])
        self.assertIn("prefer the source plural variant as the basis for translation", payload["note"])

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
        with self.assertRaisesRegex(ValueError, "same format"):
            process.validate_translation_files(["first.po", "second.ts"])

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

        translated_count = process.asyncio.run(
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
                vocabulary_text=None,
                project_rules=None,
            )
        )

        self.assertEqual(translated_count, 2)
        self.assertEqual(entry_a.msgstr, "Ashu")
        self.assertEqual(entry_b.msgstr, "Saqtau")
        self.assertIn("fuzzy", entry_a.flags)
        self.assertIn("fuzzy", entry_b.flags)
        self.assertCountEqual(saved, ["one", "two"])

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

    def test_build_language_code_candidates_include_locale_and_base(self):
        candidates = process.build_language_code_candidates("fr_CA")
        self.assertIn("fr_CA", candidates)
        self.assertIn("fr", candidates)

    def test_detect_default_text_resource_prefers_exact_match(self):
        with patch("tasks.translate.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "fr_CA", "rules.md"),
                os.path.join("data", "fr", "rules.md"),
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, os.path.join("data", "fr_CA", "rules.md"))

    def test_detect_default_text_resource_falls_back_to_base_language(self):
        with patch("tasks.translate.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path in {
                os.path.join("data", "fr", "rules.md")
            }
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, os.path.join("data", "fr", "rules.md"))

    def test_detect_default_text_resource_uses_legacy_fallback(self):
        with patch("tasks.translate.os.path.isfile") as mocked_exists:
            mocked_exists.side_effect = lambda path: path == "rules-fr.md"
            resolved = process.detect_default_text_resource("rules", "md", "fr_CA")

        self.assertEqual(resolved, "rules-fr.md")

    def test_resolve_resource_path_prefers_explicit_path(self):
        with patch("tasks.translate.detect_default_text_resource") as mocked_detect:
            resolved = process.resolve_resource_path("custom-rules.md", "rules", "md", "fr")

        mocked_detect.assert_not_called()
        self.assertEqual(resolved, "custom-rules.md")


if __name__ == "__main__":
    unittest.main()

