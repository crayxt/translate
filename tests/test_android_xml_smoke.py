import os
import unittest
import xml.etree.ElementTree as ET

import process_gui
from tasks import extract_terms, extract_terms_local, revise_translations
from tasks import translate as process


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


class AndroidXmlSmokeTests(unittest.TestCase):
    def test_detect_file_kind_supports_android_resources_xml(self):
        input_path = os.path.join(os.getcwd(), "_tmp_android_detect.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play">Play</string>\n'
                "</resources>\n",
            )

            self.assertEqual(
                process.detect_file_kind(input_path),
                process.FileKind.ANDROID_XML,
            )
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_load_android_xml_saves_strings_plurals_and_inline_tags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_android_source.xml")
        output_path = os.path.join(os.getcwd(), "_tmp_android_source.ai-translated.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<resources xmlns:xliff="urn:oasis:names:tc:xliff:document:1.2">\n'
                "  <!-- Playback -->\n"
                '  <string name="confirm_delete">Delete <xliff:g id="name">%1$s</xliff:g>?</string>\n'
                '  <plurals name="results">\n'
                '    <item quantity="one">%d result</item>\n'
                '    <item quantity="other">%d results</item>\n'
                "  </plurals>\n"
                '  <string name="app_name" translatable="false">VLC</string>\n'
                "</resources>\n",
            )

            entries, save_callback, _ = process.load_android_xml(input_path)
            entry_map = {entry.msgctxt: entry for entry in entries}

            self.assertEqual(entry_map["string:confirm_delete"].status, process.EntryStatus.UNTRANSLATED)
            self.assertEqual(entry_map["plurals:results"].status, process.EntryStatus.UNTRANSLATED)
            self.assertEqual(entry_map["string:app_name"].status, process.EntryStatus.SKIPPED)
            self.assertEqual(entry_map["string:confirm_delete"].string_type, "xml")

            self.assertTrue(
                process.apply_translation_to_entry(
                    entry_map["string:confirm_delete"],
                    process.TranslationResult(
                        text='Oshiru <xliff:g id="name">%1$s</xliff:g>?',
                    ),
                )
            )
            self.assertTrue(
                process.apply_translation_to_entry(
                    entry_map["plurals:results"],
                    process.TranslationResult(
                        text="",
                        plural_texts=["%d natije", "%d natijeler"],
                    ),
                )
            )
            save_callback()

            tree = ET.parse(output_path)
            root = tree.getroot()
            confirm_delete = root.find("./string[@name='confirm_delete']")
            self.assertIsNotNone(confirm_delete)
            self.assertEqual(confirm_delete.text, "Oshiru ")
            self.assertEqual(len(list(confirm_delete)), 1)
            child = list(confirm_delete)[0]
            self.assertTrue(str(child.tag).endswith("g"))
            self.assertEqual(child.text, "%1$s")
            self.assertEqual(child.tail, "?")
            self.assertEqual(root.find("./plurals[@name='results']/item[@quantity='one']").text, "%d natije")
            self.assertEqual(root.find("./plurals[@name='results']/item[@quantity='other']").text, "%d natijeler")
        finally:
            for path in (input_path, output_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_android_xml_normalizes_actual_newline_back_to_literal_escape(self):
        input_path = os.path.join(os.getcwd(), "_tmp_android_escape.xml")
        output_path = os.path.join(os.getcwd(), "_tmp_android_escape.ai-translated.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="warning">Line one\\nLine two</string>\n'
                "</resources>\n",
            )

            entries, save_callback, _ = process.load_android_xml(input_path)
            self.assertEqual(entries[0].msgid, "Line one\\nLine two")

            applied = process.apply_translation_to_entry(
                entries[0],
                process.TranslationResult(text="Birinshi zhol\nEkinshi zhol"),
            )
            self.assertTrue(applied)
            save_callback()

            with open(output_path, "r", encoding="utf-8") as handle:
                out_text = handle.read()
            self.assertIn('<string name="warning">Birinshi zhol\\nEkinshi zhol</string>', out_text)
            self.assertNotIn('<string name="warning">Birinshi zhol\nEkinshi zhol</string>', out_text)
        finally:
            for path in (input_path, output_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_validate_translation_files_requires_source_file_for_android_xml(self):
        input_path = os.path.join(os.getcwd(), "_tmp_android_target.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play"></string>\n'
                "</resources>\n",
            )

            with self.assertRaisesRegex(ValueError, "--source-file is required for .xml translation runs"):
                process.validate_translation_files([input_path])
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_load_review_bundle_pairs_android_xml_source_and_translation(self):
        source_path = os.path.join(os.getcwd(), "_tmp_android_source_review.xml")
        translated_path = os.path.join(os.getcwd(), "_tmp_android_translated_review.xml")
        output_path = os.path.join(os.getcwd(), "_tmp_android_translated_review.ai-translated.xml")
        try:
            _write_text(
                source_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play">Play</string>\n'
                '  <plurals name="results">\n'
                '    <item quantity="one">%d result</item>\n'
                '    <item quantity="other">%d results</item>\n'
                "  </plurals>\n"
                '  <string name="app_name" translatable="false">VLC</string>\n'
                "</resources>\n",
            )
            _write_text(
                translated_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play">Oynatu</string>\n'
                '  <plurals name="results">\n'
                '    <item quantity="one">%d natije</item>\n'
                '    <item quantity="other">%d natijeler</item>\n'
                "  </plurals>\n"
                "</resources>\n",
            )

            bundle = revise_translations.load_review_bundle(
                translated_path,
                source_file=source_path,
            )

            self.assertEqual(bundle.file_kind, process.FileKind.ANDROID_XML)
            self.assertEqual(len(bundle.items), 2)
            self.assertEqual(bundle.items[0].context, "string:play")
            self.assertEqual(bundle.items[0].source_text, "Play")
            self.assertEqual(bundle.items[0].current_text, "Oynatu")
            self.assertEqual(bundle.items[1].context, "plurals:results")
            self.assertEqual(
                bundle.items[1].source_text,
                "Singular: %d result\nPlural: %d results",
            )
            self.assertEqual(bundle.items[1].current_plural_texts, ["%d natije", "%d natijeler"])
        finally:
            for path in (source_path, translated_path, output_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_extract_term_loaders_support_android_xml(self):
        input_path = os.path.join(os.getcwd(), "_tmp_android_extract.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                "  <!-- Playback -->\n"
                '  <string name="play">Play</string>\n'
                '  <plurals name="results">\n'
                '    <item quantity="one">%d result</item>\n'
                '    <item quantity="other">%d results</item>\n'
                "  </plurals>\n"
                "</resources>\n",
            )

            entries = extract_terms.load_entries_for_file(input_path, process.FileKind.ANDROID_XML)
            local_entries = extract_terms_local.load_entries_for_file(input_path, process.FileKind.ANDROID_XML)
            messages = extract_terms.collect_source_messages(entries)

            self.assertEqual(len(entries), 2)
            self.assertEqual(len(local_entries), 2)
            self.assertTrue(any(item["source"] == "Play" for item in messages))
            self.assertTrue(any(item["source"] == "%d results" for item in messages))
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_process_gui_config_requires_source_file_for_android_xml(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_android_target.xml")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play"></string>\n'
                "</resources>\n",
            )

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertIn("Source file is required for Android .xml translation runs.", errors)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_build_process_command_includes_source_file_for_android_xml(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_android_target.xml")
        source_path = os.path.join(os.getcwd(), "_tmp_gui_android_source.xml")
        script_path = os.path.join(os.getcwd(), "_tmp_gui_android_script.py")
        try:
            _write_text(
                input_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play"></string>\n'
                "</resources>\n",
            )
            _write_text(
                source_path,
                '<?xml version="1.0" encoding="utf-8"?>\n'
                "<resources>\n"
                '  <string name="play">Play</string>\n'
                "</resources>\n",
            )
            _write_text(script_path, "print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                source_file=source_path,
                source_lang="en",
                target_lang="kk",
                model="gemini-test",
                api_key="test-key",
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "translate",
                    input_path,
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "kk",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "50",
                    "--parallel-requests",
                    "1",
                    "--source-file",
                    source_path,
                ],
            )
        finally:
            for path in (input_path, source_path, script_path):
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
