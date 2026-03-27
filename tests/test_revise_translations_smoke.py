import os
import unittest

from google.genai import types as genai_types

from tasks.translate import EntryStatus, FileKind, UnifiedEntry
from tasks import revise_translations


class ReviseTranslationsSmokeTests(unittest.TestCase):
    def test_build_revision_generation_config_includes_system_instruction(self):
        config = revise_translations.build_revision_generation_config("low")
        self.assertEqual(config.thinking_config.thinking_level, genai_types.ThinkingLevel.LOW)
        self.assertIn("revising existing software localization translations", config.system_instruction)

    def test_build_revision_system_instruction_mentions_target_script(self):
        system_instruction = revise_translations.build_revision_system_instruction("kk")
        self.assertIn("revising existing software localization translations", system_instruction)
        self.assertIn("real Kazakh Cyrillic alphabet", system_instruction)

    def test_build_revision_output_path_appends_revised_suffix(self):
        self.assertEqual(
            revise_translations.build_revision_output_path(r"C:\tmp\sample.po"),
            r"C:\tmp\sample-revised.po",
        )
        self.assertEqual(
            revise_translations.build_revision_output_path(
                r"C:\tmp\sample.ai-translated.strings"
            ),
            r"C:\tmp\sample.ai-translated-revised.strings",
        )

    def test_apply_revision_to_item_updates_and_marks_fuzzy(self):
        entry = UnifiedEntry(
            file_kind=FileKind.PO,
            msgid="Save",
            msgstr="Old term",
            status=EntryStatus.TRANSLATED,
        )
        item = revise_translations.ReviewItem(
            entry=entry,
            source_text="Save",
            current_text="Old term",
            current_plural_texts=[],
            plural_form_count=0,
            context="Menu|File",
            note="",
            pair_key="Menu|File",
        )

        changed = revise_translations.apply_revision_to_item(
            item,
            revise_translations.RevisionResult(
                action="update",
                text="New term",
            ),
        )

        self.assertTrue(changed)
        self.assertEqual(entry.msgstr, "New term")
        self.assertIn("fuzzy", entry.flags)
        self.assertEqual(entry.status, EntryStatus.FUZZY)

    def test_apply_revision_to_item_skips_identical_result(self):
        entry = UnifiedEntry(
            file_kind=FileKind.PO,
            msgid="Save",
            msgstr="Keep me",
            status=EntryStatus.TRANSLATED,
        )
        item = revise_translations.ReviewItem(
            entry=entry,
            source_text="Save",
            current_text="Keep me",
            current_plural_texts=[],
            plural_form_count=0,
            pair_key="Save",
        )

        changed = revise_translations.apply_revision_to_item(
            item,
            revise_translations.RevisionResult(
                action="update",
                text="Keep me",
            ),
        )

        self.assertFalse(changed)
        self.assertEqual(entry.msgstr, "Keep me")
        self.assertNotIn("fuzzy", entry.flags)

    def test_load_review_bundle_for_po_uses_embedded_source(self):
        input_path = os.path.join(os.getcwd(), "_tmp_revision.po")
        generated_out = os.path.join(os.getcwd(), "_tmp_revision.ai-translated.po")
        try:
            with open(input_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    'msgid ""\n'
                    'msgstr ""\n'
                    '"Language: kk\\n"\n\n'
                    'msgid "Open"\n'
                    'msgstr "Ashu"\n'
                )

            bundle = revise_translations.load_review_bundle(input_path)

            self.assertEqual(bundle.file_kind, FileKind.PO)
            self.assertEqual(len(bundle.items), 1)
            self.assertEqual(bundle.items[0].source_text, "Open")
            self.assertEqual(bundle.items[0].current_text, "Ashu")
        finally:
            for path in (input_path, generated_out):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_review_bundle_pairs_strings_source_and_translation(self):
        source_path = os.path.join(os.getcwd(), "_tmp_source.strings")
        translated_path = os.path.join(os.getcwd(), "_tmp_translated.ai-translated.strings")
        generated_out = os.path.join(
            os.getcwd(),
            "_tmp_translated.ai-translated.ai-translated.strings",
        )
        try:
            with open(source_path, "w", encoding="utf-8", newline="") as handle:
                handle.write('/* "app|Name" = "Document Viewer"; */\n')
            with open(translated_path, "w", encoding="utf-8", newline="") as handle:
                handle.write('"app|Name" = "Qyjat koru quraly";\n')

            bundle = revise_translations.load_review_bundle(
                translated_path,
                source_file=source_path,
            )

            self.assertEqual(bundle.file_kind, FileKind.STRINGS)
            self.assertEqual(len(bundle.items), 1)
            self.assertEqual(bundle.items[0].source_text, "Document Viewer")
            self.assertEqual(bundle.items[0].current_text, "Qyjat koru quraly")
            self.assertEqual(bundle.items[0].context, "app|Name")
        finally:
            for path in (source_path, translated_path, generated_out):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_paired_txt_bundle_saves_revised_output(self):
        source_path = os.path.join(os.getcwd(), "_tmp_source.txt")
        translated_path = os.path.join(os.getcwd(), "_tmp_translated.txt")
        revised_path = os.path.join(os.getcwd(), "_tmp_translated-revised.txt")
        try:
            with open(source_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("Open\n\nSave\n")
            with open(translated_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("Ashu\n\nSaqtau\n")

            bundle = revise_translations.load_review_bundle(
                translated_path,
                source_file=source_path,
            )
            self.assertEqual(bundle.file_kind, FileKind.TXT)
            self.assertEqual(len(bundle.items), 2)

            changed = revise_translations.apply_revision_to_item(
                bundle.items[1],
                revise_translations.RevisionResult(
                    action="update",
                    text="Qoru",
                ),
            )
            self.assertTrue(changed)

            bundle.save_callback()

            with open(revised_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "Ashu\n\nQoru\n")
        finally:
            for path in (source_path, translated_path, revised_path):
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
