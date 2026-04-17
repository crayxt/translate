import os
import re
import shutil
import unittest

import polib

from core.glossary_catalog import (
    GlossarySourceTerm,
    build_fallback_context_note,
    build_glossary_entry_id,
    build_glossary_pot,
    build_glossary_source_terms_from_records,
    build_locale_glossary_po_from_records,
    load_glossary_source_terms,
    suggest_glossary_sense,
    sync_locale_glossary_po,
    write_catalog,
)


class GlossaryCatalogSmokeTests(unittest.TestCase):
    def setUp(self):
        self._temp_paths: list[str] = []

    def tearDown(self):
        for path in reversed(self._temp_paths):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

    def _make_temp_dir(self, name: str) -> str:
        path = os.path.join(os.getcwd(), name)
        os.makedirs(path, exist_ok=True)
        self._temp_paths.append(path)
        return path

    def test_load_glossary_source_terms_sorts_entries(self):
        tmpdir = self._make_temp_dir("_tmp_glossary_catalog_load")
        source_path = os.path.join(tmpdir, "glossary.jsonl")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"source_term":"line","part_of_speech":"noun","sense":"geometry","context_note":"geometry","id":"line.noun.geometry"}\n'
            )
            handle.write(
                '{"source_term":"command line","part_of_speech":"noun","sense":"cli","context_note":"cli","id":"command-line.noun.cli"}\n'
            )
            handle.write(
                '{"source_term":"line","part_of_speech":"noun","sense":"text-ui","context_note":"text ui","id":"line.noun.text-ui"}\n'
            )

        terms = load_glossary_source_terms(source_path)

        self.assertEqual(
            [item.id for item in terms],
            ["command-line.noun.cli", "line.noun.geometry", "line.noun.text-ui"],
        )

    def test_build_glossary_pot_uses_stable_ids_and_comments(self):
        terms = build_glossary_source_terms_from_records(
            [
                ("line", "сызық", "noun", "geometry, border, drawing, stroke"),
                ("line", "жол", "noun", "line of text, text layout, editor text"),
            ]
        )

        catalog = build_glossary_pot(terms)
        geometry_id = build_glossary_entry_id(
            "line",
            "noun",
            "geometry",
        )

        self.assertEqual(catalog[0].msgctxt, "geometry, border, drawing, stroke")
        self.assertEqual(catalog[0].msgid, "line")
        self.assertIn(f"ID: {geometry_id}", catalog[0].comment)
        self.assertIn("POS: noun", catalog[0].comment)
        self.assertIn("Sense: geometry", catalog[0].comment)
        self.assertNotIn("Context:", catalog[0].comment)
        self.assertNotIn("Example:", catalog[0].comment)

    def test_build_glossary_pot_sets_gettext_metadata(self):
        catalog = build_glossary_pot(
            [
                GlossarySourceTerm(
                    source_term="filter",
                    part_of_speech="noun",
                    sense="default",
                    id="filter.noun.default",
                )
            ]
        )

        self.assertEqual(catalog.metadata["Project-Id-Version"], "Glossary")
        self.assertEqual(catalog.metadata["PO-Revision-Date"], "YEAR-MO-DA HO:MI+ZONE")
        self.assertEqual(catalog.metadata["Content-Transfer-Encoding"], "8bit")
        self.assertEqual(catalog.metadata["Generated-By"], "scripts/sync_glossary_catalog.py")
        self.assertEqual(catalog.metadata["X-Glossary-Source"], "jsonl")
        self.assertRegex(
            catalog.metadata["POT-Creation-Date"],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}[+-]\d{4}$",
        )

    def test_build_glossary_entry_id_is_stable_for_identity_triplet(self):
        self.assertEqual(
            build_glossary_entry_id("line", "noun", "geometry"),
            build_glossary_entry_id("line", "noun", "geometry"),
        )
        self.assertNotEqual(
            build_glossary_entry_id("line", "noun", "geometry"),
            build_glossary_entry_id("line", "noun", "text"),
        )

    def test_suggest_glossary_sense_uses_first_context_clause(self):
        self.assertEqual(
            suggest_glossary_sense("line of text, text layout, editor text", "line"),
            "text",
        )
        self.assertEqual(
            suggest_glossary_sense("geometry, border, drawing, stroke", "line"),
            "geometry",
        )
        self.assertEqual(
            suggest_glossary_sense("command-line interface", "command line"),
            "cli",
        )

    def test_build_fallback_context_note_uses_term_and_pos(self):
        self.assertEqual(
            build_fallback_context_note("filter", "verb"),
            "action to filter in the UI or technical workflow",
        )
        self.assertEqual(
            build_fallback_context_note("background", "noun"),
            "background as a visual style, graphics, or display term",
        )

    def test_build_glossary_source_terms_from_records_ignores_target_terms_for_identity(self):
        source_terms = build_glossary_source_terms_from_records(
            [
                ("line", "жол", "noun", "line of text"),
                ("line", "сызық", "noun", "geometry"),
            ]
        )

        self.assertEqual([item.source_term for item in source_terms], ["line", "line"])
        self.assertEqual([item.sense for item in source_terms], ["geometry", "text"])
        self.assertEqual(
            [item.context_note for item in source_terms],
            ["geometry", "line of text"],
        )

    def test_build_glossary_source_terms_backfills_missing_context(self):
        source_terms = build_glossary_source_terms_from_records(
            [("filter", "сүзгі", "noun", "")]
        )
        self.assertEqual(
            source_terms[0].context_note,
            "filter as a UI, document, or technical noun",
        )

    def test_sync_locale_glossary_po_preserves_translation_and_marks_source_changes_fuzzy(self):
        terms = build_glossary_source_terms_from_records(
            [("line", "жол", "noun", "line of text, text layout, editor text")]
        )
        line_text_ui_id = build_glossary_entry_id(
            "line",
            "noun",
            "text",
        )
        existing = polib.POFile()
        existing.append(
            polib.POEntry(
                msgctxt="old text line context",
                msgid="text line",
                msgstr="жол",
                comment=f"ID: {line_text_ui_id}\nPOS: noun\nSense: text",
                tcomment="Reviewed translation",
            )
        )

        catalog = sync_locale_glossary_po(terms, locale="kk", existing_catalog=existing)
        line_entry = next(
            item for item in catalog if item.msgctxt == "line of text, text layout, editor text"
        )

        self.assertEqual(line_entry.msgid, "line")
        self.assertEqual(line_entry.msgstr, "жол")
        self.assertIn("fuzzy", line_entry.flags)
        self.assertEqual(line_entry.tcomment, "Reviewed translation")
        self.assertEqual(catalog.metadata["Language"], "kk")
        self.assertEqual(catalog.metadata["Content-Transfer-Encoding"], "8bit")
        self.assertEqual(catalog.metadata["Generated-By"], "scripts/sync_glossary_catalog.py")
        self.assertEqual(catalog.metadata["X-Glossary-Source"], "jsonl")
        self.assertRegex(
            catalog.metadata["POT-Creation-Date"],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}[+-]\d{4}$",
        )
        self.assertRegex(
            catalog.metadata["PO-Revision-Date"],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}[+-]\d{4}$",
        )

    def test_sync_locale_glossary_po_extracts_id_from_multiline_comment(self):
        target_term = "kk-filter"
        terms = build_glossary_source_terms_from_records(
            [("filter", target_term, "noun", "search or display filter")]
        )
        filter_id = terms[0].id
        existing = polib.POFile()
        existing.append(
            polib.POEntry(
                msgctxt="search or display filter",
                msgid="filter",
                msgstr=target_term,
                comment=f"ID: {filter_id}\nPOS: noun\nSense: default",
            )
        )

        catalog = sync_locale_glossary_po(terms, locale="kk", existing_catalog=existing)

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].msgstr, target_term)

    def test_build_locale_glossary_po_from_records_preserves_translations(self):
        catalog = build_locale_glossary_po_from_records(
            [
                ("line", "жол", "noun", "line of text"),
                ("line", "сызық", "noun", "geometry"),
            ],
            locale="kk",
        )

        self.assertEqual(catalog[0].msgstr, "сызық")
        self.assertEqual(catalog[1].msgstr, "жол")
        self.assertNotEqual(catalog[0].msgctxt, catalog[1].msgctxt)
        self.assertEqual(catalog[0].msgctxt, "geometry")
        self.assertEqual(catalog[1].msgctxt, "line of text")

    def test_build_glossary_pot_disambiguates_duplicate_fallback_contexts(self):
        catalog = build_glossary_pot(
            [
                GlossarySourceTerm(
                    source_term="filter",
                    part_of_speech="noun",
                    sense="default",
                    id="filter.noun.default",
                ),
                GlossarySourceTerm(
                    source_term="filter",
                    part_of_speech="verb",
                    sense="default",
                    id="filter.verb.default",
                ),
            ]
        )

        self.assertEqual(catalog[0].msgctxt, "noun")
        self.assertEqual(catalog[1].msgctxt, "verb")

    def test_write_catalog_creates_parent_directories(self):
        catalog = build_glossary_pot(
            load_glossary_source_terms(os.path.join("data", "glossary", "glossary.jsonl"))
        )
        tmpdir = self._make_temp_dir("_tmp_glossary_catalog_write")
        output_path = os.path.join(tmpdir, "nested", "glossary.pot")
        write_catalog(catalog, output_path)
        self.assertTrue(os.path.exists(output_path))
        parsed = polib.pofile(output_path, wrapwidth=78)
        self.assertEqual(len(parsed), len(catalog))


if __name__ == "__main__":
    unittest.main()
