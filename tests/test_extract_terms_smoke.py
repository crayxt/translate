import unittest
import os

import polib

import extract_terms
import process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _DummyEntry:
    def __init__(
        self,
        msgid: str,
        msgid_plural: str | None = None,
        obsolete: bool = False,
        include_in_term_extraction: bool = True,
    ):
        self.msgid = msgid
        self.msgid_plural = msgid_plural
        self.obsolete = obsolete
        self.include_in_term_extraction = include_in_term_extraction


class ExtractTermsSmokeTests(unittest.TestCase):
    def test_build_term_output_path_defaults_to_all_po(self):
        path = r"C:\work\file.po"
        out = extract_terms.build_term_output_path(path)
        self.assertEqual(out, r"C:\work\file.glossary.po")

    def test_build_term_output_path_supports_missing_json(self):
        path = r"C:\work\file.po"
        out = extract_terms.build_term_output_path(path, output_format="json", mode="missing")
        self.assertEqual(out, r"C:\work\file.missing-terms.json")

    def test_collect_source_messages_dedupes_and_skips_non_alpha(self):
        entries = [
            _DummyEntry("Open file"),
            _DummyEntry("Open file"),
            _DummyEntry("12345"),
            _DummyEntry("Files", msgid_plural="Files plural"),
            _DummyEntry("Obsolete term", obsolete=True),
        ]
        messages = extract_terms.collect_source_messages(entries)
        self.assertEqual(messages, ["Open file", "Files", "Files plural"])

    def test_collect_source_messages_skips_entries_marked_for_exclusion(self):
        entries = [
            _DummyEntry("Include me"),
            _DummyEntry("Skip me", include_in_term_extraction=False),
        ]
        messages = extract_terms.collect_source_messages(entries)
        self.assertEqual(messages, ["Include me"])

    def test_parse_term_response_from_parsed_payload(self):
        payload = {
            "terms": [
                {
                    "source_term": "token",
                    "suggested_translation": "белгі",
                    "reason": "Common technical UI term",
                    "example_source": "Invalid token",
                }
            ]
        }
        results = extract_terms.parse_term_response(_DummyResponse(parsed=payload))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_term, "token")

    def test_parse_term_response_from_json_text(self):
        text = (
            '{"terms":[{"source_term":"addon","suggested_translation":"қосымша",'
            '"reason":"Product UI term","example_source":"Install addon"}]}'
        )
        results = extract_terms.parse_term_response(_DummyResponse(text=text))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].suggested_translation, "қосымша")

    def test_merge_term_candidates_is_case_insensitive(self):
        candidates = [
            extract_terms.TermCandidate(
                source_term="Addon",
                suggested_translation="қосымша",
                reason="A",
                example_source="Install addon",
            ),
            extract_terms.TermCandidate(
                source_term="addon",
                suggested_translation="",
                reason="",
                example_source="",
            ),
        ]
        merged = extract_terms.merge_term_candidates(candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_term, "Addon")
        self.assertEqual(merged[0].suggested_translation, "қосымша")

    def test_collect_messages_from_polib_entry(self):
        entry = polib.POEntry(msgid="Save", msgid_plural="Saves")
        messages = extract_terms.collect_source_messages([entry])
        self.assertEqual(messages, ["Save", "Saves"])

    def test_load_entries_for_txt_file(self):
        in_path = os.path.join(os.getcwd(), "_tmp_terms.txt")
        out_path = os.path.join(os.getcwd(), "_tmp_terms.ai-translated.txt")
        try:
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write("Open file\n\nSave file\n")

            entries = extract_terms.load_entries_for_file(in_path, process.FileKind.TXT)
            messages = extract_terms.collect_source_messages(entries)
            self.assertEqual(messages, ["Open file", "Save file"])
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_save_terms_as_po_writes_msgid_msgstr_and_notes(self):
        out_path = os.path.join(os.getcwd(), "_tmp_glossary.po")
        try:
            terms = [
                extract_terms.TermCandidate(
                    source_term="Addon",
                    suggested_translation="Qosymsha",
                    reason="UI noun",
                    example_source="Install addon",
                )
            ]

            extract_terms.save_terms_as_po(
                terms=terms,
                out_path=out_path,
                source_lang="en",
                target_lang="kk",
            )

            po = polib.pofile(out_path)
            self.assertEqual(len(po), 1)
            self.assertEqual(po[0].msgid, "Addon")
            self.assertEqual(po[0].msgstr, "Qosymsha")
            self.assertIn("fuzzy", po[0].flags)
            self.assertIn("Reason: UI noun", po[0].tcomment)
            self.assertIn("Example: Install addon", po[0].tcomment)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
