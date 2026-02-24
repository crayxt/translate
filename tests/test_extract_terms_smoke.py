import unittest

import polib

import extract_terms


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _DummyEntry:
    def __init__(self, msgid: str, msgid_plural: str | None = None, obsolete: bool = False):
        self.msgid = msgid
        self.msgid_plural = msgid_plural
        self.obsolete = obsolete


class ExtractTermsSmokeTests(unittest.TestCase):
    def test_build_term_output_path(self):
        path = r"C:\work\file.po"
        out = extract_terms.build_term_output_path(path)
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


if __name__ == "__main__":
    unittest.main()
