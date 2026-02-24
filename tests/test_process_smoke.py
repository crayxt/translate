import unittest

import polib

import process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class ProcessSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
