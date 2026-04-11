import os
import unittest
import xml.etree.ElementTree as ET

from core.formats import EntryStatus, FileKind, build_output_path, detect_file_kind, load_xliff


XLIFF_NS = {"x": "urn:oasis:names:tc:xliff:document:1.2"}


class XliffSmokeTests(unittest.TestCase):
    def test_detect_file_kind_and_output_path_preserve_xliff_extension(self):
        self.assertEqual(detect_file_kind(r"C:\work\messages.xlf"), FileKind.XLIFF)
        self.assertEqual(detect_file_kind(r"C:\work\messages.xliff"), FileKind.XLIFF)
        self.assertEqual(
            build_output_path(r"C:\work\messages.xlf", FileKind.XLIFF),
            r"C:\work\messages.ai-translated.xlf",
        )
        self.assertEqual(
            build_output_path(r"C:\work\messages.xliff", FileKind.XLIFF),
            r"C:\work\messages.ai-translated.xliff",
        )

    def test_load_xliff_round_trips_states_targets_and_inline_tags(self):
        in_path = os.path.join(os.getcwd(), "_tmp_sample.xliff")
        out_path = os.path.join(os.getcwd(), "_tmp_sample.ai-translated.xliff")
        try:
            with open(in_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file original="messages.json" source-language="en" target-language="kk">
    <body>
      <trans-unit id="greeting" resname="greeting">
        <source>Hello <g id="name">{{name}}</g></source>
        <note>Greeting for the popup title.</note>
      </trans-unit>
      <trans-unit id="copied" resname="copied">
        <source>Copied</source>
        <target state="needs-translation">Copied</target>
      </trans-unit>
      <trans-unit id="done" resname="done">
        <source>Done</source>
        <target state="translated">Дайын</target>
      </trans-unit>
      <trans-unit id="skip" translate="no">
        <source>Internal marker</source>
      </trans-unit>
    </body>
  </file>
</xliff>
"""
                )

            entries, save_callback, generated_out_path = load_xliff(in_path)

            self.assertEqual(generated_out_path, out_path)
            self.assertEqual(len(entries), 4)

            self.assertEqual(entries[0].msgctxt, "greeting")
            self.assertEqual(entries[0].msgid, 'Hello <g id="name">{{name}}</g>')
            self.assertEqual(entries[0].status, EntryStatus.UNTRANSLATED)
            self.assertEqual(entries[0].msgstr, "")
            self.assertIn("file: messages.json", entries[0].prompt_note)
            self.assertIn("Greeting for the popup title.", entries[0].prompt_note)

            self.assertEqual(entries[1].status, EntryStatus.UNTRANSLATED)
            self.assertEqual(entries[1].msgstr, "")

            self.assertEqual(entries[2].status, EntryStatus.TRANSLATED)
            self.assertEqual(entries[2].msgstr, "Дайын")

            self.assertEqual(entries[3].status, EntryStatus.SKIPPED)
            self.assertFalse(entries[3].include_in_term_extraction)

            entries[0].msgstr = 'Сәлем <g id="name">{{name}}</g>'
            entries[0].flags.append("fuzzy")
            entries[0].status = EntryStatus.FUZZY
            save_callback()

            tree = ET.parse(out_path)
            greeting_target = tree.find(".//x:trans-unit[@id='greeting']/x:target", XLIFF_NS)
            self.assertIsNotNone(greeting_target)
            self.assertEqual(greeting_target.get("state"), "needs-review-translation")

            saved_entries, _, _ = load_xliff(out_path)
            self.assertEqual(saved_entries[0].status, EntryStatus.FUZZY)
            self.assertEqual(saved_entries[0].msgstr, 'Сәлем <g id="name">{{name}}</g>')
            self.assertEqual(saved_entries[1].status, EntryStatus.UNTRANSLATED)
            self.assertEqual(saved_entries[1].msgstr, "")
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
