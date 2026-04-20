import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import polib
from google.genai import types as genai_types

from tasks import extract_terms
from tasks import translate as process


class _DummyResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class _DummyProvider:
    name = "gemini"
    default_model = "gemini-test"
    api_key_env = "GOOGLE_API_KEY"
    supports_structured_json = True
    supports_thinking = True

    def __init__(self, response=None):
        self.response = response or _DummyResponse(parsed={"terms": []})

    def create_client_from_env(self, *, flex_mode: bool = False):
        return object()

    def build_generation_config(self, *, thinking_level, json_schema, system_instruction, flex_mode=False):
        return object()

    async def generate_with_retry(self, *, client, model, contents, batch_label, max_attempts, config):
        return self.response


class _DummyEntry:
    def __init__(
        self,
        msgid: str,
        msgid_plural: str | None = None,
        obsolete: bool = False,
        include_in_term_extraction: bool = True,
        msgctxt: str = "",
        comment: str = "",
        tcomment: str = "",
    ):
        self.msgid = msgid
        self.msgid_plural = msgid_plural
        self.obsolete = obsolete
        self.include_in_term_extraction = include_in_term_extraction
        self.msgctxt = msgctxt
        self.comment = comment
        self.tcomment = tcomment
        self.occurrences = []


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
        self.assertEqual(
            messages,
            [
                {"source": "Open file"},
                {"source": "Files"},
                {"source": "Files plural"},
            ],
        )

    def test_collect_source_messages_skips_entries_marked_for_exclusion(self):
        entries = [
            _DummyEntry("Include me"),
            _DummyEntry("Skip me", include_in_term_extraction=False),
        ]
        messages = extract_terms.collect_source_messages(entries)
        self.assertEqual(messages, [{"source": "Include me"}])

    def test_collect_source_messages_includes_context_and_note(self):
        entries = [
            _DummyEntry(
                "Open",
                msgctxt="toolbar",
                comment="Primary action",
            )
        ]
        messages = extract_terms.collect_source_messages(entries)
        self.assertEqual(
            messages,
            [
                {
                    "source": "Open",
                    "context": "toolbar",
                    "note": "Primary action",
                }
            ],
        )

    def test_build_term_generation_config_includes_thinking_level(self):
        config = extract_terms.build_term_generation_config("medium")
        self.assertEqual(
            config.thinking_config.thinking_level,
            genai_types.ThinkingLevel.MEDIUM,
        )
        self.assertIn("software localization terminology analyst", config.system_instruction)

    def test_build_term_system_instruction_prefers_atomic_terms(self):
        instruction = extract_terms.build_term_system_instruction("kk")
        self.assertIn("Prefer atomic reusable terms", instruction)
        self.assertIn("access token", instruction)
        self.assertIn("split loose modifier+noun", instruction)

    def test_build_terms_prompt_requires_atomic_term_extraction(self):
        prompt = extract_terms.build_terms_prompt(
            messages={"0": {"source": "Choose audio channel", "context": "Audio output"}},
            source_lang="en",
            target_lang="kk",
            mode="all",
            vocabulary=None,
            max_terms_per_batch=25,
        )
        self.assertIn("Default to the smallest standalone reusable term", prompt)
        self.assertIn("glossary-ready terms", prompt)
        self.assertIn('"context": "Audio output"', prompt)

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
        self.assertEqual(messages, [{"source": "Save"}, {"source": "Saves"}])

    def test_load_entries_for_txt_file(self):
        in_path = os.path.join(os.getcwd(), "_tmp_terms.txt")
        out_path = os.path.join(os.getcwd(), "_tmp_terms.ai-translated.txt")
        try:
            with open(in_path, "w", encoding="utf-8", newline="") as f:
                f.write("Open file\n\nSave file\n")

            entries = extract_terms.load_entries_for_file(in_path, process.FileKind.TXT)
            messages = extract_terms.collect_source_messages(entries)
            self.assertEqual(
                messages,
                [
                    {"source": "Open file", "context": "line:1"},
                    {"source": "Save file", "context": "line:3"},
                ],
            )
        finally:
            for path in (in_path, out_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_entries_for_xliff_file(self):
        in_path = os.path.join(os.getcwd(), "_tmp_terms.xliff")
        out_path = os.path.join(os.getcwd(), "_tmp_terms.ai-translated.xliff")
        try:
            with open(in_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file original="messages.json">
    <body>
      <trans-unit id="tokenLabel" resname="tokenLabel">
        <source>Access token</source>
        <note>Shown in settings.</note>
      </trans-unit>
    </body>
  </file>
</xliff>
"""
                )

            entries = extract_terms.load_entries_for_file(in_path, process.FileKind.XLIFF)
            messages = extract_terms.collect_source_messages(entries)
            self.assertEqual(
                messages,
                [
                    {
                        "source": "Access token",
                        "context": "tokenLabel",
                        "note": "file: messages.json | Shown in settings.",
                    }
                ],
            )
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

    def test_save_terms_as_po_sets_wrapwidth_78(self):
        class _CapturePO:
            def __init__(self):
                self.wrapwidth = None
                self.metadata = None
                self.entries = []
                self.saved_path = None

            def append(self, entry):
                self.entries.append(entry)

            def save(self, path):
                self.saved_path = path

        fake_po = _CapturePO()

        with patch("tasks.extract_terms.polib.POFile", return_value=fake_po):
            extract_terms.save_terms_as_po(
                terms=[
                    extract_terms.TermCandidate(
                        source_term="Addon",
                        suggested_translation="Qosymsha",
                        reason="UI noun",
                        example_source="Install addon",
                    )
                ],
                out_path="glossary.po",
                source_lang="en",
                target_lang="kk",
            )

        self.assertEqual(fake_po.wrapwidth, extract_terms.PO_WRAP_WIDTH)
        self.assertEqual(fake_po.saved_path, "glossary.po")
        self.assertEqual(len(fake_po.entries), 1)

    def test_save_terms_as_po_merges_base_glossary_pairs_first(self):
        out_path = os.path.join(os.getcwd(), "_tmp_glossary_merged.po")
        try:
            extract_terms.save_terms_as_po(
                terms=[
                    extract_terms.TermCandidate(
                        source_term="open",
                        suggested_translation="ashu-new",
                        reason="UI verb",
                        example_source="Open file",
                    ),
                    extract_terms.TermCandidate(
                        source_term="edit",
                        suggested_translation="tuzetu",
                        reason="UI verb",
                        example_source="Edit menu",
                    ),
                ],
                out_path=out_path,
                source_lang="en",
                target_lang="kk",
                base_glossary_pairs=[("open", "ashu"), ("save", "saqtau")],
            )

            po = polib.pofile(out_path)
            by_msgid = {entry.msgid: entry for entry in po}
            self.assertEqual(len(po), 3)
            self.assertEqual(by_msgid["open"].msgstr, "ashu")
            self.assertNotIn("fuzzy", by_msgid["open"].flags)
            self.assertEqual(by_msgid["save"].msgstr, "saqtau")
            self.assertEqual(by_msgid["edit"].msgstr, "tuzetu")
            self.assertIn("fuzzy", by_msgid["edit"].flags)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_main_auto_loads_vocabulary_for_mode_all(self):
        runtime_context = SimpleNamespace(
            provider=_DummyProvider(),
            client=object(),
            resources=SimpleNamespace(
                glossary_text=None,
                glossary_source=os.path.join("data", "locales", "kk", "glossary.po"),
                glossary_pairs=[],
            ),
        )
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
            patch("tasks.extract_terms.build_task_runtime_context", return_value=runtime_context) as runtime_mock,
            patch("tasks.extract_terms.detect_file_kind", return_value=process.FileKind.TXT),
            patch("tasks.extract_terms.load_entries_for_file", return_value=[]),
            patch(
                "tasks.extract_terms.sys.argv",
                ["extract_terms.py", "input.po", "--target-lang", "kk", "--mode", "all"],
            ),
            patch("builtins.print"),
        ):
            extract_terms.main()

        runtime_mock.assert_called_once_with(
            provider_name="gemini",
            target_lang="kk",
            flex_mode=False,
            seed=None,
            explicit_glossary_path=None,
            include_rules=False,
            load_glossary_pairs_flag=False,
        )

    def test_main_missing_po_output_loads_glossary_pairs_for_merged_po(self):
        provider = _DummyProvider(response=_DummyResponse(parsed={"terms": []}))
        runtime_context = SimpleNamespace(
            provider=provider,
            client=object(),
            resources=SimpleNamespace(
                glossary_text="save - saqtau",
                glossary_source=os.path.join("data", "locales", "kk", "glossary.po"),
                glossary_pairs=[("save", "saqtau")],
            ),
        )
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False),
            patch("tasks.extract_terms.build_task_runtime_context", return_value=runtime_context) as runtime_mock,
            patch("tasks.extract_terms.detect_file_kind", return_value=process.FileKind.TXT),
            patch("tasks.extract_terms.load_entries_for_file", return_value=[_DummyEntry("Open file")]),
            patch("tasks.extract_terms.normalize_limits", return_value=(100, 1, "manual")),
            patch(
                "tasks.extract_terms.sys.argv",
                [
                    "extract_terms.py",
                    "input.po",
                    "--target-lang",
                    "kk",
                    "--mode",
                    "missing",
                    "--out-format",
                    "po",
                ],
            ),
            patch("builtins.print"),
        ):
            extract_terms.main()

        runtime_mock.assert_called_once_with(
            provider_name="gemini",
            target_lang="kk",
            flex_mode=False,
            seed=None,
            explicit_glossary_path=None,
            include_rules=False,
            load_glossary_pairs_flag=True,
        )


if __name__ == "__main__":
    unittest.main()
