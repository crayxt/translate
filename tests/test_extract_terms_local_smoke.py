import argparse
import json
import os
import shutil
import unittest
from unittest.mock import patch

import polib

from core.cli_errors import CliError
from tasks import extract_terms_local


class ExtractTermsLocalSmokeTests(unittest.TestCase):
    def _cleanup_paths(self, *paths: str) -> None:
        for path in paths:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                pass

    def test_run_from_args_can_write_json_and_po_in_one_shot(self):
        input_path = os.path.abspath("_tmp_extract_terms_local.po")
        json_path = os.path.abspath("_tmp_extract_terms_local.prototype-missing-terms.json")
        po_path = os.path.abspath("_tmp_extract_terms_local.prototype-missing-terms.po")
        glossary_path = os.path.abspath("_tmp_extract_terms_local_glossary.jsonl")
        self._cleanup_paths(input_path, glossary_path, json_path, po_path)

        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("")

            args = argparse.Namespace(
                file=input_path,
                source_lang="en",
                glossary_source=None,
                to_po=False,
                also_po=True,
                mode="missing",
                max_length=2,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

            with patch.object(extract_terms_local, "DEFAULT_LOCAL_GLOSSARY_PATH", glossary_path):
                extract_terms_local.run_from_args(args)

            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(po_path))

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["output_file"], json_path)
            self.assertIsNone(payload["target_lang"])
            self.assertGreaterEqual(payload["translation_candidate_count"], 1)

            po = polib.pofile(po_path, wrapwidth=78)
            self.assertGreaterEqual(len(po), 1)
            self.assertEqual(po[0].msgid, "access token")
            self.assertFalse(po.metadata.get("Language"))
        finally:
            self._cleanup_paths(input_path, glossary_path, json_path, po_path)

    def test_run_from_args_can_extract_from_directory_tree(self):
        input_root = os.path.abspath("_tmp_extract_terms_tree")
        nested_root = os.path.join(input_root, "sub")
        po_one = os.path.join(input_root, "first.po")
        po_two = os.path.join(nested_root, "second.po")
        glossary_path = os.path.abspath("_tmp_extract_terms_tree_glossary.jsonl")
        json_path = os.path.abspath("_tmp_extract_terms_tree.json")
        po_path = os.path.abspath("_tmp_extract_terms_tree.po")
        self._cleanup_paths(input_root, glossary_path, json_path, po_path)

        try:
            os.makedirs(nested_root, exist_ok=True)
            with open(po_one, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(po_two, "w", encoding="utf-8") as handle:
                handle.write('msgid "Playback speed"\nmsgstr ""\n')
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("")

            args = argparse.Namespace(
                file=input_root,
                source_lang="en",
                glossary_source=None,
                to_po=False,
                also_po=True,
                mode="missing",
                max_length=2,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

            with patch.object(extract_terms_local, "DEFAULT_LOCAL_GLOSSARY_PATH", glossary_path):
                extract_terms_local.run_from_args(args)

            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(po_path))

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["source_file"], input_root)
            self.assertGreaterEqual(payload["total_source_messages"], 2)
            self.assertGreaterEqual(payload["translation_candidate_count"], 1)
            terms = {item["source_term"]: item for item in payload["terms"]}
            self.assertIn("access token", terms)
            self.assertEqual(terms["access token"]["file_count"], 1)
            self.assertEqual(terms["access token"]["files"], ["first.po"])

            po = polib.pofile(po_path, wrapwidth=78)
            msgids = {entry.msgid for entry in po}
            self.assertIn("access token", msgids)
        finally:
            self._cleanup_paths(input_root, glossary_path, json_path, po_path)

    def test_run_from_args_preserves_all_caps_term_in_po_handoff(self):
        input_path = os.path.abspath("_tmp_extract_terms_local_caps.po")
        glossary_path = os.path.abspath("_tmp_extract_terms_local_caps_glossary.jsonl")
        json_path = os.path.abspath("_tmp_extract_terms_local_caps.prototype-missing-terms.json")
        po_path = os.path.abspath("_tmp_extract_terms_local_caps.prototype-missing-terms.po")
        self._cleanup_paths(input_path, glossary_path, json_path, po_path)

        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write(
                    'msgctxt "Calc functions"\nmsgid "SUM"\nmsgstr ""\n\n'
                    'msgctxt "Formula help"\nmsgid "SUM"\nmsgstr ""\n'
                )
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("")

            args = argparse.Namespace(
                file=input_path,
                source_lang="en",
                glossary_source=None,
                to_po=False,
                also_po=True,
                mode="missing",
                max_length=1,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

            with patch.object(extract_terms_local, "DEFAULT_LOCAL_GLOSSARY_PATH", glossary_path):
                extract_terms_local.run_from_args(args)

            po = polib.pofile(po_path, wrapwidth=78)
            self.assertEqual(len(po), 1)
            self.assertEqual(po[0].msgid, "SUM")
        finally:
            self._cleanup_paths(input_path, glossary_path, json_path, po_path)

    def test_run_from_args_uses_default_glossary_jsonl_without_target_lang(self):
        input_path = os.path.abspath("_tmp_extract_terms_local_default_glossary.po")
        glossary_path = os.path.abspath("_tmp_extract_terms_local_default_glossary.jsonl")
        json_path = os.path.abspath("_tmp_extract_terms_local_default_glossary.json")
        self._cleanup_paths(input_path, glossary_path, json_path)

        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"source_term":"access token","part_of_speech":"noun","sense":"security","context_note":"security token","id":"access-token.noun.security","example":""}\n'
                )

            args = argparse.Namespace(
                file=input_path,
                source_lang="en",
                glossary_source=None,
                to_po=False,
                also_po=False,
                mode="missing",
                max_length=2,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

            with patch.object(extract_terms_local, "DEFAULT_LOCAL_GLOSSARY_PATH", glossary_path):
                extract_terms_local.run_from_args(args)

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["glossary_source_path"], glossary_path)
            self.assertIsNone(payload["target_lang"])
            self.assertEqual(payload["accepted_candidate_count"], 0)
            self.assertEqual(payload["translation_candidate_count"], 0)
        finally:
            self._cleanup_paths(input_path, glossary_path, json_path)

    def test_run_from_args_reports_missing_explicit_glossary_source_cleanly(self):
        input_path = os.path.abspath("_tmp_extract_terms_local_missing_glossary.po")
        self._cleanup_paths(input_path)

        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            args = argparse.Namespace(
                file=input_path,
                source_lang="en",
                glossary_source="missing-glossary.jsonl",
                to_po=False,
                also_po=False,
                mode="missing",
                max_length=1,
                out=None,
                include_rejected=False,
                include_borderline=False,
            )

            with self.assertRaises(CliError) as raised:
                extract_terms_local.run_from_args(args)

            self.assertEqual(
                str(raised.exception),
                "Glossary source file 'missing-glossary.jsonl' not found.",
            )
        finally:
            self._cleanup_paths(input_path)

    def test_directory_extraction_keeps_identical_messages_from_different_files_distinct(self):
        input_root = os.path.abspath("_tmp_extract_terms_provenance")
        nested_root = os.path.join(input_root, "sub")
        po_one = os.path.join(input_root, "first.po")
        po_two = os.path.join(nested_root, "second.po")
        self._cleanup_paths(input_root)

        try:
            os.makedirs(nested_root, exist_ok=True)
            with open(po_one, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(po_two, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')

            messages, scanned_files = extract_terms_local.load_messages_for_input(input_root)

            self.assertEqual(len(scanned_files), 2)
            self.assertEqual(len(messages), 2)
            self.assertEqual([item.source_file for item in messages], ["first.po", "sub/second.po"])
        finally:
            self._cleanup_paths(input_root)

    def test_load_messages_for_input_supports_xliff_file(self):
        input_path = os.path.abspath("_tmp_extract_terms_local.xliff")
        self._cleanup_paths(input_path)

        try:
            with open(input_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file original="messages.json">
    <body>
      <trans-unit id="tokenLabel" resname="tokenLabel">
        <source>Access token</source>
      </trans-unit>
    </body>
  </file>
</xliff>
"""
                )

            messages, scanned_files = extract_terms_local.load_messages_for_input(input_path)

            self.assertEqual(scanned_files, [input_path])
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].source, "Access token")
            self.assertEqual(messages[0].context, "tokenLabel")
            self.assertEqual(messages[0].source_file, os.path.basename(input_path))
        finally:
            self._cleanup_paths(input_path)


if __name__ == "__main__":
    unittest.main()
