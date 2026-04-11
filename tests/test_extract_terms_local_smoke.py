import argparse
import json
import os
import shutil
import unittest

import polib

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
        vocab_path = os.path.abspath("_tmp_extract_terms_local_vocab.txt")
        json_path = os.path.abspath("_tmp_extract_terms_local.prototype-missing-terms.json")
        po_path = os.path.abspath("_tmp_extract_terms_local.prototype-missing-terms.po")
        self._cleanup_paths(input_path, vocab_path, json_path, po_path)

        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("")

            args = argparse.Namespace(
                file=input_path,
                source_lang="en",
                target_lang="kk",
                vocab=vocab_path,
                to_po=False,
                also_po=True,
                mode="missing",
                max_length=2,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

            extract_terms_local.run_from_args(args)

            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(po_path))

            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["output_file"], json_path)
            self.assertGreaterEqual(payload["translation_candidate_count"], 1)

            po = polib.pofile(po_path, wrapwidth=78)
            self.assertGreaterEqual(len(po), 1)
            self.assertEqual(po[0].msgid, "access token")
        finally:
            self._cleanup_paths(input_path, vocab_path, json_path, po_path)

    def test_run_from_args_can_extract_from_directory_tree(self):
        input_root = os.path.abspath("_tmp_extract_terms_tree")
        nested_root = os.path.join(input_root, "sub")
        po_one = os.path.join(input_root, "first.po")
        po_two = os.path.join(nested_root, "second.po")
        vocab_path = os.path.abspath("_tmp_extract_terms_tree_vocab.txt")
        json_path = os.path.abspath("_tmp_extract_terms_tree.json")
        po_path = os.path.abspath("_tmp_extract_terms_tree.po")
        self._cleanup_paths(input_root, vocab_path, json_path, po_path)

        try:
            os.makedirs(nested_root, exist_ok=True)
            with open(po_one, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(po_two, "w", encoding="utf-8") as handle:
                handle.write('msgid "Playback speed"\nmsgstr ""\n')
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("")

            args = argparse.Namespace(
                file=input_root,
                source_lang="en",
                target_lang="kk",
                vocab=vocab_path,
                to_po=False,
                also_po=True,
                mode="missing",
                max_length=2,
                out=json_path,
                include_rejected=False,
                include_borderline=False,
            )

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
            self._cleanup_paths(input_root, vocab_path, json_path, po_path)

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
