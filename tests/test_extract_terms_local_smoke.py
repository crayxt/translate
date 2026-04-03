import argparse
import json
import os
import unittest

import polib

from tasks import extract_terms_local


class ExtractTermsLocalSmokeTests(unittest.TestCase):
    def _cleanup_paths(self, *paths: str) -> None:
        for path in paths:
            try:
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


if __name__ == "__main__":
    unittest.main()
