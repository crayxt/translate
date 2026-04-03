import json
import os
import unittest

import polib

from core import term_handoff as handoff


class PrototypeTermsToPoSmokeTests(unittest.TestCase):
    def _cleanup_paths(self, *paths: str) -> None:
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def test_build_output_path_replaces_json_extension(self):
        out = handoff.build_po_output_path(r"C:\work\sample.prototype-missing-terms.json")
        self.assertEqual(out, r"C:\work\sample.prototype-missing-terms.po")

    def test_convert_json_to_po_writes_context_notes_and_occurrences(self):
        payload = {
            "source_lang": "en",
            "target_lang": "kk",
            "translation_candidates": [
                {
                    "source_term": "access token",
                    "decision": "accepted",
                    "score": 6,
                    "reasons": ["fixed_multiword_allowlist", "has_meaningful_context"],
                    "contexts": ["Security settings"],
                    "examples": ["Access token"],
                    "notes": ["API authentication label"],
                    "location_files": ["src/auth/api.c:10"],
                    "location_scopes": ["src/auth"],
                    "known_translation": "",
                }
            ],
            "borderline_terms": [
                {
                    "source_term": "audio channel",
                    "decision": "borderline",
                    "score": 2,
                    "reasons": ["needs_review"],
                }
            ],
        }

        json_path = os.path.abspath("_tmp_prototype_terms.sample.prototype-missing-terms.json")
        po_path = os.path.abspath("_tmp_prototype_terms.sample.prototype-missing-terms.po")
        self._cleanup_paths(json_path, po_path)
        try:
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            out_path = handoff.convert_json_to_po(json_path, out_path=po_path)

            self.assertEqual(out_path, po_path)
            po = polib.pofile(po_path, wrapwidth=78)
            self.assertEqual(len(po), 1)
            self.assertEqual(po.metadata["Language"], "kk")
            self.assertEqual(po.metadata["X-Source-Language"], "en")
            self.assertEqual(po.metadata["X-Prototype-Source"], json_path)
            self.assertEqual(po[0].msgid, "access token")
            self.assertEqual(po[0].msgstr, "")
            self.assertEqual(po[0].msgctxt, "Security settings")
            self.assertEqual(po[0].occurrences, [("src/auth/api.c", "10")])
            self.assertIn("Decision: accepted", po[0].tcomment)
            self.assertIn("Score: 6", po[0].tcomment)
            self.assertIn("Reasons: fixed_multiword_allowlist, has_meaningful_context", po[0].tcomment)
            self.assertIn("Example: Access token", po[0].tcomment)
            self.assertIn("Note: API authentication label", po[0].tcomment)
            self.assertIn("Location scopes: src/auth", po[0].tcomment)
        finally:
            self._cleanup_paths(json_path, po_path)

    def test_convert_json_to_po_can_include_borderline_terms(self):
        payload = {
            "source_lang": "en",
            "target_lang": "kk",
            "translation_candidates": [
                {"source_term": "access token", "decision": "accepted"}
            ],
            "borderline_terms": [
                {
                    "source_term": "audio channel",
                    "decision": "borderline",
                    "score": 2,
                    "reasons": ["needs_review"],
                }
            ],
        }

        json_path = os.path.abspath("_tmp_prototype_terms.include-borderline.prototype-missing-terms.json")
        po_path = os.path.abspath("_tmp_prototype_terms.include-borderline.prototype-missing-terms.po")
        self._cleanup_paths(json_path, po_path)
        try:
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            handoff.convert_json_to_po(
                json_path,
                out_path=po_path,
                include_borderline=True,
            )

            po = polib.pofile(po_path, wrapwidth=78)
            self.assertEqual([entry.msgid for entry in po], ["access token", "audio channel"])
            self.assertIn("Decision: borderline", po[1].tcomment)
        finally:
            self._cleanup_paths(json_path, po_path)


if __name__ == "__main__":
    unittest.main()
