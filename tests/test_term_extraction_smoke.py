import unittest

from core import term_extraction


class TermExtractionSmokeTests(unittest.TestCase):
    def test_tokenize_source_text_strips_accelerator_markers(self):
        self.assertEqual(
            term_extraction.tokenize_source_text("Wall&paper"),
            ["wallpaper"],
        )

    def test_tokenize_source_text_strips_underscore_accelerators(self):
        self.assertEqual(
            term_extraction.tokenize_source_text("Default fra_me delay"),
            ["default", "frame", "delay"],
        )

    def test_tokenize_source_text_preserves_leading_percent_and_dash_markers(self):
        self.assertEqual(
            term_extraction.tokenize_source_text("%PRODUCTNAME --help -output 3D"),
            ["%productname", "-help", "-output", "3d"],
        )

    def test_tokenize_source_text_strips_xml_markup_and_url_words(self):
        self.assertEqual(
            term_extraction.tokenize_source_text(
                'Open <image src="https://example.com/icon.png" alt="Preview"> gallery'
            ),
            ["open", "-urlnoise", "gallery"],
        )

    def test_extract_message_candidate_counts_does_not_bridge_across_skipped_marker_tokens(self):
        counts = term_extraction.extract_message_candidate_counts("Version 3D model", max_length=2)

        self.assertIn("version", counts)
        self.assertIn("model", counts)
        self.assertNotIn("version model", counts)

    def test_extract_message_candidate_counts_does_not_bridge_across_url_noise(self):
        counts = term_extraction.extract_message_candidate_counts(
            "Open https://example.com dialog",
            max_length=2,
        )

        self.assertIn("open", counts)
        self.assertIn("dialog", counts)
        self.assertNotIn("open dialog", counts)

    def test_build_relevant_vocabulary_matches_rich_entries(self):
        entries = term_extraction.build_scoped_vocabulary_entries(
            "start|бастау|verb|Start playback\nplayback|ойнату|noun|Media playback"
        )

        relevant = term_extraction.build_relevant_vocabulary("Start playback", entries)

        self.assertEqual(
            relevant,
            [
                {
                    "source_term": "playback",
                    "target_term": "ойнату",
                    "part_of_speech": "noun",
                    "context_note": "Media playback",
                },
                {
                    "source_term": "start",
                    "target_term": "бастау",
                    "part_of_speech": "verb",
                    "context_note": "Start playback",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
