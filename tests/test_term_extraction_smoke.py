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
