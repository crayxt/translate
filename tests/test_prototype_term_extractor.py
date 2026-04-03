import unittest

import prototype_term_extractor as prototype


class PrototypeTermExtractorTests(unittest.TestCase):
    def test_build_output_path_for_missing_mode(self):
        out = prototype.build_output_path(r"C:\work\input.po", "missing")
        self.assertEqual(out, r"C:\work\input.prototype-missing-terms.json")

    def test_single_occurrence_loose_phrase_is_rejected_but_atomic_terms_remain(self):
        result = prototype.extract_terms_locally(
            [prototype.SourceMessage(source="Choose audio channel")],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term for item in result.rejected_terms}

        self.assertIn("audio", accepted)
        self.assertIn("channel", accepted)
        self.assertIn("audio channel", rejected)

    def test_allowlisted_fixed_compound_is_kept(self):
        result = prototype.extract_terms_locally(
            [prototype.SourceMessage(source="Access token expired")],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        self.assertIn("access token", accepted)

    def test_missing_mode_filters_known_vocabulary_terms(self):
        result = prototype.extract_terms_locally(
            [prototype.SourceMessage(source="Audio channel")],
            mode="missing",
            vocabulary_pairs=[("channel", "арна")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("channel", accepted)
        self.assertIn("channel", rejected)
        self.assertIn("already_in_vocabulary", rejected["channel"].reasons)

    def test_context_and_note_are_attached_to_candidate_evidence(self):
        result = prototype.extract_terms_locally(
            [
                prototype.SourceMessage(
                    source="Access token",
                    context="Security settings",
                    note="API authentication label",
                )
            ],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        self.assertIn("access token", accepted)
        self.assertEqual(accepted["access token"].contexts, ["Security settings"])
        self.assertEqual(accepted["access token"].notes, ["API authentication label"])
        self.assertEqual(accepted["access token"].examples, ["Access token"])


if __name__ == "__main__":
    unittest.main()
