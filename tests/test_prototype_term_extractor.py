import unittest

from core import term_extraction as extraction
from core import term_handoff as handoff


class PrototypeTermExtractorTests(unittest.TestCase):
    def test_build_output_path_for_missing_mode(self):
        out = handoff.build_output_path(r"C:\work\input.po", "missing")
        self.assertEqual(out, r"C:\work\input.prototype-missing-terms.json")

    def test_build_source_messages_from_payloads_matches_main_message_shape(self):
        messages = extraction.build_source_messages_from_payloads(
            [
                {"source": "Open file", "context": "Toolbar", "note": "Primary action"},
                {"source": "Open file", "context": "Toolbar", "note": "Primary action"},
            ]
        )

        self.assertEqual(
            messages,
            [
                extraction.SourceMessage(
                    source="Open file",
                    context="Toolbar",
                    note="Primary action",
                )
            ],
        )

    def test_ampersand_accelerator_is_removed_before_tokenization(self):
        self.assertEqual(extraction.tokenize_source_text("Wall&paper"), ["wallpaper"])

    def test_wallpaper_accelerator_form_does_not_extract_paper(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Wall&paper", context="Appearance")],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term for item in result.rejected_terms}

        self.assertIn("wallpaper", accepted)
        self.assertNotIn("paper", accepted)
        self.assertNotIn("paper", rejected)

    def test_underscore_accelerator_form_does_not_extract_false_subterm(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Default fra_me delay", context="Rendering")],
            mode="all",
            vocabulary_pairs=[],
        )

        all_terms = {
            item.source_term
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertNotIn("me", all_terms)

    def test_percent_prefixed_placeholder_is_not_extracted_as_term(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="%PRODUCTNAME", context="Branding")],
            mode="all",
            vocabulary_pairs=[],
        )

        all_terms = {
            item.source_term
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertEqual(all_terms, set())

    def test_dash_prefixed_option_is_not_extracted_as_term(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Use --verbose mode", context="CLI help")],
            mode="all",
            vocabulary_pairs=[],
            max_length=2,
        )

        all_terms = {
            item.source_term
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertNotIn("-verbose", all_terms)
        self.assertNotIn("use mode", all_terms)

    def test_digit_led_label_is_not_extracted_as_term(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="3D", context="Graphics")],
            mode="all",
            vocabulary_pairs=[],
        )

        all_terms = {
            item.source_term
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertEqual(all_terms, set())

    def test_xml_attribute_and_url_words_are_not_extracted(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(
                    source='Open <image src="https://example.com/icon.png" alt="Preview"> Gallery',
                    context="Help",
                )
            ],
            mode="all",
            vocabulary_pairs=[],
            max_length=2,
        )

        all_terms = {
            item.source_term
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertIn("gallery", all_terms)
        self.assertNotIn("src", all_terms)
        self.assertNotIn("image", all_terms)
        self.assertNotIn("example", all_terms)
        self.assertNotIn("com", all_terms)
        self.assertNotIn("preview", all_terms)
        self.assertNotIn("open gallery", all_terms)

    def test_single_occurrence_loose_phrase_is_rejected_and_atomic_terms_go_borderline(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Choose audio channel", context="Audio settings")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        borderline = {item.source_term: item for item in result.borderline_terms}
        rejected = {item.source_term for item in result.rejected_terms}

        self.assertNotIn("audio", accepted)
        self.assertNotIn("channel", accepted)
        self.assertIn("audio", borderline)
        self.assertIn("channel", borderline)
        self.assertEqual(borderline["audio"].contexts, ["Audio settings"])
        self.assertEqual(borderline["audio"].decision, "borderline")
        self.assertIn("audio channel", rejected)

    def test_low_signal_one_off_words_are_rejected(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Please wait later")],
            mode="all",
            vocabulary_pairs=[],
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_expanded_low_value_words_are_filtered(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Any much next", context="General")],
            mode="all",
            vocabulary_pairs=[],
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_instructional_phrase_is_filtered(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="You can use this later", context="Help")],
            mode="all",
            vocabulary_pairs=[],
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_bigram_mode_keeps_fixed_compound(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Access token expired")],
            mode="all",
            vocabulary_pairs=[],
            max_length=2,
        )

        accepted = {item.source_term for item in result.accepted_terms}
        self.assertIn("access token", accepted)

    def test_missing_mode_filters_known_vocabulary_terms(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Audio channel")],
            mode="missing",
            vocabulary_pairs=[("channel", "arna")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("channel", accepted)
        self.assertIn("channel", rejected)
        self.assertIn("already_in_vocabulary", rejected["channel"].reasons)

    def test_missing_mode_uses_normalized_vocabulary_keys(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Access token")],
            mode="missing",
            vocabulary_pairs=[("  ACCESS   TOKEN  ", "token")],
            max_length=2,
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("access token", accepted)
        self.assertIn("access token", rejected)
        self.assertIn("already_in_vocabulary", rejected["access token"].reasons)

    def test_missing_mode_excludes_singular_candidate_when_vocab_has_plural(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="File")],
            mode="missing",
            vocabulary_pairs=[("Files", "faildar")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("file", accepted)
        self.assertIn("file", rejected)
        self.assertIn("already_in_vocabulary", rejected["file"].reasons)

    def test_context_note_and_locations_are_attached_to_candidate_evidence(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(
                    source="Access token",
                    context="Security settings",
                    note="API authentication label locations: src/auth/api.c:10 src/gui/token.c:42",
                    source_file="ui/security.po",
                )
            ],
            mode="all",
            vocabulary_pairs=[],
            max_length=2,
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        self.assertIn("access token", accepted)
        self.assertEqual(accepted["access token"].contexts, ["Security settings"])
        self.assertEqual(
            accepted["access token"].notes,
            ["API authentication label locations: src/auth/api.c:10 src/gui/token.c:42"],
        )
        self.assertEqual(accepted["access token"].examples, ["Access token"])
        self.assertEqual(accepted["access token"].context_diversity, 1)
        self.assertEqual(accepted["access token"].file_count, 1)
        self.assertEqual(accepted["access token"].files, ["ui/security.po"])
        self.assertEqual(accepted["access token"].location_file_count, 2)
        self.assertEqual(accepted["access token"].location_scope_count, 2)
        self.assertEqual(accepted["access token"].location_scopes, ["src/auth", "src/gui"])

    def test_unicode_single_word_label_can_be_extracted(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="\u0422\u043e\u043a\u0435\u043d", context="Security settings")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        accepted = {item.source_term for item in result.accepted_terms}
        self.assertIn("\u0442\u043e\u043a\u0435\u043d", accepted)

    def test_any_is_filtered_before_candidate_classification(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Any", context="General")],
            mode="all",
            vocabulary_pairs=[],
        )

        self.assertNotIn("any", {item.source_term for item in result.accepted_terms})
        self.assertNotIn("any", {item.source_term for item in result.borderline_terms})
        self.assertNotIn("any", {item.source_term for item in result.rejected_terms})

    def test_known_brand_and_protocol_terms_are_filtered(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="GNOME Bluetooth Firefox HTTP HTTPS QGIS VLC", context="About")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_common_file_format_abbreviations_are_filtered(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="MP3 AVI DOCX PDF PNG JSON XML ZIP", context="Formats")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_known_multiword_brand_term_is_filtered(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="The Document Foundation")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        self.assertEqual({item.source_term for item in result.accepted_terms}, set())
        self.assertEqual({item.source_term for item in result.borderline_terms}, set())
        self.assertEqual({item.source_term for item in result.rejected_terms}, set())

    def test_cross_module_usage_promotes_term_to_accepted(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(source="Use demuxer", note="locations: modules/demux/main.c:10"),
                extraction.SourceMessage(source="Select demuxer", note="locations: src/input/access.c:20"),
            ],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        self.assertIn("demuxer", accepted)
        self.assertIn("cross_module_usage", accepted["demuxer"].reasons)
        self.assertEqual(accepted["demuxer"].decision, "accepted")

    def test_compositional_phrase_is_capped_at_borderline(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(source="Video", context="Playback"),
                extraction.SourceMessage(source="Stream", context="Playback"),
                extraction.SourceMessage(
                    source="Video stream",
                    context="Playback",
                    note="locations: modules/video/output.c:10",
                ),
                extraction.SourceMessage(
                    source="Video stream",
                    context="Playback",
                    note="locations: src/stream/main.c:20",
                ),
            ],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        borderline = {item.source_term: item for item in result.borderline_terms}

        self.assertIn("video", accepted)
        self.assertIn("stream", accepted)
        self.assertIn("video stream", borderline)
        self.assertIn("compositional_phrase", borderline["video stream"].reasons)
        self.assertEqual(borderline["video stream"].decision, "borderline")

    def test_build_json_payload_includes_borderline_terms(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Choose audio channel", context="Audio settings")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )
        messages = [extraction.SourceMessage(source="Choose audio channel", context="Audio settings")]

        payload = handoff.build_json_payload(
            file_path="sample.po",
            out_path="sample.prototype.json",
            source_lang="en",
            target_lang="kk",
            mode="all",
            max_length=3,
            vocabulary_path=None,
            total_source_messages=len(messages),
            result=result,
            include_rejected=True,
        )

        self.assertEqual(payload["prototype"], "local_term_extractor_v2")
        self.assertEqual(payload["max_length"], 3)
        self.assertEqual(payload["borderline_candidate_count"], 2)
        self.assertEqual(
            {item["source_term"] for item in payload["borderline_terms"]},
            {"audio", "channel"},
        )
        self.assertEqual(payload["translation_candidate_count"], 0)
        self.assertEqual(payload["translation_candidates"], [])

    def test_build_translation_candidate_payload_exports_accepted_terms(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(
                    source="Access token",
                    context="Security settings",
                    note="API authentication label locations: src/auth/api.c:10",
                    source_file="ui/security.po",
                )
            ],
            mode="all",
            vocabulary_pairs=[],
            max_length=2,
        )

        payload = handoff.build_translation_candidate_payload(result)

        exported = {item["source_term"]: item for item in payload}
        self.assertIn("access token", exported)
        self.assertEqual(exported["access token"]["decision"], "accepted")
        self.assertEqual(exported["access token"]["contexts"], ["Security settings"])
        self.assertEqual(exported["access token"]["examples"], ["Access token"])
        self.assertEqual(exported["access token"]["file_count"], 1)
        self.assertEqual(exported["access token"]["files"], ["ui/security.po"])
        self.assertEqual(exported["access token"]["location_files"], ["src/auth/api.c:10"])
        self.assertEqual(exported["access token"]["location_scopes"], ["src/auth"])

    def test_build_translation_candidate_payload_can_include_borderline_terms(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Choose audio channel", context="Audio settings")],
            mode="all",
            vocabulary_pairs=[],
            max_length=3,
        )

        payload = handoff.build_translation_candidate_payload(result, include_borderline=True)

        self.assertEqual({item["source_term"] for item in payload}, {"audio", "channel"})

    def test_default_max_length_is_unigrams_only(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Access token expired", context="Security")],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        borderline = {item.source_term for item in result.borderline_terms}
        rejected = {item.source_term for item in result.rejected_terms}

        self.assertNotIn("access token", accepted)
        self.assertNotIn("access token", borderline)
        self.assertNotIn("access token", rejected)

    def test_bigram_max_length_includes_bigrams(self):
        counts = extraction.extract_message_candidate_counts(
            "Access token and audio channel",
            max_length=2,
        )

        self.assertIn("access token", counts)
        self.assertIn("audio channel", counts)

    def test_all_caps_term_is_kept_distinct_from_normal_case_variants(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(source="SUM", context="Calc functions"),
                extraction.SourceMessage(source="sum", context="Math notes"),
                extraction.SourceMessage(source="Sum", context="Math headings"),
            ],
            mode="all",
            vocabulary_pairs=[],
        )

        all_terms = {
            item.source_term: item
            for bucket in (result.accepted_terms, result.borderline_terms, result.rejected_terms)
            for item in bucket
        }

        self.assertIn("SUM", all_terms)
        self.assertIn("sum", all_terms)
        self.assertNotIn("Sum", all_terms)
        self.assertEqual(all_terms["SUM"].surface_forms, ["SUM"])

    def test_missing_mode_treats_all_caps_vocabulary_as_distinct_from_normal_case(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="SUM", context="Calc functions")],
            mode="missing",
            vocabulary_pairs=[("sum", "somasy")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        borderline = {item.source_term for item in result.borderline_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertIn("SUM", accepted | borderline)
        self.assertNotIn("SUM", rejected)

    def test_missing_mode_excludes_all_caps_term_when_vocab_matches_all_caps(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="SUM", context="Calc functions")],
            mode="missing",
            vocabulary_pairs=[("SUM", "SUM")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("SUM", accepted)
        self.assertIn("SUM", rejected)
        self.assertIn("already_in_vocabulary", rejected["SUM"].reasons)

    def test_plural_variant_is_canonicalized_when_base_form_exists(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(source="File"),
                extraction.SourceMessage(source="Files"),
            ],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        self.assertIn("file", accepted)
        self.assertNotIn("files", accepted)
        self.assertEqual(accepted["file"].surface_forms, ["file", "files"])

    def test_possessive_variant_is_canonicalized_when_base_form_exists(self):
        result = extraction.extract_terms_locally(
            [
                extraction.SourceMessage(source="Piece"),
                extraction.SourceMessage(source="Piece's"),
            ],
            mode="all",
            vocabulary_pairs=[],
        )

        accepted = {item.source_term: item for item in result.accepted_terms}
        self.assertIn("piece", accepted)
        self.assertNotIn("piece's", accepted)
        self.assertEqual(accepted["piece"].surface_forms, ["piece", "piece's"])

    def test_missing_mode_excludes_possessive_candidate_when_vocab_has_base_form(self):
        result = extraction.extract_terms_locally(
            [extraction.SourceMessage(source="Piece's")],
            mode="missing",
            vocabulary_pairs=[("Piece", "bolshek")],
        )

        accepted = {item.source_term for item in result.accepted_terms}
        rejected = {item.source_term: item for item in result.rejected_terms}

        self.assertNotIn("piece", accepted)
        self.assertIn("piece", rejected)
        self.assertIn("already_in_vocabulary", rejected["piece"].reasons)


if __name__ == "__main__":
    unittest.main()
