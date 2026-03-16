import unittest

from core import review_common


class ReviewCommonSmokeTests(unittest.TestCase):
    def test_json_load_maybe_supports_fenced_json(self):
        payload = review_common.json_load_maybe("```json\n{\"a\": 1}\n```")
        self.assertEqual(payload, {"a": 1})

    def test_plural_key_sort_key_prefers_numeric_order(self):
        values = ["10", "2", "x"]
        ordered = sorted(values, key=review_common.plural_key_sort_key)
        self.assertEqual(ordered, ["2", "10", "x"])

    def test_build_target_script_guidance_uses_kazakh_specific_wording(self):
        guidance = review_common.build_target_script_guidance(
            "kk",
            update_wording=lambda: "updated target text",
        )
        self.assertIn("Kazakh Cyrillic alphabet", guidance)
        self.assertIn("updated target text", guidance)

    def test_build_target_script_guidance_generic_non_kazakh(self):
        guidance = review_common.build_target_script_guidance("fr")
        self.assertIn("real writing system", guidance)


if __name__ == "__main__":
    unittest.main()
