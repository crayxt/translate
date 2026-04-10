import unittest
from pathlib import Path


class RuleFilesSmokeTests(unittest.TestCase):
    def test_kazakh_rules_file_is_utf8_and_not_mojibake(self):
        text = Path("data/locales/kk/rules.md").read_text(encoding="utf-8")

        self.assertIn("Kazakh UI Translation Rules", text)
        self.assertIn("real Kazakh Cyrillic", text)
        self.assertIn("ма/ме/ба/бе/па/пе", text)
        self.assertIn("ә, і, ң, ғ, ү, ұ, қ, ө, һ", text)

        for broken_fragment in ("Ð", "Ñ", "Ò", "Â", "â€"):
            self.assertNotIn(broken_fragment, text)


if __name__ == "__main__":
    unittest.main()
