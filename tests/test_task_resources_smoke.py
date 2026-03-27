import os
import unittest

from core.task_resources import load_task_resource_context


class TaskResourcesSmokeTests(unittest.TestCase):
    def test_load_task_resource_context_loads_vocab_rules_and_pairs(self):
        vocab_path = os.path.join(os.getcwd(), "_tmp_task_vocab.txt")
        rules_path = os.path.join(os.getcwd(), "_tmp_task_rules.md")
        try:
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("save - enregistrer\n")
                handle.write("open - ouvrir\n")
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Keep labels short.\n")

            context = load_task_resource_context(
                target_lang="fr",
                explicit_vocab_path=vocab_path,
                explicit_rules_path=rules_path,
                inline_rules="Use neutral tone.",
                load_vocab_pairs_flag=True,
            )

            self.assertEqual(context.vocabulary_source, f"file:{vocab_path}")
            self.assertEqual(
                context.vocabulary_pairs,
                [("save", "enregistrer"), ("open", "ouvrir")],
            )
            self.assertIn("Keep labels short.", context.project_rules)
            self.assertIn("Use neutral tone.", context.project_rules)
            self.assertEqual(
                context.rules_source,
                f"file:{rules_path}, inline:--rules-str",
            )
        finally:
            for path in (vocab_path, rules_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_task_resource_context_can_skip_rules(self):
        vocab_path = os.path.join(os.getcwd(), "_tmp_task_vocab_only.txt")
        try:
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("save - enregistrer\n")

            context = load_task_resource_context(
                target_lang="fr",
                explicit_vocab_path=vocab_path,
                include_rules=False,
            )

            self.assertEqual(context.vocabulary_source, f"file:{vocab_path}")
            self.assertIsNone(context.rules_path)
            self.assertIsNone(context.project_rules)
            self.assertIsNone(context.rules_source)
        finally:
            if os.path.exists(vocab_path):
                os.remove(vocab_path)


if __name__ == "__main__":
    unittest.main()
