import os
import unittest

import polib

from core.task_resources import load_task_resource_context


class TaskResourcesSmokeTests(unittest.TestCase):
    def test_load_task_resource_context_loads_glossary_rules_and_pairs(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_task_glossary.po")
        rules_path = os.path.join(os.getcwd(), "_tmp_task_rules.md")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="save", msgstr="enregistrer", msgctxt="verb"))
            po.append(polib.POEntry(msgid="open", msgstr="ouvrir", msgctxt="verb"))
            po.save(glossary_path)
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Keep labels short.\n")

            context = load_task_resource_context(
                target_lang="fr",
                explicit_glossary_path=glossary_path,
                explicit_rules_path=rules_path,
                inline_rules="Use neutral tone.",
                load_glossary_pairs_flag=True,
            )

            self.assertEqual(context.glossary_source, f"file:{glossary_path}")
            self.assertEqual(
                context.glossary_pairs,
                [("save", "enregistrer"), ("open", "ouvrir")],
            )
            self.assertIn("Keep labels short.", context.project_rules)
            self.assertIn("Use neutral tone.", context.project_rules)
            self.assertEqual(
                context.rules_source,
                f"file:{rules_path}, inline:--rules-str",
            )
        finally:
            for path in (glossary_path, rules_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_load_task_resource_context_can_skip_rules(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_task_glossary_only.po")
        try:
            po = polib.POFile()
            po.append(polib.POEntry(msgid="save", msgstr="enregistrer", msgctxt="verb"))
            po.save(glossary_path)

            context = load_task_resource_context(
                target_lang="fr",
                explicit_glossary_path=glossary_path,
                include_rules=False,
            )

            self.assertEqual(context.glossary_source, f"file:{glossary_path}")
            self.assertIsNone(context.rules_path)
            self.assertIsNone(context.project_rules)
            self.assertIsNone(context.rules_source)
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)

    def test_load_task_resource_context_supports_glossary_directory(self):
        glossary_dir = os.path.join(os.getcwd(), "_tmp_task_glossary_dir")
        common_path = os.path.join(glossary_dir, "10-common.po")
        colors_path = os.path.join(glossary_dir, "20-colors.po")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            common_po = polib.POFile()
            common_po.append(polib.POEntry(msgid="save", msgstr="enregistrer", msgctxt="verb"))
            common_po.save(common_path)
            colors_po = polib.POFile()
            colors_po.append(polib.POEntry(msgid="blue", msgstr="bleu", msgctxt="adjective"))
            colors_po.save(colors_path)

            context = load_task_resource_context(
                target_lang="fr",
                explicit_glossary_path=glossary_dir,
                include_rules=False,
                load_glossary_pairs_flag=True,
            )

            self.assertEqual(context.glossary_source, f"dir:{glossary_dir}")
            self.assertEqual(
                context.glossary_pairs,
                [("save", "enregistrer"), ("blue", "bleu")],
            )
        finally:
            for path in (common_path, colors_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)

    def test_load_task_resource_context_selects_target_language_from_tbx(self):
        glossary_path = os.path.join(os.getcwd(), "_tmp_task_glossary.tbx")
        try:
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<martif type="TBX" xml:lang="en">
  <text>
    <body>
      <termEntry>
        <langSet xml:lang="en"><tig><term>save</term></tig></langSet>
        <langSet xml:lang="fr"><tig><term>enregistrer</term></tig></langSet>
        <langSet xml:lang="kk"><tig><term>сақтау</term></tig></langSet>
      </termEntry>
    </body>
  </text>
</martif>
"""
                )

            context = load_task_resource_context(
                target_lang="kk",
                explicit_glossary_path=glossary_path,
                include_rules=False,
                load_glossary_pairs_flag=True,
            )

            self.assertEqual(context.glossary_source, f"file:{glossary_path}")
            self.assertEqual(context.glossary_pairs, [("save", "сақтау")])
            self.assertEqual(context.glossary_text, "save|сақтау||")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)


if __name__ == "__main__":
    unittest.main()
