# Revise Task

## Purpose

`revise` applies a targeted natural-language instruction to an already translated file.

Use it when you want selective corrections or terminology changes, not a full retranslation and not a QA-only report.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML

Some formats require a paired source file because the translated file does not retain the original source text.

## Source-File Requirements

`--source-file` is:

- optional for `.po` and `.ts`
- required for Android `.xml`
- required for `.resx`
- required for `.strings`
- required for `.txt`

The source file must match the translated file type.

## Inputs And Resources

Core inputs:

- translated file
- natural-language revision instruction
- optional paired source file
- source language
- target language
- provider and model

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt`, a glossary PO, or a glossary TBX
- rules from `data/locales/<target-lang>/rules.md`

## Main CLI Shape

```powershell
python translate_cli.py revise translated.po --instruction "Use a shorter term for Preferences"
python translate_cli.py revise translated.ts --instruction "Replace archive with package where the source says package"
python translate_cli.py revise translated.ai-translated.xml --source-file source.xml --instruction "Use natural confirmation questions"
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--probe`
- `--vocab`
- `--rules`
- `--rules-str`
- `--source-file`
- `--out`
- `--in-place`
- `--dry-run`

## Output

Default revised output:

- `<input>.revised.<ext>`

Alternative behaviors:

- `--in-place` overwrites the translated input
- `--dry-run` reports revision results without writing output

## Structured Issues

Revision responses can include structured review issues with:

- `code`
- `message`
- `severity`

Current issue namespace:

- `revise.instruction_ambiguous`
- `revise.source_unclear`
- `revise.glossary_variant_choice`
- `revise.placeholder_attention`
- `revise.length_or_ui_fit_risk`
- `revise.other`

Typical severities:

- `warning`
- `info`

## Important Behavior

- unchanged translations are kept unchanged
- only items that clearly need modification are updated
- empty revised translations are not allowed
- the task respects shared localization invariants and glossary-sense rules
- changed AI-reviewed entries are marked review-required where the format supports it

## When Not To Use This Task

- use `translate` for initial translation
- use `check` for QA reporting
- use `extract-terms` or `extract-terms-local` for glossary discovery
