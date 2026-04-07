# Extract Terms Task

## Purpose

`extract-terms` uses a model to propose glossary candidates from source localization files.

Use it when you want model-driven term discovery and suggested translations, not deterministic local discovery.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML

## Inputs And Resources

Core inputs:

- source file
- source language
- target language
- provider and model

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt` or a glossary PO

In `missing` mode, vocabulary is used to suppress already known terms and keep the output focused on missing glossary entries.

## Main CLI Shape

```powershell
python translate_cli.py extract-terms source.po
python translate_cli.py extract-terms source.xml
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--vocab`
- `--mode`
- `--out-format`
- `--out`
- `--max-terms-per-batch`

## Modes

- `all`
  - broader glossary building from source content
- `missing`
  - focus on terms not already present in the supplied vocabulary

## Output

Supported output formats:

- `po`
- `json`

Default paths:

- `all` + `po` -> `<input>.glossary.po`
- `missing` + `po` -> `<input>.missing-terms.po`
- `missing` + `json` -> `<input>.missing-terms.json`

## Important Behavior

- the task now uses the shared source-message projection path from `core/term_extraction.py`
- contextual source messages are sent instead of only bare strings
- the model is instructed to prefer atomic reusable terms over phrase-shaped message fragments
- term candidates are merged case-insensitively before final output

If `missing` mode is used with `--out-format po`, known vocabulary entries are merged into the generated PO before new missing terms are appended.

## Relationship To Local Discovery

This task is the model-driven extractor.

If you want:

- deterministic local filtering
- accepted / borderline / rejected classification
- no API usage

use `extract-terms-local` instead.
