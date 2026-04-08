# Local Extract Task

## Purpose

`extract-terms-local` runs deterministic local term discovery without a model API call.

Use it when you want a cheap, repeatable source-side glossary pass before any model validation or translation.

## Supported Inputs

Source discovery input:

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML
- a directory tree containing supported source files

JSON conversion input:

- local extraction JSON produced by this task

## Inputs And Resources

Core inputs:

- source file, source directory tree, or local extraction JSON
- source language
- target language

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt`, a glossary `.tbx`, or a glossary directory bundle

Extraction behavior is driven by data files under:

- `data/extract/common/...`
- `data/extract/en/...`

## Main CLI Shape

Source discovery:

```powershell
python translate_cli.py extract-terms-local source.po
python translate_cli.py extract-terms-local C:\path\to\source-tree
```

JSON to PO handoff:

```powershell
python translate_cli.py extract-terms-local source.prototype-missing-terms.json --to-po
```

Useful options:

- `--mode`
- `--max-length`
- `--vocab`
- `--include-rejected`
- `--also-po`
- `--include-borderline`
- `--out`
- `--to-po`

## Modes

- `all`
  - keep all local candidates
- `missing`
  - drop candidates already covered by the approved vocabulary

## Output

Normal extraction writes JSON:

- `<input>.prototype-glossary.json`
- or `<input>.prototype-missing-terms.json`

Optional one-shot PO handoff:

- `--also-po` writes the matching PO handoff beside the JSON report

JSON to PO conversion:

- `--to-po` converts an existing local extraction JSON into a translation-ready PO glossary file

## What The JSON Contains

The JSON report includes:

- accepted terms
- borderline terms
- rejected terms, when requested
- translation candidates
- extraction metadata such as mode, max length, vocabulary source, and counts

## Important Behavior

- repeated source messages are deduplicated across files
- extraction uses shared logic from `core/term_extraction.py`
- JSON shaping and PO conversion use `core/term_handoff.py`
- `max-length 1` means unigrams only
- `max-length 2` enables bigrams
- `max-length 3` enables bi- and tri-grams

## Relationship To Model Extraction

This task is intentionally separate from `extract-terms`.

Use `extract-terms-local` for:

- deterministic local candidate discovery
- vocabulary-aware filtering in `missing` mode
- JSON and PO handoff preparation

Use `extract-terms` for:

- model-driven term discovery
- model-suggested target-language glossary entries
