# Translate Task

## Purpose

`translate` fills unfinished localization entries, or retranslates existing ones when explicitly requested.

Use it when you want normal translation output, not QA findings, revision-only changes, or glossary extraction.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML
- a directory tree containing supported files

Multi-file translation is supported when all resolved inputs use the same format.

Android XML is a special case:

- it currently supports one translated target file at a time
- it requires `--source-file` because the translated XML does not retain the original source text

## Inputs And Resources

Core inputs:

- source file or files
- source language
- target language
- provider and model

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt`, a glossary PO, or a glossary TBX
- rules from `data/locales/<target-lang>/rules.md`

At runtime, the task builds message-scoped glossary hints:

- the full approved vocabulary is parsed into rich entries
- only the relevant subset for each message is attached as `relevant_vocabulary`

For plural messages, the task sends structured plural data:

- `source_singular`
- `source_plural`
- `plural_forms`
- `plural_slots`

## Main CLI Shape

```powershell
python translate_cli.py translate file.po
python translate_cli.py translate first.po second.po third.po
python translate_cli.py translate translated.xml --source-file source.xml
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--vocab`
- `--rules`
- `--rules-str`
- `--retranslate-all`
- `--warnings-report`

## Output

Default translated output:

- `<input>.ai-translated.<ext>`

Optional warning sidecar:

- `<input>.ai-translated.translation-warnings.json`

The warning sidecar contains only messages where the model reported a review-worthy concern.

## Structured Warning Reporting

When `--warnings-report` is enabled, the task can return per-message structured issues.

Current warning namespace:

- `translate.ambiguous_term`
- `translate.unclear_source_meaning`
- `translate.glossary_variant_choice`
- `translate.possible_untranslated_token`
- `translate.placeholder_attention`
- `translate.length_or_ui_fit_risk`

Each warning item uses:

- `code`
- `message`
- `severity`

Translation warnings use:

- `warning` for real ambiguity or risk
- `info` for notable but non-risk notes

## Important Behavior

- by default, only unfinished or fuzzy entries are translated
- `--retranslate-all` forces all translatable entries back through the model
- recursive directory translation skips generated toolkit artifacts such as `*.ai-translated.*`, `*.glossary.po`, `*.missing-terms.po`, and `*.prototype-*.po`
- when the scan root is this toolkit repository itself, recursive translation also skips toolkit-owned directories such as `data/`, `logs/`, `docs/`, `tests/`, `tasks/`, and `core/`
- placeholders and protected tokens are preserved
- message `context` and `note` are sent to the model when available
- `relevant_vocabulary` can include multiple variants for the same source term, and the model is instructed to choose by context and part of speech

## When Not To Use This Task

Use another task when the goal is different:

- `check` for QA findings
- `revise` for targeted edits to an already translated file
- `extract-terms` for model glossary discovery
- `extract-terms-local` for deterministic local glossary discovery without an API call
