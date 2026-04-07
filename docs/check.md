# Check Task

## Purpose

`check` reviews translated files for QA issues.

Use it when you want structured findings about translation quality, placeholders, tags, terminology, plural handling, or script usage.

## Supported Inputs

- `.po`
- `.ts`

This task currently does not review `.resx`, `.strings`, `.txt`, or Android XML.

## Inputs And Resources

Core inputs:

- translated file
- source language
- target language
- provider and model

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt` or a glossary PO
- rules from `data/locales/<target-lang>/rules.md`

The checker combines:

- model review
- deterministic local checks for placeholders, tags, accelerators, plural slots, and approved vocabulary usage

## Main CLI Shape

```powershell
python translate_cli.py check translated.po
python translate_cli.py check translated.ts
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
- `--include-ok`
- `--out`

## Output

Default report path:

- `<input>.translation-check.json`

The report is JSON only.

## Structured Issue Model

Check findings use structured issues with:

- `code`
- `message`
- `severity`

Current issue namespace:

- `check.meaning`
- `check.grammar`
- `check.tone`
- `check.terminology`
- `check.placeholder`
- `check.tag`
- `check.accelerator`
- `check.plural`
- `check.fluency`
- `check.script`
- `check.other`

Typical severities:

- `error`
- `warning`

## Important Behavior

- only reviewable translated entries are sent for checking
- `--probe` limits how many entries are checked, mainly for testing
- `--include-ok` keeps entries with no findings in the output JSON
- vocabulary and rules are treated as real QA constraints, not just suggestions

## When Not To Use This Task

- use `translate` to generate translations
- use `revise` to apply a targeted natural-language instruction
- use `extract-terms` or `extract-terms-local` for glossary work
