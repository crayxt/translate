# Translation Toolkit

Translate and maintain software localization files with one CLI and one GUI.

This repository is for software-localization work, not generic document translation. It gives you a shared backend for five jobs:

- translate unfinished localization files
- revise existing translations with a precise instruction
- check translated PO files for QA issues
- extract glossary terms with a model
- discover glossary candidates locally without any API call

Supported formats:

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML

Supported providers:

- Gemini
- OpenAI
- Anthropic

## What This Project Is

The toolkit is built around three ideas:

1. One task-oriented CLI: `translate`, `revise`, `check`, `extract-terms`, `extract-terms-local`
2. Shared language resources: vocabulary and rules are loaded by target language
3. One format backend: supported file types are normalized into a shared entry model and then written back in their native format

That matters because the same vocabulary, rules, runtime controls, batching, and format handling are reused across CLI and GUI instead of being reimplemented per script.

The preferred entry point is:

```text
python translate_cli.py
```

Legacy wrapper scripts still exist for compatibility, but the unified CLI is the main surface.

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set the API key for the provider you want to use:

```powershell
$env:GOOGLE_API_KEY = "your_google_api_key"
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:ANTHROPIC_API_KEY = "your_anthropic_api_key"
```

Notes:

- You only need the key for the provider you are actually using.
- Gemini can run against AI Studio or Vertex API-key mode.
- Vertex API-key mode currently supports the `global` endpoint only.

## The Main Workflows

### 1. Translate a localization file

Translate one file:

```powershell
python translate_cli.py translate source.po
python translate_cli.py translate source.ts
python translate_cli.py translate source.resx
python translate_cli.py translate source.strings
python translate_cli.py translate source.txt
```

Translate several files in one run when they are the same format:

```powershell
python translate_cli.py translate first.po second.po third.po
```

Choose provider and model explicitly:

```powershell
python translate_cli.py translate source.po --provider openai --model your-model
python translate_cli.py translate source.po --provider anthropic --model your-model
python translate_cli.py translate source.po --provider gemini --model your-model
```

Useful controls:

```powershell
python translate_cli.py translate source.po --target-lang fr
python translate_cli.py translate source.po --thinking-level medium
python translate_cli.py translate source.po --batch-size 100 --parallel-requests 4
python translate_cli.py translate source.po --retranslate-all
python translate_cli.py translate source.po --flex
```

Behavior:

- default target language is `kk`
- by default, only unfinished messages are translated
- `--retranslate-all` forces already translated messages through translation again
- translated output is written as `*.ai-translated.<ext>`

### 2. Translate Android XML with a paired source file

Android translated exports often contain only resource IDs on the target side, so translation uses a paired-source workflow:

```powershell
python translate_cli.py translate translated.xml --source-file source.xml
```

The Android XML backend:

- supports `<string>` and `<plurals>`
- pairs `<string>` items by resource name
- pairs `<plurals>` by resource name plus quantity
- preserves inline XML such as `<xliff:g>`
- preserves literal escapes such as `\n` in source style

Current translation constraint:

- Android `.xml` translation currently supports one target file at a time and requires `--source-file`

### 3. Revise an existing translation

Use `revise` when a file is already translated and you want targeted changes rather than full retranslation.

For formats that still contain source and translation together:

```powershell
python translate_cli.py revise translated.po --instruction "Use a shorter term for Preferences"
python translate_cli.py revise translated.ts --instruction "Replace archive with package where the source says package"
```

For formats where the translated file no longer carries the original source text, pass the matching source file:

```powershell
python translate_cli.py revise translated.ai-translated.xml --source-file source.xml --instruction "Use natural confirmation questions and preserve literal \\n escapes"
python translate_cli.py revise translated.ai-translated.strings --source-file source.strings --instruction "Shorten viewer labels where possible"
python translate_cli.py revise translated.ai-translated.resx --source-file source.resx --instruction "Use command bar instead of toolbar"
python translate_cli.py revise translated.txt --source-file source.txt --instruction "Use formal tone for Exit"
```

Revision behavior:

- default output path is `<input>.revised.<ext>`
- `--in-place` overwrites the translated input file
- `--dry-run` reviews and reports changes without writing output
- changed AI-reviewed entries are marked as review-required where the format supports it

### 4. Check a translated PO file

Use `check` for QA on an already translated PO file:

```powershell
python translate_cli.py check translated.po
python translate_cli.py check translated.po --probe 50
python translate_cli.py check translated.po --out report.json --include-ok
```

The checker combines model findings with deterministic local checks for:

- placeholders
- tags
- accelerators
- plural slots
- approved vocabulary usage

Default output path:

```text
translated.translation-check.json
```

### 5. Extract glossary terms with a model

Use `extract-terms` when you want the model to propose glossary entries:

```powershell
python translate_cli.py extract-terms source.po
python translate_cli.py extract-terms source.xml
```

Useful variants:

```powershell
python translate_cli.py extract-terms source.po --mode missing --vocab data/locales/kk/vocab --out-format po
python translate_cli.py extract-terms source.po --mode missing --out-format json --vocab data/locales/kk/vocab
python translate_cli.py extract-terms source.po --out glossary.po --batch-size 200 --parallel-requests 4
```

Modes:

- `all`: build a broader glossary from the source content
- `missing`: focus on terms that are not already in your existing vocabulary

Output defaults:

- `all` + `po` -> `<input>.glossary.po`
- `missing` + `po` -> `<input>.missing-terms.po`
- `missing` + `json` -> `<input>.missing-terms.json`

When you run missing-term extraction with `--out-format po`, the generated PO is designed to go straight back into review and then translation:

- known terms from the supplied vocabulary are imported
- new missing terms are added as reviewable entries

### 6. Discover terms locally, without a model

Use `extract-terms-local` when you want a fast local analysis pass with no API call.

Single-file usage:

```powershell
python translate_cli.py extract-terms-local source.po
python translate_cli.py extract-terms-local source.xml --mode missing --max-length 1
```

Directory-tree usage:

```powershell
python translate_cli.py extract-terms-local C:\path\to\source-tree --also-po
```

Convert a local JSON report into a PO handoff:

```powershell
python translate_cli.py extract-terms-local source.prototype-missing-terms.json --to-po
```

Local extraction behavior:

- works on one supported source file or a whole directory tree
- deduplicates repeated source messages across files
- writes a JSON report with accepted, borderline, and translation-candidate terms
- can also write a PO handoff file with `--also-po`

The local extractor deliberately filters common localization noise before scoring terms, including:

- placeholders and variable-like tokens
- CLI flags and digit-led labels
- mnemonic fragments such as underscore accelerators
- URL, tag, and attribute noise such as `href`, `src`, domain fragments, and embedded markup payloads

## Vocabulary And Rules

By default, the toolkit looks up language resources from `data/locales/<target-lang>/`.

Auto-detected resources:

- `data/locales/<target-lang>/vocab.txt`
- `data/locales/<target-lang>/vocab/`
- `data/locales/<target-lang>/rules.md`

Locale fallback is supported. For example, `fr_CA` falls back to `fr` if the region-specific resource is not present.

You can override both resources per run:

```powershell
python translate_cli.py translate source.po --vocab custom-vocab.txt --rules custom-rules.md
python translate_cli.py translate source.po --vocab custom-vocab --rules-str "Use concise imperative labels."
```

`--vocab` accepts:

- a glossary `.txt`
- a glossary `.po`
- a directory containing glossary `.txt` and `.po` files

When a vocabulary directory is used:

- files are loaded in filename order
- later duplicates override earlier ones

Recommended layout:

```text
data/
  locales/
    kk/
      vocab/
        common.txt
        colors.txt
        media.txt
      rules.md
    fr/
      vocab.txt
      rules.md
  extract/
    common/
      abbreviations.txt
      excluded_terms.txt
    en/
      stopwords.txt
      low_value_words.txt
      fixed_multiword_allowlist.txt
```

Rich vocabulary entries use this schema:

```text
source_term|target_term|part_of_speech|context_note
```

Example:

```text
archive|package|noun|software package manager context
save|store|verb|short imperative UI action
```

During translation, the toolkit still sends the full vocabulary for compatibility, but it also computes `relevant_vocabulary` per message so each message sees the subset of glossary entries that actually match it.

## Format Behavior At A Glance

| Format | Translate | Revise | Extract Terms | Notes |
| --- | --- | --- | --- | --- |
| `.po` | Yes | Yes | Yes | Source and translation live together |
| `.ts` | Yes | Yes | Yes | Source and translation live together |
| `.resx` | Yes | Yes | Yes | Revision requires `--source-file` |
| `.strings` | Yes | Yes | Yes | Revision requires `--source-file` |
| `.txt` | Yes | Yes | Yes | One line equals one message; revision requires `--source-file` |
| Android `.xml` | Yes | Yes | Yes | Translation and revision use paired-source matching |

Additional notes:

- `check` is currently for translated `.po` files
- `.strings` translation treats commented key/value entries as untranslated source entries and uncommented entries as translated entries
- `.strings` output preserves file encoding and common literal escape sequences
- `.txt` output preserves original line order and blank lines

## GUI

The Tk desktop UI is available here:

```powershell
python process_gui.py
```

The GUI is a frontend over the same backend concepts as the CLI. It includes:

- shared provider, model, API-key, thinking, and runtime controls
- instruction preview for the resolved system prompt and language rules
- a `Translate` tab with Android `Source file` support
- a `Local Extract` tab for file, folder, and JSON-to-PO local extraction workflows

## Project Layout

The repository is intentionally split between shared mechanics and task-specific logic:

```text
translate_cli.py          unified CLI entry point
process_gui.py            Tk frontend
tasks/                    task-specific contracts and runners
core/                     formats, providers, runtime, resources, shared helpers
data/locales/             per-language vocabulary and rules
data/extract/             local-extraction stop words and filters
tests/                    smoke and regression coverage
```

If you are changing behavior, the important design line is:

- `core/` owns shared mechanics
- `tasks/` owns task-specific prompts, schemas, and result handling
- `process_gui.py` should stay frontend-oriented

## Gettext Placeholder Note

If Poedit complains after placeholder order changes, check the format flag on the entry:

- `#, c-format`: reordering is allowed with positional placeholders such as `%2$s`, `%1$s`
- `#, python-format`: `%2$s` is not valid, so plain `%s` placeholders cannot be safely reordered

For `python-format`, safe reordering requires named placeholders in the source, for example:

```po
msgid "From %(src)s to %(dst)s"
msgstr "%(dst)s to %(src)s"
```

Always preserve the same placeholder set and types between source and translation.

## Smoke Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
