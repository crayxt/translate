# Translation Toolkit

Translate and maintain software localization files with one CLI and one GUI.

This repository is for software-localization work, not generic document translation. It gives you a shared backend for five jobs:

- translate unfinished localization files
- revise existing translations with a precise instruction
- check translated PO or TS files for QA issues
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
2. Shared language resources: glossary and rules are loaded by target language
3. One format backend: supported file types are normalized into a shared entry model and then written back in their native format

That matters because the same glossary, rules, runtime controls, batching, and format handling are reused across CLI and GUI instead of being reimplemented per script.

The preferred entry point is:

```text
python translate_cli.py
```

`translate_cli.py` is the main CLI surface, and `process_gui.py` is the GUI entry point.

## Task Docs

Detailed task guides live in `docs/`:

- `docs/translate.md`
- `docs/check.md`
- `docs/extract.md`
- `docs/extract-local.md`
- `docs/revise.md`
- `docs/extraction-refactor.md`

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
python translate_cli.py translate source.po --target-lang kk
python translate_cli.py translate source.ts --target-lang kk
python translate_cli.py translate source.resx --target-lang kk
python translate_cli.py translate source.strings --target-lang kk
python translate_cli.py translate source.txt --target-lang kk
```

Translate several files in one run when they are the same format:

```powershell
python translate_cli.py translate first.po second.po third.po --target-lang kk
```

Choose provider and model explicitly:

```powershell
python translate_cli.py translate source.po --target-lang kk --provider openai --model your-model
python translate_cli.py translate source.po --target-lang kk --provider anthropic --model your-model
python translate_cli.py translate source.po --target-lang kk --provider gemini --model your-model
```

Useful controls:

```powershell
python translate_cli.py translate source.po --target-lang kk
python translate_cli.py translate source.po --target-lang kk --thinking-level medium
python translate_cli.py translate source.po --target-lang kk --batch-size 100 --parallel-requests 4
python translate_cli.py translate source.po --target-lang kk --translation-scope untranslated
python translate_cli.py translate source.po --target-lang kk --translation-scope all
python translate_cli.py translate source.po --target-lang kk --flex
python translate_cli.py translate source.po --target-lang kk --warnings-report
```

Behavior:

- target language is explicit; pass `--target-lang` for translation, revision, checking, and model-based extraction
- by default, only unfinished messages are translated; in this task, `unfinished` means fuzzy plus untranslated
- `--translation-scope unfinished` translates fuzzy plus untranslated entries
- `--translation-scope untranslated` translates only untranslated entries
- `--translation-scope all` forces already translated messages through translation again
- `--retranslate-all` remains available as a compatibility alias for `--translation-scope all`
- recursive directory translation skips generated toolkit artifacts such as `*.ai-translated.*`, `*.glossary.po`, `*.missing-terms.po`, and `*.prototype-*.po`
- when the scan root is this toolkit repository itself, recursive translation also skips toolkit-owned directories such as `data/`, `logs/`, `docs/`, `tests/`, `tasks/`, and `core/`
- translated output is written as `*.ai-translated.<ext>`
- `--warnings-report` also writes `*.translation-warnings.json` with only the messages where the model reported ambiguity, unclear meaning, risky glossary choice, or another review-worthy concern

Warnings sidecar behavior:

- warnings are emitted per translated message, not as one batch-level summary
- each warning item includes structured issues with `code`, `message`, and `severity`, plus the source text, translated text, and any available `context`, `note`, or matched `relevant_vocabulary`
- translation warning codes use the `translate.*` namespace, for example `translate.ambiguous_term`
- `severity` is `warning` for real risk or ambiguity, and `info` for notable but non-risk notes such as preserved structure
- this is a lightweight translator self-report; the dedicated `check` task remains the real QA pass

### 2. Translate Android XML with a paired source file

Android translated exports often contain only resource IDs on the target side, so translation uses a paired-source workflow:

```powershell
python translate_cli.py translate translated.xml --source-file source.xml --target-lang kk
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
python translate_cli.py revise translated.po --target-lang kk --instruction "Use a shorter term for Preferences"
python translate_cli.py revise translated.ts --target-lang kk --instruction "Replace archive with package where the source says package"
```

For formats where the translated file no longer carries the original source text, pass the matching source file:

```powershell
python translate_cli.py revise translated.ai-translated.xml --target-lang kk --source-file source.xml --instruction "Use natural confirmation questions and preserve literal \\n escapes"
python translate_cli.py revise translated.ai-translated.strings --target-lang kk --source-file source.strings --instruction "Shorten viewer labels where possible"
python translate_cli.py revise translated.ai-translated.resx --target-lang kk --source-file source.resx --instruction "Use command bar instead of toolbar"
python translate_cli.py revise translated.txt --target-lang kk --source-file source.txt --instruction "Use formal tone for Exit"
```

Revision behavior:

- default output path is `<input>.revised.<ext>`
- `--in-place` overwrites the translated input file
- `--dry-run` reviews and reports changes without writing output
- changed AI-reviewed entries are marked as review-required where the format supports it

### 4. Check a translated PO or TS file

Use `check` for QA on an already translated PO or TS file:

```powershell
python translate_cli.py check translated.po --target-lang kk
python translate_cli.py check translated.ts --target-lang kk
python translate_cli.py check translated.po --target-lang kk --probe 50
python translate_cli.py check translated.po --target-lang kk --out report.json --include-ok
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

Check-report issue shape:

- each issue uses a structured `code`, `message`, and `severity`
- check issue codes use the `check.*` namespace, for example `check.meaning`, `check.placeholder`, or `check.terminology`

### 5. Extract glossary terms with a model

Use `extract-terms` when you want the model to propose glossary entries:

```powershell
python translate_cli.py extract-terms source.po --target-lang kk
python translate_cli.py extract-terms source.xml --target-lang kk
```

Useful variants:

```powershell
python translate_cli.py extract-terms source.po --target-lang kk --mode missing --glossary data/locales/kk/glossary.po --out-format po
python translate_cli.py extract-terms source.po --target-lang kk --mode missing --out-format json --glossary data/locales/kk/glossary.po
python translate_cli.py extract-terms source.po --target-lang kk --out glossary.po --batch-size 200 --parallel-requests 4
```

Modes:

- `all`: build a broader glossary from the source content
- `missing`: focus on terms that are not already in your existing glossary

Output defaults:

- `all` + `po` -> `<input>.glossary.po`
- `missing` + `po` -> `<input>.missing-terms.po`
- `missing` + `json` -> `<input>.missing-terms.json`

When you run missing-term extraction with `--out-format po`, the generated PO is designed to go straight back into review and then translation:

- known terms from the supplied glossary are imported
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

## Recommended Workflow

For larger localization work, the recommended flow is:

1. Run local extraction first.
2. Translate the resulting glossary PO handoff.
3. Review and approve that glossary.
4. Use the approved glossary as the base glossary for the main translation.
5. Review and approve the main translated source file.

In practice, that looks like this:

```powershell
# 1. Local extraction from one file or a source tree
python translate_cli.py extract-terms-local source.po --mode missing --also-po
python translate_cli.py extract-terms-local C:\path\to\source-tree --mode missing --also-po

# 2. Translate the generated glossary PO handoff
python translate_cli.py translate source.prototype-missing-terms.po --target-lang kk --glossary data/locales/kk/glossary.po

# 3. Review and approve the glossary PO
#    Keep only good terms, fix bad translations, and save the approved glossary.

# 4. Use the approved glossary as the base glossary for the main translation
python translate_cli.py translate source.po --target-lang kk --glossary approved-glossary.po

# 5. Review and approve the main translated source file
```

Why this workflow is recommended:

- `extract-terms-local` can deterministically avoid terms already present in your approved glossary and skip local noise such as stop words, excluded abbreviations, placeholders, tags, and weak phrase candidates
- the glossary is reviewed before bulk translation, so terminology is stabilized early
- the main `translate` task can load the approved glossary PO directly through `--glossary`
- the final source translation still needs review, because approved terminology does not replace full QA

Keep a distinction between:

- candidate glossary output from local extraction
- approved glossary used as translation input

That approved glossary can stay as a reviewed `.po` passed with `--glossary`, or it can be merged into your canonical locale glossary under `data/locales/<target-lang>/`.

## Glossary And Rules

By default, the toolkit looks up language resources from `data/locales/<target-lang>/`.

Auto-detected resources:

- `data/locales/<target-lang>/glossary.po`
- `data/locales/<target-lang>/glossary/`
- `data/locales/<target-lang>/rules.md`

You can override both resources per run:

```powershell
python translate_cli.py translate source.po --target-lang kk --glossary approved-glossary.po --rules custom-rules.md
python translate_cli.py translate source.po --target-lang kk --glossary custom-glossary --rules-str "Use concise imperative labels."
```

`--glossary` accepts:

- a glossary `.po`
- a glossary `.tbx`
- a directory containing glossary `.po` and `.tbx` files

When a glossary directory is used:

- files are loaded in filename order
- later duplicates override earlier ones

Recommended layout:

```text
data/
  locales/
    kk/
      glossary/
        common.po
        colors.po
        media.tbx
      glossary.po
      rules.md
    fr/
      glossary.po
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

Rich glossary entries use this schema:

```text
source_term|target_term|part_of_speech|context_note
```

Example:

```text
archive|package|noun|software package manager context
save|store|verb|short imperative UI action
```

During translation, the toolkit still sends the full glossary for compatibility, but it also computes `relevant_vocabulary` per message so each message sees the subset of glossary entries that actually match it.

When warnings reporting is enabled, the translation response can also include a per-message `warnings` field. Those warnings are written to a separate JSON sidecar so you can inspect ambiguous or risky messages without rereading the whole translated file.

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

- `check` currently supports translated `.po` and `.ts` files
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

Translate-tab note:

- the GUI enables the translation warnings JSON sidecar by default
- a normal translate run writes the translated output file and a matching `*.translation-warnings.json` report

## Project Layout

The repository is intentionally split between shared mechanics and task-specific logic:

```text
translate_cli.py          unified CLI entry point
process_gui.py            Tk frontend
tasks/                    task-specific contracts and runners
core/                     formats, providers, runtime, resources, shared helpers
data/locales/             per-language glossary and rules
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
