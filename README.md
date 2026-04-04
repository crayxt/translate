# Translation Toolkit
Toolkit for translating, checking, revising, and extracting glossary terms from PO/TS/RESX/STRINGS/TXT/Android XML localization files using Gemini, OpenAI, or Anthropic APIs.

# Recent Updates

- Added Android `<resources>` XML support for translation, revision, and term extraction. Translation and revision use a paired-source workflow (`--source-file`) for Android target exports that retain only resource IDs.
- Android XML support handles both `<string>` and `<plurals>`, pairs entries by resource name, and preserves inline XML such as `<xliff:g>` placeholders.
- Literal escape normalization now works in both directions, so formats that use literal control escapes (for example Android XML `\n`) keep them as literal escapes instead of being rewritten as real line breaks or doubled escapes.
- The bundled Kazakh rules now explicitly prefer natural active confirmation questions over passive calques for yes/no prompts such as delete, replace, hide, restart, enable, and similar UI actions.
- Translation, check, revise, and extract now use real provider-level system instructions instead of embedding faux "system" text inside the normal request prompt.
- Gemini requests now send structured batch payloads and use structured response schemas. Legacy text-prompt rendering is still kept as a provider fallback path.
- Added native OpenAI provider support, including structured JSON output support and Flex mode.
- Added native Anthropic provider support using the Messages API and tool-use for structured task results.
- Gemini now supports both AI Studio and Vertex API-key mode through the same provider. Vertex API-key mode currently uses the global endpoint only.
- Translate now supports CLI and GUI multi-file batching for same-format files, so smaller files can share one request batch.
- `process_gui.py` now exposes a read-only instruction preview so you can inspect the resolved system prompt and language rules for the active task.
- Provider, API key, thinking level, Flex mode, and Gemini backend settings are now shared GUI controls across task tabs.
- Shared runtime/bootstrap, CLI argument setup, and batch execution helpers were moved into `core/` to reduce duplication across task entrypoints.
- The local extraction pipeline now lives in `core/term_extraction.py`, its JSON/PO handoff lives in `core/term_handoff.py`, and translation uses the shared matcher to attach per-message `relevant_vocabulary` hints from the approved glossary.

# Setup
Install dependencies:

```
pip install -r requirements.txt
```

Obtain a Google API key, then set:

```
set GOOGLE_API_KEY=your_google_api_key
```

That same variable is used for:

- Gemini AI Studio / Gemini API mode
- Gemini Vertex API-key mode

Vertex API-key mode currently supports the global endpoint only.

Or use another supported provider:

```
set OPENAI_API_KEY=your_openai_api_key
set ANTHROPIC_API_KEY=your_anthropic_api_key
```

# Run

```
python translate_cli.py translate your_file.po
python translate_cli.py translate your_file.ts
python translate_cli.py translate your_file.resx
python translate_cli.py translate your_file.strings
python translate_cli.py translate your_file.txt
python translate_cli.py translate target.xml --source-file source.xml
```

Provider examples:

```
python translate_cli.py translate your_file.po --provider gemini --model gemini-3-flash-preview
python translate_cli.py translate your_file.po --provider openai --model gpt-4.1-mini
python translate_cli.py translate your_file.po --provider anthropic --model claude-sonnet-4-20250514
```

Gemini backend examples:

```
python translate_cli.py translate your_file.po --provider gemini --model gemini-3-flash-preview
python translate_cli.py translate your_file.po --provider gemini --gemini-backend vertex --google-cloud-location global --model gemini-3-flash-preview
```

Multi-file translation is supported when all input files use the same format:

```
python translate_cli.py translate a.po b.po c.po
```

Android XML uses a paired-source workflow because translated exports often retain only resource IDs:

```
python translate_cli.py translate translated.xml --source-file source.xml
```

The Tk desktop UI is also available:

```
python process_gui.py
```

Each task tab shares provider/model/API-key controls and shows the resolved instruction inputs for the current run:

- system prompt preview
- rules preview when the task uses language rules
- target-language-sensitive prompt text where applicable

The GUI also includes a dedicated `Local Extract` tab for source-side term discovery with automatic JSON+PO handoff generation, plus JSON-to-PO conversion without any model/API call.

For Android XML translation runs, the `Translate` tab also exposes a `Source file` field for the matching English XML.

For Gemini, the shared controls also include:

- backend: `studio` or `vertex`
- Google Cloud location

Current constraint: Gemini Vertex API-key mode supports `global` only.

Output files are written as `*.ai-translated.po`, `*.ai-translated.ts`, `*.ai-translated.resx`, `*.ai-translated.strings`, `*.ai-translated.txt`, or `*.ai-translated.xml`.

Set target language (default is `kk`):

```
python translate_cli.py translate your_file.po --target-lang fr
python translate_cli.py translate your_file.po --target-lang fr_CA
python translate_cli.py translate your_file.po --thinking-level medium
```

Use Flex mode when the selected provider supports it:

```
python translate_cli.py translate your_file.po --provider openai --model gpt-4.1-mini --flex
```

Force re-translation of all translatable messages:

```
python translate_cli.py translate your_file.po --retranslate-all
```

Current translation constraint:

- Android `.xml` translation currently supports one target file at a time and requires `--source-file`

Default processing behavior:

- translates unfinished messages (`untranslated` + `fuzzy`/`unfinished`)
- skips already translated messages unless `--retranslate-all` is used

By default, vocabulary and project rules are auto-detected from target language under `data/`:

- `data/locales/<target-lang>/vocab.txt`
- `data/locales/<target-lang>/rules.md`

Recommended layout:

```
data/
  locales/
    kk/
      vocab.txt
      rules.md
    fr/
      vocab.txt
      rules.md
    fr_CA/
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

Locale fallback is supported:

- for `--target-lang fr_CA`, the script first tries `data/locales/fr_CA/vocab.txt` / `data/locales/fr_CA/rules.md`
- if not found, it falls back to `data/locales/fr/vocab.txt` / `data/locales/fr/rules.md`

Legacy flat naming is still accepted as a fallback:

- `vocab-<target-lang>.txt`
- `rules-<target-lang>.md`

Override them per run:

```
python translate_cli.py translate your_file.po --vocab custom-vocab.txt --rules custom-rules.md
```

`--vocab` also accepts a glossary `.po` file. Only entries that are actually translated
(so untranslated, fuzzy, and obsolete entries are ignored) are converted to vocabulary pairs
and injected into the translation prompt:

```
python translate_cli.py translate your_file.po --vocab approved-glossary.po
```

Quick inline rule override:

```
python translate_cli.py translate your_file.po --rules-str "Use polite formal tone for settings labels."
```

`--rules-str` is merged with file-based rules when both are present.

Startup output prints both:

- `Vocabulary source` (`file:<path>` or `none`)
- `Rules source` (`file:<path>`, `inline:--rules-str`, combined, or `none`)
- `Thinking level` (`minimal`, `low`, `medium`, `high`, or provider default)

## Translation-Time Vocabulary Matching

The translation task now uses two vocabulary layers:

- top-level `vocabulary`: the full approved glossary text, kept for compatibility
- per-message `relevant_vocabulary`: only the glossary entries that match that message

The per-message suggestions are derived locally from the rich glossary schema:

```text
term|translation|pos|context/example
```

Example message payload shape:

```json
{
  "source": "Start playback",
  "relevant_vocabulary": [
    {
      "source_term": "start",
      "target_term": "бастау",
      "part_of_speech": "verb",
      "context_note": "Start playback"
    }
  ]
}
```

This reduces glossary noise compared with dumping every known term into every batch.

# Extract Glossary Terms

Run a terminology discovery pass that builds a translated glossary (`msgid=term`, `msgstr=translation`) as PO:

```
python translate_cli.py extract-terms your_file.po
python translate_cli.py extract-terms source.xml
```

Optional controls:

```
python translate_cli.py extract-terms your_file.po --out glossary.po --batch-size 200 --parallel-requests 4
python translate_cli.py extract-terms your_file.po --thinking-level high
```

Defaults:

- mode: `--mode all` (extract full glossary)
- output format: `--out-format po`
- output path: `<input>.glossary.po`

The generated glossary `.po` can be reviewed and then reused directly during translation:

```
python translate_cli.py translate your_file.po --vocab your_file.glossary.po
```

When you run missing-term extraction with `--vocab` and `--out-format po`, the output PO is merged automatically:

```
python translate_cli.py extract-terms your_file.po --mode missing --vocab data/locales/kk/vocab.txt --out-format po
```

That PO contains:

- translated entries imported from the supplied vocabulary
- newly extracted missing terms as `fuzzy` entries for review

So the resulting file can be passed straight back into translation:

```
python translate_cli.py translate your_file.po --vocab your_file.missing-terms.po
```

To get previous behavior (missing terms only, JSON output):

```
python translate_cli.py extract-terms your_file.po --mode missing --out-format json --vocab data/locales/kk/vocab.txt
```

## Local Term Discovery

Local term discovery is now a first-class task with shared core modules.

- shared extraction core: `core/term_extraction.py`
- shared JSON/PO handoff: `core/term_handoff.py`
- task entrypoint: `tasks/extract_terms_local.py`
- unified CLI command: `extract-terms-local`

Usage:

```text
python translate_cli.py extract-terms-local your_file.po
python translate_cli.py extract-terms-local source.xml
python translate_cli.py extract-terms-local your_file.po --mode missing --max-length 1
python translate_cli.py extract-terms-local your_file.po --also-po
python translate_cli.py extract-terms-local your_file.prototype-missing-terms.json --to-po
```

GUI:

- open `process_gui.py`
- use the `Local Extract` tab for local discovery; normal extraction writes both `.json` and `.po`
- enable `Convert local JSON to PO handoff` when converting an existing local-extraction JSON report into a PO glossary file

The local extraction JSON contains:

- `terms`
- `borderline_terms`
- `translation_candidates`

The same shared extraction core is also used by `translate` to build per-message glossary suggestions.

# Check Translated PO Files

Run a QA pass on an already translated `.po` file. The checker sends structured `source` / `translation`
pairs to Gemini and merges model findings with deterministic local checks for placeholders, tags,
accelerators, plural slots, and approved vocabulary usage:

```
python translate_cli.py check your_file.po
```

Default output path:

```
your_file.translation-check.json
```

Optional controls:

```
python translate_cli.py check your_file.po --probe 25
python translate_cli.py check your_file.po --out report.json --batch-size 100 --parallel-requests 4
python translate_cli.py check your_file.po --vocab approved-glossary.po --rules custom-rules.md
python translate_cli.py check your_file.po --rules-str "Keep menu labels short and imperative."
python translate_cli.py check your_file.po --thinking-level low
```

`--probe` and `--num-messages` are aliases. They limit how many translated messages are sent to Gemini,
which is useful for prompt testing and quick validation runs.

Defaults follow the same resource lookup as the translation script:

- `data/locales/<target-lang>/vocab.txt`
- `data/locales/<target-lang>/rules.md`

`--vocab` also accepts a glossary `.po` file, so you can point the checker at a reviewed glossary PO
directly.

# Revise Existing Translations

Run an instruction-driven revision pass on an already translated file. The script reviews each
existing source/translation pair, keeps entries that already satisfy the instruction, and updates
only the entries that actually need a change.

For formats that keep source and translation in the same file (`.po`, `.ts`):

```
python translate_cli.py revise your_file.po --instruction "Change the translation of Save to Store"
python translate_cli.py revise your_file.ts --instruction "Use a shorter translation for Close"
```

For formats where the translated file no longer retains the original source text (`.xml`, `.strings`, `.resx`, `.txt`),
pass both the translated file and the original source file:

```
python translate_cli.py revise translated.ai-translated.xml --source-file original.xml --instruction "Use natural active confirmation questions and preserve literal \\n escapes"
python translate_cli.py revise translated.ai-translated.strings --source-file original.strings --instruction "Change the translation of viewer to browser only where needed"
python translate_cli.py revise translated.ai-translated.resx --source-file original.resx --instruction "Replace toolbar with command bar when the source says toolbar"
python translate_cli.py revise translated.txt --source-file source.txt --instruction "Use formal tone for the word Exit"
```

Behavior:

- default output path: `<input>-revised.<ext>`
- original file is left untouched unless `--in-place` is used
- `--dry-run` previews the review without writing a file
- changed AI-reviewed entries are marked as `fuzzy` / `unfinished` for human review

Useful controls:

```
python translate_cli.py revise your_file.po --instruction "Replace preferences with settings" --dry-run --probe 50
python translate_cli.py revise your_file.po --instruction "Use the term archive instead of package" --out revised.po
python translate_cli.py revise translated.ai-translated.xml --source-file original.xml --instruction "Rewrite passive yes/no prompts as natural active Kazakh questions"
python translate_cli.py revise translated.ai-translated.strings --source-file original.strings --instruction "Shorten app names where possible" --batch-size 80 --parallel-requests 4
```

## `.strings` behavior

For `.strings` files, this project uses the following convention:

- commented key/value entries (`/* "key" = "value"; */`) are treated as untranslated source entries
- uncommented entries (`"key" = "value";`) are treated as already translated
- leading `.strings` comments are passed to the model as contextual translator notes

Translated output for `.strings` preserves file encoding (including UTF BOM when present) and writes translated entries as uncommented lines.

Escaped sequences are preserved in `.strings` output. The pipeline normalizes model-returned literal escapes to the source style for common control escapes (for example `\n`, `\t`, `\r`, `\a`, `\b`, `\f`, `\v`) to avoid accidental double-escaping like `\\n` or `\\a`.

## `.txt` behavior

For `.txt` files:

- each line is treated as one independent message
- blank/whitespace-only lines are preserved and skipped
- translated output preserves original line order and line breaks

## Android `.xml` behavior

For Android `<resources>` XML files:

- supported resource types: `<string>` and `<plurals>`
- translation and revision pair source and target entries by resource name
- `<plurals>` are paired by resource name and item quantity
- translated Android XML usually requires `--source-file` because the target export may keep only resource IDs
- term extraction can run directly on a source/English Android XML file
- inline XML such as `<xliff:g>` is preserved
- literal escapes such as `\n`, `\t`, and similar control sequences are preserved in the source style rather than rewritten as real control characters

## Internal Unified Entry Model

The translation pipeline now normalizes all formats (`.po`, `.ts`, `.resx`, `.strings`, `.txt`, `.xml`) into a shared internal entry model with common fields (message, context, note, status, flags, plural data, and string type), then syncs updates back to each native file format on save.

Status values are normalized as:

- `untranslated`: no translation content yet
- `fuzzy`: review-required translations (for example PO `fuzzy` or TS `unfinished`)
- `translated`: translated and not fuzzy
- `skipped`: non-localizable entries (for example typed/binary `.resx` resources)

## Prompt and Request Architecture

The current request flow is split into three layers:

- system instruction: hard invariants such as placeholder/tag preservation, glossary obedience, and task role
- task payload: structured batch data (messages, vocabulary, rules, target language, and task-specific fields)
- response schema: structured parsing of model output back into Python objects

Provider transport details:

- Gemini: structured request contents plus schema-backed structured responses
- OpenAI: text fallback input plus Structured Outputs on the response side
- Anthropic: native Messages API input plus tool-use for structured task results

The older plain-text prompt rendering is still present as a compatibility path for providers that do not support structured request contents.

Language rules and vocabulary stay outside the system prompt:

- system prompt handles non-negotiable behavior
- `rules.md` handles language- or project-specific style policy
- `vocab.txt` or glossary `.po` handles approved terminology

# Notes

## Gettext placeholder reordering (`%s`, `%d`)

If Poedit complains after you change placeholder order, check the format flag on that entry:

- `#, c-format`: reordering is allowed with positional placeholders, e.g. `%2$s`, `%1$s`.
- `#, python-format`: positional `%2$s` is not valid; you cannot safely reorder plain `%s` placeholders.

For `python-format` entries, reordering requires source-side named placeholders, for example:

```po
msgid "From %(src)s to %(dst)s"
msgstr "%(dst)s konumuna %(src)s"
```

Always preserve the same placeholder set and types (`%s`, `%d`, names) between `msgid` and `msgstr`.

# Smoke tests

```
python -m unittest discover -s tests -p "test_*.py" -v
```
