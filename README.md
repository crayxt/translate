# Translate script
Script for translating PO/TS/RESX/STRINGS/TXT localization files using Google Gemini API.

# Setup
Install dependencies:

```
pip install -r requirements.txt
```

Obtain a Google API key (for example from AI Studio), then set:

```
set GOOGLE_API_KEY=your_google_api_key
```

# Run

```
python process.py your_file.po
python process.py your_file.ts
python process.py your_file.resx
python process.py your_file.strings
python process.py your_file.txt
```

Output files are written as `*.ai-translated.po`, `*.ai-translated.ts`, `*.ai-translated.resx`, `*.ai-translated.strings`, or `*.ai-translated.txt`.

Set target language (default is `kk`):

```
python process.py your_file.po --target-lang fr
python process.py your_file.po --target-lang fr_CA
```

Force re-translation of all translatable messages:

```
python process.py your_file.po --retranslate-all
```

Default processing behavior:

- translates unfinished messages (`untranslated` + `fuzzy`/`unfinished`)
- skips already translated messages unless `--retranslate-all` is used

By default, vocabulary and project rules are auto-detected from target language under `data/`:

- `data/<target-lang>/vocab.txt`
- `data/<target-lang>/rules.md`

Recommended layout:

```
data/
  kk/
    vocab.txt
    rules.md
  fr/
    vocab.txt
    rules.md
  fr_CA/
    vocab.txt
    rules.md
```

Locale fallback is supported:

- for `--target-lang fr_CA`, the script first tries `data/fr_CA/vocab.txt` / `data/fr_CA/rules.md`
- if not found, it falls back to `data/fr/vocab.txt` / `data/fr/rules.md`

Legacy flat naming is still accepted as a fallback:

- `vocab-<target-lang>.txt`
- `rules-<target-lang>.md`

Override them per run:

```
python process.py your_file.po --vocab custom-vocab.txt --rules custom-rules.md
```

`--vocab` also accepts a glossary `.po` file. Only entries that are actually translated
(so untranslated, fuzzy, and obsolete entries are ignored) are converted to vocabulary pairs
and injected into the translation prompt:

```
python process.py your_file.po --vocab approved-glossary.po
```

Quick inline rule override:

```
python process.py your_file.po --rules-str "Use polite formal tone for settings labels."
```

`--rules-str` is merged with file-based rules when both are present.

Startup output prints both:

- `Vocabulary source` (`file:<path>` or `none`)
- `Rules source` (`file:<path>`, `inline:--rules-str`, combined, or `none`)

# Extract Glossary Terms

Run a terminology discovery pass that builds a translated glossary (`msgid=term`, `msgstr=translation`) as PO:

```
python extract_terms.py your_file.po
```

Optional controls:

```
python extract_terms.py your_file.po --out glossary.po --batch-size 200 --parallel-requests 4
```

Defaults:

- mode: `--mode all` (extract full glossary)
- output format: `--out-format po`
- output path: `<input>.glossary.po`

The generated glossary `.po` can be reviewed and then reused directly during translation:

```
python process.py your_file.po --vocab your_file.glossary.po
```

When you run missing-term extraction with `--vocab` and `--out-format po`, the output PO is merged automatically:

```
python extract_terms.py your_file.po --mode missing --vocab data/kk/vocab.txt --out-format po
```

That PO contains:

- translated entries imported from the supplied vocabulary
- newly extracted missing terms as `fuzzy` entries for review

So the resulting file can be passed straight back into translation:

```
python process.py your_file.po --vocab your_file.missing-terms.po
```

To get previous behavior (missing terms only, JSON output):

```
python extract_terms.py your_file.po --mode missing --out-format json --vocab data/kk/vocab.txt
```

# Check Translated PO Files

Run a QA pass on an already translated `.po` file. The checker sends structured `source` / `translation`
pairs to Gemini and merges model findings with deterministic local checks for placeholders, tags,
accelerators, plural slots, and approved vocabulary usage:

```
python check_translations.py your_file.po
```

Default output path:

```
your_file.translation-check.json
```

Optional controls:

```
python check_translations.py your_file.po --probe 25
python check_translations.py your_file.po --out report.json --batch-size 100 --parallel-requests 4
python check_translations.py your_file.po --vocab approved-glossary.po --rules custom-rules.md
python check_translations.py your_file.po --rules-str "Keep menu labels short and imperative."
python check_translations.py your_file.po --thinking-level low
```

`--probe` and `--num-messages` are aliases. They limit how many translated messages are sent to Gemini,
which is useful for prompt testing and quick validation runs.

Defaults follow the same resource lookup as the translation script:

- `data/<target-lang>/vocab.txt`
- `data/<target-lang>/rules.md`

`--vocab` also accepts a glossary `.po` file, so you can point the checker at a reviewed glossary PO
directly.

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

## Internal Unified Entry Model

The translation pipeline now normalizes all formats (`.po`, `.ts`, `.resx`, `.strings`, `.txt`) into a shared internal entry model with common fields (message, context, note, status, flags, plural data, and string type), then syncs updates back to each native file format on save.

Status values are normalized as:

- `untranslated`: no translation content yet
- `fuzzy`: review-required translations (for example PO `fuzzy` or TS `unfinished`)
- `translated`: translated and not fuzzy
- `skipped`: non-localizable entries (for example typed/binary `.resx` resources)

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
