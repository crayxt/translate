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

Quick inline rule override:

```
python process.py your_file.po --rules-str "Use polite formal tone for settings labels."
```

`--rules-str` is merged with file-based rules when both are present.

Startup output prints both:

- `Vocabulary source` (`file:<path>` or `none`)
- `Rules source` (`file:<path>`, `inline:--rules-str`, combined, or `none`)

# Extract Missing Terms

Run a terminology discovery pass that suggests missing glossary terms from source messages:

```
python extract_terms.py your_file.po
```

Optional controls:

```
python extract_terms.py your_file.po --vocab data/kk/vocab.txt --out missing-terms.json --batch-size 200 --parallel-requests 4
```

Output is saved as `<input>.missing-terms.json` unless `--out` is specified.

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
