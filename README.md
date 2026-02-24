# Translate script
Script for translating PO/TS/RESX localization files using Google Gemini API.

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
```

Output files are written as `*.ai-translated.po`, `*.ai-translated.ts`, or `*.ai-translated.resx`.

By default, vocabulary and project rules are loaded from:

- `vocab-kk.txt`
- `rules-kk.md`

Override them per run:

```
python process.py your_file.po --vocab custom-vocab.txt --rules custom-rules.md
```

Quick inline rule override:

```
python process.py your_file.po --rules-str "Use polite formal tone for settings labels."
```

When rules are active, startup output prints `Rules source` (file path and/or `--rules-str`).

# Extract Missing Terms

Run a terminology discovery pass that suggests missing glossary terms from source messages:

```
python extract_terms.py your_file.po
```

Optional controls:

```
python extract_terms.py your_file.po --vocab vocab-kk.txt --out missing-terms.json --batch-size 200 --parallel-requests 4
```

Output is saved as `<input>.missing-terms.json` unless `--out` is specified.

# Smoke tests

```
python -m unittest discover -s tests -p "test_*.py" -v
```
