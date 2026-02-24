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

# Smoke tests

```
python -m unittest discover -s tests -p "test_*.py" -v
```
