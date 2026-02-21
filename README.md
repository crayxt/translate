# Translate script
My script for translation files in PO/TS/RESX format using Google Gemini API.


# Run method


Have a python-polib, google-genai installed


Obtain Google API Key (you can get a trial from AI Studio).

Edit process.py to set your language code, model and batch size.


```
set GOOGLE_API_KEY=your_google_api_key
python process.py your_file.po
python process.py your_file.ts
python process.py your_file.resx
```


It will save translated files as `*.ai-translated.po`, `*.ai-translated.ts`, or `*.ai-translated.resx`.


Adjust your instruction and vocabulary.


Good luck!
