# Translate script
My script for translation files in PO/TS format using Google Gemini API.


# Run method


Have a python-polib, google-genai installed


Obtain Google API Key (you can get a trial from AI Studio).

Edit process.py to set your language code, model and batch size.


```
set GOOGLE_API_KEY=your_google_api_key
python process.py your_file.po
```


It will save translated PO/TS file with ai-translated.po extension


Adjust your instruction and vocabulary.


Good luck!
