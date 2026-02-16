#!/usr/bin/env python3

import argparse
import os
import sys
import math
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

import polib
from google import genai


# ===========================
# TS File Adapter
# ===========================

class TSEntryAdapter:
    """Adapts a Qt .ts XML <message> element to look like a polib entry."""
    
    def __init__(self, message_elem: ET.Element):
        self.elem = message_elem
        self.source_elem = message_elem.find("source")
        self.translation_elem = message_elem.find("translation")
        
        # Ensure translation element exists
        if self.translation_elem is None:
            self.translation_elem = ET.SubElement(message_elem, "translation")

    @property
    def msgid(self) -> str:
        return self.source_elem.text if self.source_elem is not None else ""

    @property
    def msgstr(self) -> str:
        return self.translation_elem.text if self.translation_elem is not None else ""

    @msgstr.setter
    def msgstr(self, value: str):
        self.translation_elem.text = value

    @property
    def flags(self):
        return self._Flags(self.translation_elem)

    @property
    def obsolete(self) -> bool:
        # TS files don't strictly have an obsolete flag in the same way
        return False

    def translated(self) -> bool:
        t = self.translation_elem
        if t is None or not t.text:
            return False
        # In TS, type="unfinished" means it's not fully translated/approved
        if t.get("type") == "unfinished":
            return False
        return True

    class _Flags(list):
        """Helper to mimic polib's flags list, mapping 'fuzzy' to type='unfinished'."""
        def __init__(self, elem: ET.Element):
            self.elem = elem
            super().__init__()
            if self.elem.get("type") == "unfinished":
                self.append("fuzzy")
        
        def append(self, item):
            if item == "fuzzy":
                self.elem.set("type", "unfinished")
            super().append(item)
            
        def __contains__(self, item):
            if item == "fuzzy":
                return self.elem.get("type") == "unfinished"
            return super().__contains__(item)


# ===========================
# Prompt & translation rules
# ===========================

SYSTEM_INSTRUCTION = """
You are a professional software localization translator.

STRICT RULES:
- Preserve all placeholders EXACTLY (%s, %d, %(name)s, {var}, {{var}})
- % and _ placeholders in messages like '_Apply' should not be assigned to Kazakh letters "әіңғүұқөһ" in translation
- Preserve HTML/XML tags EXACTLY
- Do NOT reorder placeholders
- Do NOT add or remove content
- Keep original punctuation and capitalization style
- Use consistent terminology
- Translate ONLY the message text
- There Is No Camel Case in Kazakh. But ALL CAPS should be ALL CAPS.
- If the input contains 'Singular:' and 'Plural:', return only the single correct translation for Kazakh (which typically uses the singular form with numbers).
"""


def build_prompt(
    messages: Dict[str, str],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
) -> str:
    vocab_block = f"\nProject vocabulary (mandatory):\n{vocabulary}\n" if vocabulary else ""

    blocks: List[str] = []
    for msg_id, text in messages.items():
        blocks.append(
            f"<<<<MSG:{msg_id}>>>>\n{text}\n<<<<END>>>>"
        )

    return f"""
{SYSTEM_INSTRUCTION}

Project context:
This is a software application UI localization project.
Source language: {source_lang}
Target language: {target_lang}
{vocab_block}

Instructions:
- Translate each message independently
- Do NOT merge messages
- Return translations using the SAME markers and IDs
- Use consistent translation throughout the messages.

Messages to translate:
{chr(10).join(blocks)}
"""


# ===========================
# Gemini response parsing
# ===========================

def parse_response(text: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    current_id: str | None = None
    buffer: List[str] = []

    for line in text.splitlines():
        if line.startswith("<<<<MSG:"):
            current_id = line.replace("<<<<MSG:", "").replace(">>>>", "").strip()
            buffer = []
        elif line.startswith("<<<<END>>>>"):
            if current_id is not None:
                results[current_id] = "\n".join(buffer).strip()
            current_id = None
        elif current_id is not None:
            buffer.append(line)

    return results


# ===========================
# Main logic
# ===========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process and translate PO or TS files using Google Gemini"
    )
    parser.add_argument("file", help="Input .po or .ts file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk (Kazakh)")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--vocab", default="vocab-kk.txt", help="Optional vocabulary file")

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    vocabulary_text: str | None = None
    if args.vocab:
        try:
            with open(args.vocab, "r", encoding="utf-8") as f:
                vocabulary_text = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Vocabulary file '{args.vocab}' not found.")

    file_path = args.file
    is_ts = file_path.endswith(".ts")
    entries = []
    save_callback = None
    output_path = ""

    if is_ts:
        print(f"Processing TS file: {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find all message tags
        # Structure is usually <context><message>...</message></context>
        # but we just want all messages
        for message in root.findall(".//message"):
            entries.append(TSEntryAdapter(message))
            
        output_path = file_path.replace(".ts", ".ai-translated.ts")
        
        def save_ts():
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            
        save_callback = save_ts

    else:
        print(f"Processing PO file: {file_path}")
        po = polib.pofile(file_path)
        entries = [e for e in po] # Keep list reference mostly for consistent type hint, though filtering happens next
        
        output_path = file_path.replace(".po", ".ai-translated.po")
        save_callback = lambda: po.save(output_path)

    # Filter for untranslated items
    # Note: TSEntryAdapter implements .obsolete and .translated() to match polib
    work_items = [
        entry for entry in entries
        if not entry.obsolete and not entry.translated()
    ]

    total = len(work_items)
    if total == 0:
        print("No untranslated or fuzzy messages found.")
        return

    batches = math.ceil(total / args.batch_size)
    translated_count = 0

    print(f"Found {total} items to translate.")

    for batch_index in range(batches):
        batch = work_items[
            batch_index * args.batch_size:
            (batch_index + 1) * args.batch_size
        ]

        msg_map: Dict[str, str] = {}
        for i, entry in enumerate(batch):
            text = entry.msgid
            # Check for PO plural (polib.POEntry has msgid_plural)
            if hasattr(entry, 'msgid_plural') and entry.msgid_plural:
                # Present both forms to the LLM for context
                text = f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
            msg_map[str(i)] = text

        prompt = build_prompt(
            msg_map,
            args.source_lang,
            args.target_lang,
            vocabulary_text,
        )

        response = None
        # Retry up to 5 times for 503/transient errors
        for attempt in range(1, 6):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                )
                break
            except Exception as e:
                print(f"\nAPI Error (Attempt {attempt}/5): {e}")
                if attempt == 5:
                    sys.exit("Aborting due to repeated API errors.")
                # Exponential backoff: 2, 4, 8, 16 seconds...
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        translations = parse_response(response.text or "")

        # --- RETRY LOGIC FOR MISSING ITEMS ---
        missing_indices = []
        for i in range(len(batch)):
            if str(i) not in translations:
                missing_indices.append(i)

        if missing_indices:
            print(f"  Warning: {len(missing_indices)} items missing from response. Retrying them...")
            # Build mini-batch for missing items
            retry_map = {}
            for idx in missing_indices:
                entry = batch[idx]
                text = entry.msgid
                if hasattr(entry, 'msgid_plural') and entry.msgid_plural:
                    text = f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
                retry_map[str(idx)] = text
            
            # Simple retry prompt
            retry_prompt = build_prompt(
                retry_map, args.source_lang, args.target_lang, vocabulary_text
            )
            
            try:
                retry_resp = client.models.generate_content(
                    model=args.model,
                    contents=retry_prompt,
                )
                retry_translations = parse_response(retry_resp.text or "")
                # Merge retry results
                translations.update(retry_translations)
            except Exception as e:
                print(f"  Retry failed: {e}")

        # --- APPLY TRANSLATIONS ---
        for i, entry in enumerate(batch):
            key = str(i)
            if key in translations and translations[key]:
                val = translations[key]
                # Check for PO plural and assign correctly
                if hasattr(entry, 'msgid_plural') and entry.msgid_plural:
                    # For Kazakh (nplurals=1), we assign to index 0
                    # If dealing with other languages, this might need logic based on nplurals
                    entry.msgstr_plural[0] = val
                else:
                    entry.msgstr = val

                if "fuzzy" not in entry.flags:
                    entry.flags.append("fuzzy")
                translated_count += 1

        percent = (translated_count / total) * 100
        print(f"Progress: {percent:.1f}% ({translated_count}/{total})")

        # Save progress after every batch to prevent total data loss on crash
        if save_callback:
            save_callback()

        time.sleep(0.5)

    if save_callback:
        save_callback()

    print("\nTranslation complete.")
    print(f"Saved file: {output_path}")
    print("All AI-generated translations are marked as fuzzy/unfinished for human review.")


if __name__ == "__main__":
    main()
