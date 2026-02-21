#!/usr/bin/env python3

import argparse
import asyncio
import os
import sys
import math
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Dict, List, Any, Callable, Tuple

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


class ResxEntryAdapter:
    """Adapts a .resx XML <data> element to look like a polib entry."""

    def __init__(self, data_elem: ET.Element):
        self.elem = data_elem
        self.value_elem = data_elem.find("value")
        self.comment_elem = data_elem.find("comment")
        self._flags: List[str] = []

        if self.value_elem is None:
            self.value_elem = ET.SubElement(data_elem, "value")

        self._translate = self._should_translate()

    def _should_translate(self) -> bool:
        if self.elem.get("type") or self.elem.get("mimetype"):
            return False

        value_text = self.msgid.strip()
        if not value_text:
            return False

        comment_text = (self.comment_elem.text or "") if self.comment_elem is not None else ""
        if "donottranslate" in comment_text.lower():
            return False

        if not any(ch.isalpha() for ch in value_text):
            return False

        return True

    @property
    def msgid(self) -> str:
        return self.value_elem.text if self.value_elem is not None and self.value_elem.text else ""

    @property
    def msgstr(self) -> str:
        return self.value_elem.text if self.value_elem is not None and self.value_elem.text else ""

    @msgstr.setter
    def msgstr(self, value: str):
        self.value_elem.text = value

    @property
    def flags(self):
        return self._flags

    @property
    def obsolete(self) -> bool:
        return False

    def translated(self) -> bool:
        # RESX files do not track translation state; we treat translatable
        # values as work items and skip obvious non-localizable entries.
        return not self._translate


class FileKind(str, Enum):
    PO = "po"
    TS = "ts"
    RESX = "resx"


DEFAULT_BATCH_SIZE = 1000
DEFAULT_PARALLEL_REQUESTS = 10
MIN_ITEMS_PER_WORKER = 50


def detect_file_kind(file_path: str) -> FileKind:
    lower_path = file_path.lower()
    if lower_path.endswith(".po"):
        return FileKind.PO
    if lower_path.endswith(".ts"):
        return FileKind.TS
    if lower_path.endswith(".resx"):
        return FileKind.RESX
    raise ValueError("Unsupported file type. Use .po, .ts, or .resx")


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

async def generate_content_async(client: genai.Client, model: str, prompt: str) -> Any:
    """Use Gemini async client when available, fallback to thread offload."""
    if hasattr(client, "aio") and client.aio:
        return await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
    )


async def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    batch_label: str,
    max_attempts: int = 5,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return await generate_content_async(client, model, prompt)
        except Exception as e:
            print(f"\nAPI Error [{batch_label}] (Attempt {attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                raise RuntimeError(f"Aborting [{batch_label}] due to repeated API errors.") from e
            wait_time = 2 ** attempt
            print(f"Retrying [{batch_label}] in {wait_time}s...")
            await asyncio.sleep(wait_time)


def load_ts(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing TS file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    entries: List[Any] = []
    for message in root.findall(".//message"):
        entries.append(TSEntryAdapter(message))

    output_path = file_path[:-3] + ".ai-translated.ts"

    def save_ts() -> None:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_ts, output_path


def load_resx(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing RESX file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    entries: List[Any] = []
    for data_node in root.findall("./data"):
        entries.append(ResxEntryAdapter(data_node))

    output_path = file_path[:-5] + ".ai-translated.resx"

    def save_resx() -> None:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_resx, output_path


def load_po(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing PO file: {file_path}")
    po = polib.pofile(file_path)
    entries = [e for e in po]
    output_path = file_path.replace(".po", ".ai-translated.po")
    return entries, lambda: po.save(output_path), output_path


def resolve_runtime_limits(
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
) -> Tuple[int, int, str]:
    if batch_size_arg is not None and batch_size_arg <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if parallel_arg is not None and parallel_arg <= 0:
        raise ValueError("--parallel-requests must be greater than 0")

    if batch_size_arg is None and parallel_arg is None:
        return DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL_REQUESTS, "defaults"
    if batch_size_arg is not None and parallel_arg is not None:
        return batch_size_arg, parallel_arg, "explicit"
    if batch_size_arg is not None:
        derived_parallel = max(
            1,
            min(DEFAULT_PARALLEL_REQUESTS, math.ceil(total_items / batch_size_arg)),
        )
        return batch_size_arg, derived_parallel, "auto parallel"

    # Only parallel was provided.
    assert parallel_arg is not None
    derived_batch = max(MIN_ITEMS_PER_WORKER, math.ceil(total_items / parallel_arg))
    return derived_batch, parallel_arg, "auto batch"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process and translate PO, TS, or RESX files using Google Gemini"
    )
    parser.add_argument("file", help="Input .po, .ts, or .resx file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk (Kazakh)")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent Gemini requests (auto if omitted)")
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
    try:
        file_kind = detect_file_kind(file_path)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    if file_kind == FileKind.TS:
        entries, save_callback, output_path = load_ts(file_path)
    elif file_kind == FileKind.RESX:
        entries, save_callback, output_path = load_resx(file_path)
    else:
        entries, save_callback, output_path = load_po(file_path)

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

    try:
        batch_size, parallel_requests, limits_mode = resolve_runtime_limits(
            total_items=total,
            batch_size_arg=args.batch_size,
            parallel_arg=args.parallel_requests,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    print("Startup configuration:")
    print(f"  Model: {args.model}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")

    all_batches: List[List[Any]] = []
    small_file_threshold = parallel_requests * batch_size
    if total < small_file_threshold:
        # For smaller files, split work evenly but keep >=50 items per worker.
        if total >= parallel_requests * MIN_ITEMS_PER_WORKER:
            worker_count = parallel_requests
        else:
            worker_count = total // MIN_ITEMS_PER_WORKER

        if worker_count < 2:
            all_batches = [work_items]
            print(
                f"Found {total} items to translate. "
                "Small file mode: using 1 batch (minimum items/worker not met)."
            )
        else:
            base = total // worker_count
            remainder = total % worker_count
            start = 0
            for worker_index in range(worker_count):
                size = base + (1 if worker_index < remainder else 0)
                end = start + size
                all_batches.append(work_items[start:end])
                start = end
            print(
                f"Found {total} items to translate. "
                f"Small file mode: split evenly into {worker_count} parallel batches "
                f"(min batch size: {min(len(b) for b in all_batches)})."
            )
    else:
        batches = math.ceil(total / batch_size)
        all_batches = [
            work_items[i * batch_size: (i + 1) * batch_size]
            for i in range(batches)
        ]
        print(
            f"Found {total} items to translate. "
            f"Running up to {parallel_requests} batch requests in parallel."
        )

    batches = len(all_batches)
    print(
        f"Total batches: {batches}"
    )

    async def process_batch(batch_index: int, batch: List[Any], sem: asyncio.Semaphore):
        async with sem:
            msg_map: Dict[str, str] = {}
            for i, entry in enumerate(batch):
                text = entry.msgid
                if hasattr(entry, "msgid_plural") and entry.msgid_plural:
                    text = f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
                msg_map[str(i)] = text

            prompt = build_prompt(
                msg_map,
                args.source_lang,
                args.target_lang,
                vocabulary_text,
            )

            response = await generate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                batch_label=f"batch {batch_index + 1}/{batches}",
                max_attempts=5,
            )
            translations = parse_response(response.text or "")

            missing_indices = [i for i in range(len(batch)) if str(i) not in translations]
            if missing_indices:
                print(
                    f"  Warning [batch {batch_index + 1}/{batches}]: "
                    f"{len(missing_indices)} items missing from response. Retrying them..."
                )
                retry_map: Dict[str, str] = {}
                for idx in missing_indices:
                    entry = batch[idx]
                    text = entry.msgid
                    if hasattr(entry, "msgid_plural") and entry.msgid_plural:
                        text = f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
                    retry_map[str(idx)] = text

                retry_prompt = build_prompt(
                    retry_map,
                    args.source_lang,
                    args.target_lang,
                    vocabulary_text,
                )

                try:
                    retry_resp = await generate_with_retry(
                        client=client,
                        model=args.model,
                        prompt=retry_prompt,
                        batch_label=f"batch {batch_index + 1}/{batches} missing-items",
                        max_attempts=3,
                    )
                    retry_translations = parse_response(retry_resp.text or "")
                    translations.update(retry_translations)
                except Exception as e:
                    print(f"  Retry failed [batch {batch_index + 1}/{batches}]: {e}")

            return batch_index, batch, translations

    async def run_translation() -> int:
        translated_count = 0
        sem = asyncio.Semaphore(parallel_requests)

        tasks = [
            asyncio.create_task(process_batch(batch_index, batch, sem))
            for batch_index, batch in enumerate(all_batches)
        ]

        completed_batches = 0
        for finished in asyncio.as_completed(tasks):
            batch_index, batch, translations = await finished
            batch_translated = 0

            for i, entry in enumerate(batch):
                key = str(i)
                if key in translations and translations[key]:
                    val = translations[key]
                    if hasattr(entry, "msgid_plural") and entry.msgid_plural:
                        entry.msgstr_plural[0] = val
                    else:
                        entry.msgstr = val

                    if "fuzzy" not in entry.flags:
                        entry.flags.append("fuzzy")
                    batch_translated += 1

            translated_count += batch_translated
            completed_batches += 1
            percent = (translated_count / total) * 100
            print(
                f"Progress: {percent:.1f}% ({translated_count}/{total}), "
                f"completed batches: {completed_batches}/{batches} "
                f"(latest: {batch_index + 1}/{batches})"
            )

            if save_callback:
                save_callback()

        return translated_count

    try:
        translated_count = asyncio.run(run_translation())
    except RuntimeError as e:
        sys.exit(str(e))

    if save_callback:
        save_callback()

    print("\nTranslation complete.")
    print(f"Saved file: {output_path}")
    print("All AI-generated translations are marked as fuzzy/unfinished for human review.")


if __name__ == "__main__":
    main()
