#!/usr/bin/env python3

import argparse
import ast
import asyncio
import json
import re
import os
import sys
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Callable, Tuple

import polib
from google import genai
from google.genai import types as genai_types

PO_WRAP_WIDTH = 78


# ===========================
# TS File Adapter
# ===========================

class TSEntryAdapter:
    """Adapts a Qt .ts XML <message> element to look like a polib entry."""
    
    def __init__(self, message_elem: ET.Element, context_name: str | None = None):
        self.elem = message_elem
        self.source_elem = message_elem.find("source")
        self.translation_elem = message_elem.find("translation")
        self.comment_elem = message_elem.find("comment")
        self.extracomment_elem = message_elem.find("extracomment")
        self.context_name = context_name or ""
        
        # Ensure translation element exists
        if self.translation_elem is None:
            self.translation_elem = ET.SubElement(message_elem, "translation")

    @property
    def msgid(self) -> str:
        return self.source_elem.text if self.source_elem is not None else ""

    @property
    def msgid_plural(self) -> str:
        if self.elem.get("numerus") == "yes":
            # Qt TS stores only a single source string for numerus entries.
            # Reuse it as a plural hint so downstream logic uses plural paths.
            return self.msgid
        return ""

    @property
    def msgstr(self) -> str:
        if self.translation_elem is None:
            return ""
        if self.elem.get("numerus") == "yes":
            forms = [self._clean_form_text(node.text) for node in self._numerusform_nodes()]
            return forms[0] if forms else ""
        return self.translation_elem.text if self.translation_elem.text else ""

    @msgstr.setter
    def msgstr(self, value: str):
        if self.translation_elem is None:
            return
        if self.elem.get("numerus") == "yes":
            plural_map = self.msgstr_plural
            if plural_map:
                for key in list(plural_map.keys()):
                    plural_map[key] = value
            else:
                self._append_numerusform(value)
            return
        self.translation_elem.text = value

    @property
    def msgstr_plural(self):
        if self.elem.get("numerus") != "yes" or self.translation_elem is None:
            return {}
        return self._PluralMap(self)

    @property
    def flags(self):
        return self._Flags(self.translation_elem)

    @property
    def obsolete(self) -> bool:
        # TS files don't strictly have an obsolete flag in the same way
        return False

    def translated(self) -> bool:
        t = self.translation_elem
        if t is None:
            return False
        # In TS, type="unfinished" means it's not fully translated/approved
        if t.get("type") == "unfinished":
            return False
        if self.elem.get("numerus") == "yes":
            forms = self._numerusform_nodes()
            if not forms:
                return False
            return all(bool(self._clean_form_text(node.text)) for node in forms)
        if not t.text:
            return False
        return True

    @property
    def prompt_context(self) -> str:
        return self.context_name.strip()

    @property
    def prompt_note(self) -> str:
        parts: List[str] = []
        for elem in (self.comment_elem, self.extracomment_elem):
            text = (elem.text or "") if elem is not None else ""
            text = text.strip()
            if text:
                parts.append(text)
        return " | ".join(parts)

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

    @staticmethod
    def _clean_form_text(value: str | None) -> str:
        return value if value else ""

    def _numerusform_nodes(self) -> List[ET.Element]:
        if self.translation_elem is None:
            return []
        return self.translation_elem.findall("numerusform")

    def _append_numerusform(self, value: str) -> ET.Element:
        if self.translation_elem is None:
            self.translation_elem = ET.SubElement(self.elem, "translation")
        node = ET.SubElement(self.translation_elem, "numerusform")
        node.text = value
        return node

    class _PluralMap(dict):
        """Dict-like view over Qt TS <numerusform> nodes."""

        def __init__(self, adapter: "TSEntryAdapter"):
            self.adapter = adapter
            super().__init__()
            for idx, node in enumerate(self.adapter._numerusform_nodes()):
                super().__setitem__(idx, adapter._clean_form_text(node.text))

        def _ensure_node(self, idx: int) -> ET.Element:
            nodes = self.adapter._numerusform_nodes()
            while len(nodes) <= idx:
                self.adapter._append_numerusform("")
                nodes = self.adapter._numerusform_nodes()
            return nodes[idx]

        def __setitem__(self, key, value):
            try:
                idx = int(key)
            except (TypeError, ValueError):
                idx = len(self)
            if idx < 0:
                idx = 0
            text_value = value if isinstance(value, str) else str(value)
            node = self._ensure_node(idx)
            node.text = text_value
            super().__setitem__(idx, text_value)


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

    @property
    def prompt_context(self) -> str:
        return (self.elem.get("name") or "").strip()

    @property
    def prompt_note(self) -> str:
        if self.comment_elem is None or not self.comment_elem.text:
            return ""
        return self.comment_elem.text.strip()


STRINGS_LINE_RE = re.compile(
    r'^(?P<indent>\s*)"(?P<key>(?:\\.|[^"\\])*)"\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"\s*;\s*$'
)
STRINGS_COMMENTED_LINE_RE = re.compile(
    r'^(?P<indent>\s*)/\*\s*"(?P<key>(?:\\.|[^"\\])*)"\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"\s*;\s*\*/\s*$'
)


def _decode_strings_literal(raw: str) -> str:
    try:
        return ast.literal_eval(f'"{raw}"')
    except (SyntaxError, ValueError):
        return raw


def _encode_strings_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\a", "\\a")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\v", "\\v")
    )


def _detect_text_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        head = f.read(4)

    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _normalize_strings_comment_lines(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        text = line.strip()
        if text.startswith("*"):
            text = text[1:].lstrip()
        if text:
            cleaned.append(text)
    return " ".join(cleaned).strip()


class StringsEntryAdapter:
    """Adapts a .strings key/value line to look like a polib entry."""

    def __init__(
        self,
        lines: List[str],
        line_index: int,
        line_ending: str,
        indent: str,
        key: str,
        source_text: str,
        commented: bool,
        prompt_note: str = "",
    ):
        self._lines = lines
        self._line_index = line_index
        self._line_ending = line_ending
        self._indent = indent
        self._key = key
        self._source_text = source_text
        self._commented = commented
        self._prompt_note = prompt_note
        self._flags: List[str] = []
        self.include_in_term_extraction = commented

    @property
    def msgid(self) -> str:
        return self._source_text

    @property
    def msgstr(self) -> str:
        if self._commented:
            return ""
        return self._source_text

    @msgstr.setter
    def msgstr(self, value: str):
        self._source_text = value
        encoded_key = _encode_strings_literal(self._key)
        encoded_value = _encode_strings_literal(value)
        self._lines[self._line_index] = (
            f'{self._indent}"{encoded_key}" = "{encoded_value}";{self._line_ending}'
        )
        self._commented = False
        self.include_in_term_extraction = False

    @property
    def flags(self):
        return self._flags

    @property
    def obsolete(self) -> bool:
        return False

    def translated(self) -> bool:
        # Project convention: commented entries are untranslated source items.
        return not self._commented

    @property
    def prompt_context(self) -> str:
        return self._key

    @property
    def prompt_note(self) -> str:
        return self._prompt_note


class FileKind(str, Enum):
    PO = "po"
    TS = "ts"
    RESX = "resx"
    STRINGS = "strings"
    TXT = "txt"


class EntryStatus(str, Enum):
    UNTRANSLATED = "untranslated"
    FUZZY = "fuzzy"
    TRANSLATED = "translated"
    SKIPPED = "skipped"


@dataclass
class UnifiedEntry:
    """Format-agnostic localization entry used by translation/extraction pipeline."""

    file_kind: FileKind
    msgid: str
    msgid_plural: str = ""
    msgstr: str = ""
    msgstr_plural: Dict[Any, str] = field(default_factory=dict)
    msgctxt: str = ""
    prompt_note_text: str = ""
    occurrences: List[Tuple[str, str]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    obsolete: bool = False
    include_in_term_extraction: bool = True
    status: EntryStatus = EntryStatus.UNTRANSLATED
    _commit_callback: Callable[["UnifiedEntry"], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def prompt_context(self) -> str:
        return self.msgctxt

    @property
    def prompt_note(self) -> str:
        return self.prompt_note_text

    @property
    def string_type(self) -> str:
        return self.file_kind.value

    def translated(self) -> bool:
        return self.status in (EntryStatus.TRANSLATED, EntryStatus.SKIPPED)

    def mark_translated(self) -> None:
        self.status = EntryStatus.TRANSLATED

    def commit(self) -> None:
        if self._commit_callback is not None:
            self._commit_callback(self)


DEFAULT_BATCH_SIZE = 1000
DEFAULT_PARALLEL_REQUESTS = 10
MIN_ITEMS_PER_WORKER = 50
MAX_PROMPT_CONTEXT_CHARS = 180
MAX_PROMPT_NOTE_CHARS = 300
MAX_PROMPT_OCCURRENCES = 3


def detect_file_kind(file_path: str) -> FileKind:
    lower_path = file_path.lower()
    if lower_path.endswith((".po", ".pot")):
        return FileKind.PO
    if lower_path.endswith(".ts"):
        return FileKind.TS
    if lower_path.endswith(".resx"):
        return FileKind.RESX
    if lower_path.endswith(".strings"):
        return FileKind.STRINGS
    if lower_path.endswith(".txt"):
        return FileKind.TXT
    raise ValueError("Unsupported file type. Use .po, .ts, .resx, .strings, or .txt")


# ===========================
# Prompt & translation rules
# ===========================

SYSTEM_INSTRUCTION = """
You are a professional software localization translator.

STRICT RULES:
- Preserve all placeholders EXACTLY (%s, %d, %(name)s, {var}, {{var}})
- Preserve keyboard accelerators/hotkeys EXACTLY (`_`, `&`) and keep them usable in target text
- Preserve HTML/XML tags EXACTLY
- Do NOT reorder placeholders
- Do NOT add or remove content
- Keep original punctuation and capitalization style
- Use consistent terminology
- Translate ONLY the message text
- Avoid unnatural CamelCase in the target language unless source uses intentional branded casing
- If source text is ALL CAPS, keep translation ALL CAPS
- If the input contains 'Singular:' and 'Plural:', provide a natural plural-aware translation for the target language.
"""

TRANSLATION_RESPONSE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "translations": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "id": genai_types.Schema(type=genai_types.Type.STRING),
                    "text": genai_types.Schema(type=genai_types.Type.STRING),
                    "plural_texts": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                },
                required=["id", "text"],
            ),
        ),
    },
    required=["translations"],
)


THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")


def add_thinking_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Gemini thinking level (default: provider/model default)",
    )


def build_thinking_config(thinking_level: str | None) -> genai_types.ThinkingConfig | None:
    if thinking_level is None:
        return None

    normalized = str(thinking_level).strip().lower()
    thinking_level_map = {
        "minimal": genai_types.ThinkingLevel.MINIMAL,
        "low": genai_types.ThinkingLevel.LOW,
        "medium": genai_types.ThinkingLevel.MEDIUM,
        "high": genai_types.ThinkingLevel.HIGH,
    }
    resolved = thinking_level_map.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported thinking level: {thinking_level!r}. "
            f"Expected one of: {', '.join(THINKING_LEVEL_CHOICES)}"
        )

    return genai_types.ThinkingConfig(thinking_level=resolved)


def build_translation_generation_config(
    thinking_level: str | None = None,
) -> genai_types.GenerateContentConfig:
    config_kwargs: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": TRANSLATION_RESPONSE_SCHEMA,
    }
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    return genai_types.GenerateContentConfig(**config_kwargs)


def build_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    force_non_empty: bool = False,
) -> str:
    vocab_block = f"\nProject vocabulary (mandatory):\n{vocabulary}\n" if vocabulary else ""
    rules_block = (
        f"\nProject translation rules/instructions (mandatory when present):\n{translation_rules}\n"
        if translation_rules
        else ""
    )
    non_empty_block = (
        "- Every translation must be non-empty.\n"
        "- Never return empty strings. If uncertain, provide your best translation.\n"
        if force_non_empty
        else ""
    )
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    return f"""
{SYSTEM_INSTRUCTION}

Project context:
This is a software application UI localization project.
Source language: {source_lang}
Target language: {target_lang}
{vocab_block}
{rules_block}

Instructions:
- Translate each message independently
- Do NOT merge messages
- Return ONLY valid JSON, no Markdown fences or extra text
- Keep each translation item's "id" exactly the same as the input key
- Use "plural_texts" only for plural entries when you can provide explicit forms
- For plural entries (source contains 'Singular:'/'Plural:' or item.plural_forms is present),
  return non-empty "plural_texts" with exactly item.plural_forms forms (or at least 2 if absent)
- If the target language effectively has one plural form but multiple slots are required,
  repeat the same translation identically in all required plural slots
- If the target language effectively has one plural form (for example Kazakh), prefer the source plural form as the basis for translation when it carries numeric placeholders or fuller wording than the singular form.
- Do not derive the final wording from a singular-only source variant such as "one" when the plural source contains a numeric placeholder like %d or %n.
- For such entries, keep the plural-source placeholder structure in the translation and use that plural-based wording in every required plural slot.
- Keep "text" non-empty for every item, including plural entries
- Use optional "context" and "note" fields only for disambiguation
- Translate ONLY the "source" field for each item
- Do NOT copy or translate the context/note metadata itself
- Apply project translation rules when provided
- If project rules conflict with STRICT RULES above, STRICT RULES win
- Use consistent translation throughout the messages.
- When the source string have \\n line wrapping marker within the text, try to wrap translated text to lines of similar length with the \\n marker.
- When project vocabulary is supplied, it is mandatory, not advisory.
- Treat each vocabulary pair as a required source_term -> target_term mapping.
- First translate the message, then run a silent vocabulary audit on your own output before returning JSON.
- If a source message contains a vocabulary term, the final translation must use the mapped target term, not a synonym or alternative wording.
- If you used an alternative, rewrite the translation to use the approved term before returning the result.
- Use the approved target term as the lexical choice; inflect it only as grammar requires.
- Return only the corrected final JSON.
{non_empty_block}

Messages to translate (JSON map of id -> item):
- item.source: source text to translate
- item.context: optional disambiguation context
- item.note: optional developer/translator note
- item.plural_forms: optional count of plural forms required in output "plural_texts"
{messages_json}
"""


# ===========================
# Gemini response parsing
# ===========================

@dataclass
class TranslationResult:
    text: str = ""
    plural_texts: List[str] = field(default_factory=list)


def _json_load_maybe(text: str) -> Any:
    payload = text.strip()
    if not payload:
        return None

    # Defensive fallback when a model still wraps JSON in fences.
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _normalize_translation_payload(payload: Any) -> Dict[str, TranslationResult]:
    results: Dict[str, TranslationResult] = {}

    if isinstance(payload, dict):
        if "translations" in payload:
            items = payload.get("translations")
            if not isinstance(items, list):
                return results
            for item in items:
                if not isinstance(item, dict):
                    continue
                msg_id = item.get("id")
                if msg_id is None:
                    continue
                text = item.get("text")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                plural_texts_raw = item.get("plural_texts")
                plural_texts: List[str] = []
                if isinstance(plural_texts_raw, list):
                    for val in plural_texts_raw:
                        if val is None:
                            continue
                        plural_texts.append(val if isinstance(val, str) else str(val))
                results[str(msg_id)] = TranslationResult(text=text, plural_texts=plural_texts)
            return results

        # Backward-compatible fallback for map-style JSON {"0": "text"}.
        for key, val in payload.items():
            if isinstance(val, str):
                results[str(key)] = TranslationResult(text=val)
            elif isinstance(val, dict):
                text = val.get("text")
                if isinstance(text, str):
                    results[str(key)] = TranslationResult(text=text)
        return results

    return results


def parse_response(response_payload: Any) -> Dict[str, TranslationResult]:
    if isinstance(response_payload, (dict, list)):
        return _normalize_translation_payload(response_payload)

    if isinstance(response_payload, str):
        return _normalize_translation_payload(_json_load_maybe(response_payload))

    parsed_payload = getattr(response_payload, "parsed", None)
    if parsed_payload is not None:
        return _normalize_translation_payload(parsed_payload)

    text_payload = getattr(response_payload, "text", None) or ""
    return _normalize_translation_payload(_json_load_maybe(text_payload))


def _is_non_empty_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_model_escaped_text(source_text: str, candidate_text: str) -> str:
    """Align escaped control-char style with source when model returns literal escapes."""
    if not isinstance(candidate_text, str):
        return candidate_text

    normalized = candidate_text
    replacements = (
        ("\n", "\\n"),
        ("\t", "\\t"),
        ("\r", "\\r"),
        ("\a", "\\a"),
        ("\b", "\\b"),
        ("\f", "\\f"),
        ("\v", "\\v"),
    )

    for actual_char, escaped_seq in replacements:
        source_uses_actual = actual_char in source_text and escaped_seq not in source_text
        candidate_uses_literal = escaped_seq in normalized and actual_char not in normalized
        if source_uses_actual and candidate_uses_literal:
            normalized = normalized.replace(escaped_seq, actual_char)

    source_uses_quote = '"' in source_text and '\\"' not in source_text
    candidate_uses_escaped_quote = '\\"' in normalized and '"' not in normalized
    if source_uses_quote and candidate_uses_escaped_quote:
        normalized = normalized.replace('\\"', '"')

    return normalized


def is_plural_entry(entry: Any) -> bool:
    return bool(getattr(entry, "msgid_plural", None))


def get_plural_form_count(entry: Any) -> int:
    if not is_plural_entry(entry):
        return 0

    plural_map = getattr(entry, "msgstr_plural", None)
    if isinstance(plural_map, dict) and plural_map:
        return max(2, len(plural_map))
    return 2


def translation_has_content(result: TranslationResult | None) -> bool:
    if result is None:
        return False
    if _is_non_empty_text(result.text):
        return True
    return any(_is_non_empty_text(t) for t in result.plural_texts)


def _plural_key_sort_key(key: Any) -> Tuple[int, Any]:
    try:
        return 0, int(key)
    except (TypeError, ValueError):
        return 1, str(key)


def apply_translation_to_entry(entry: Any, result: TranslationResult) -> bool:
    source_text = build_entry_source_text(entry)

    if is_plural_entry(entry):
        plural_map = getattr(entry, "msgstr_plural", None)
        if not isinstance(plural_map, dict):
            return False

        plural_keys = sorted(plural_map.keys(), key=_plural_key_sort_key)
        if not plural_keys:
            plural_keys = [0]

        usable_forms = [
            _normalize_model_escaped_text(source_text, s)
            for s in result.plural_texts
            if _is_non_empty_text(s)
        ]
        if usable_forms:
            for idx, key in enumerate(plural_keys):
                plural_map[key] = usable_forms[idx] if idx < len(usable_forms) else usable_forms[-1]
            if hasattr(entry, "mark_translated"):
                entry.mark_translated()
            return True

        if _is_non_empty_text(result.text):
            normalized_text = _normalize_model_escaped_text(source_text, result.text)
            for key in plural_keys:
                plural_map[key] = normalized_text
            if hasattr(entry, "mark_translated"):
                entry.mark_translated()
            return True

        return False

    if not result.text:
        return False

    entry.msgstr = _normalize_model_escaped_text(source_text, result.text)
    if hasattr(entry, "mark_translated"):
        entry.mark_translated()
    return True


def _normalize_prompt_hint(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."

def build_entry_source_text(entry: Any) -> str:
    if is_plural_entry(entry):
        return f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
    return entry.msgid


def get_entry_prompt_context_and_note(entry: Any) -> Tuple[str | None, str | None]:
    context_parts: List[str] = []
    note_parts: List[str] = []

    def add_unique(items: List[str], value: str | None) -> None:
        if value is None:
            return
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in items:
            items.append(cleaned)

    add_unique(context_parts, getattr(entry, "msgctxt", None))
    add_unique(context_parts, getattr(entry, "prompt_context", None))
    add_unique(note_parts, getattr(entry, "prompt_note", None))
    add_unique(note_parts, getattr(entry, "tcomment", None))
    add_unique(note_parts, getattr(entry, "comment", None))

    occurrences = getattr(entry, "occurrences", None)
    if isinstance(occurrences, list) and occurrences:
        formatted: List[str] = []
        for occurrence in occurrences[:MAX_PROMPT_OCCURRENCES]:
            if not isinstance(occurrence, tuple) or not occurrence:
                continue
            file_part = str(occurrence[0]).strip() if occurrence[0] is not None else ""
            if not file_part:
                continue
            line_part = ""
            if len(occurrence) > 1 and occurrence[1] is not None and str(occurrence[1]).strip():
                line_part = str(occurrence[1]).strip()
            formatted.append(f"{file_part}:{line_part}" if line_part else file_part)
        if formatted:
            add_unique(note_parts, f"locations: {', '.join(formatted)}")

    context = _normalize_prompt_hint(" | ".join(context_parts), MAX_PROMPT_CONTEXT_CHARS)
    note = _normalize_prompt_hint(" | ".join(note_parts), MAX_PROMPT_NOTE_CHARS)

    return (context or None), (note or None)


def build_prompt_message_payload(entry: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"source": build_entry_source_text(entry)}
    context, note = get_entry_prompt_context_and_note(entry)
    if is_plural_entry(entry):
        plural_forms = get_plural_form_count(entry)
        payload["plural_forms"] = plural_forms
        plural_note = (
            f"plural forms required: {plural_forms}"
            " | for target languages with no plural difference (for example Kazakh),"
            " prefer the source plural variant as the basis for translation and repeat"
            " that wording in all required plural slots"
        )
        note = f"{note} | {plural_note}" if note else plural_note
    if context:
        payload["context"] = context
    if note:
        payload["note"] = note
    return payload


# ===========================
# Unified Entry Adapters
# ===========================

def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    if isinstance(entry, ResxEntryAdapter) and not getattr(entry, "_translate", True):
        return EntryStatus.SKIPPED
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def _copy_legacy_plural_map(entry: Any) -> Dict[Any, str]:
    plural_map = getattr(entry, "msgstr_plural", None)
    if not isinstance(plural_map, dict):
        return {}
    copied: Dict[Any, str] = {}
    for key, value in plural_map.items():
        copied[key] = value if isinstance(value, str) else str(value)
    return copied


def _copy_legacy_occurrences(entry: Any) -> List[Tuple[str, str]]:
    occurrences = getattr(entry, "occurrences", None)
    if not isinstance(occurrences, list):
        return []
    copied: List[Tuple[str, str]] = []
    for item in occurrences:
        if isinstance(item, tuple) and len(item) >= 1:
            file_part = str(item[0]) if item[0] is not None else ""
            line_part = ""
            if len(item) > 1 and item[1] is not None:
                line_part = str(item[1])
            copied.append((file_part, line_part))
    return copied


def _build_unified_entry(
    entry: Any,
    file_kind: FileKind,
    commit_callback: Callable[[UnifiedEntry], None],
) -> UnifiedEntry:
    raw_msgid = getattr(entry, "msgid", "") or ""
    raw_msgid_plural = getattr(entry, "msgid_plural", "") or ""
    raw_msgstr = getattr(entry, "msgstr", "") or ""
    context = (getattr(entry, "msgctxt", None) or getattr(entry, "prompt_context", None) or "").strip()
    note = (getattr(entry, "prompt_note", None) or "").strip()
    raw_flags = getattr(entry, "flags", []) or []
    flags = [str(flag) for flag in raw_flags]
    obsolete = bool(getattr(entry, "obsolete", False))
    include_terms = bool(getattr(entry, "include_in_term_extraction", True))

    return UnifiedEntry(
        file_kind=file_kind,
        msgid=str(raw_msgid),
        msgid_plural=str(raw_msgid_plural),
        msgstr=str(raw_msgstr),
        msgstr_plural=_copy_legacy_plural_map(entry),
        msgctxt=context,
        prompt_note_text=note,
        occurrences=_copy_legacy_occurrences(entry),
        flags=flags,
        obsolete=obsolete,
        include_in_term_extraction=include_terms,
        status=_entry_status_from_legacy(entry),
        _commit_callback=commit_callback,
    )


def _translation_result_from_unified(entry: UnifiedEntry) -> TranslationResult:
    plural_texts: List[str] = []
    if entry.msgstr_plural:
        for key in sorted(entry.msgstr_plural.keys(), key=_plural_key_sort_key):
            plural_texts.append(entry.msgstr_plural[key])
    return TranslationResult(text=entry.msgstr, plural_texts=plural_texts)


def _commit_unified_to_legacy(entry: UnifiedEntry, legacy_entry: Any) -> None:
    result = _translation_result_from_unified(entry)
    if translation_has_content(result):
        apply_translation_to_entry(legacy_entry, result)

    legacy_flags = getattr(legacy_entry, "flags", None)
    if isinstance(legacy_flags, list):
        for flag in entry.flags:
            if flag not in legacy_flags:
                legacy_flags.append(flag)


def _wrap_legacy_entries(entries: List[Any], file_kind: FileKind) -> List[UnifiedEntry]:
    wrapped: List[UnifiedEntry] = []
    for legacy_entry in entries:
        wrapped.append(
            _build_unified_entry(
                entry=legacy_entry,
                file_kind=file_kind,
                commit_callback=lambda unified, le=legacy_entry: _commit_unified_to_legacy(unified, le),
            )
        )
    return wrapped


# ===========================
# Main logic
# ===========================

async def generate_content_async(
    client: genai.Client,
    model: str,
    prompt: str,
    config: genai_types.GenerateContentConfig | None = None,
) -> Any:
    """Use Gemini async client when available, fallback to thread offload."""
    if config is None:
        config = build_translation_generation_config()
    if hasattr(client, "aio") and client.aio:
        return await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=config,
    )


async def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    batch_label: str,
    max_attempts: int = 5,
    config: genai_types.GenerateContentConfig | None = None,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return await generate_content_async(client, model, prompt, config=config)
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

    legacy_entries: List[Any] = []
    seen_messages: set[int] = set()
    for context_node in root.findall(".//context"):
        name_elem = context_node.find("./name")
        context_name = (name_elem.text or "").strip() if name_elem is not None and name_elem.text else ""
        for message in context_node.findall("./message"):
            legacy_entries.append(TSEntryAdapter(message, context_name=context_name))
            seen_messages.add(id(message))

    for message in root.findall(".//message"):
        if id(message) in seen_messages:
            continue
        legacy_entries.append(TSEntryAdapter(message))

    entries = _wrap_legacy_entries(legacy_entries, FileKind.TS)

    output_path = build_output_path(file_path, FileKind.TS)

    def save_ts() -> None:
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_ts, output_path


def load_resx(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing RESX file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    legacy_entries: List[Any] = []
    for data_node in root.findall("./data"):
        legacy_entries.append(ResxEntryAdapter(data_node))

    entries = _wrap_legacy_entries(legacy_entries, FileKind.RESX)

    output_path = build_output_path(file_path, FileKind.RESX)

    def save_resx() -> None:
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_resx, output_path


def load_po(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing PO file: {file_path}")
    po = polib.pofile(file_path, wrapwidth=PO_WRAP_WIDTH)
    legacy_entries = [e for e in po]
    entries = _wrap_legacy_entries(legacy_entries, FileKind.PO)
    output_path = build_output_path(file_path, FileKind.PO)

    def save_po() -> None:
        for entry in entries:
            entry.commit()
        po.save(output_path)

    return entries, save_po, output_path


def load_strings(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing STRINGS file: {file_path}")
    encoding = _detect_text_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    legacy_entries: List[Any] = []
    pending_notes: List[str] = []
    in_block_comment = False
    block_comment_lines: List[str] = []

    def flush_block_comment() -> None:
        if not block_comment_lines:
            return
        normalized = _normalize_strings_comment_lines(block_comment_lines)
        if normalized:
            pending_notes.append(normalized)
        block_comment_lines.clear()

    for idx, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        line_ending = line[len(stripped):]
        lstripped = stripped.lstrip()

        if in_block_comment:
            before, sep, _ = stripped.partition("*/")
            block_comment_lines.append(before)
            if sep:
                in_block_comment = False
                flush_block_comment()
            continue

        match = STRINGS_COMMENTED_LINE_RE.match(stripped)
        commented = True
        if not match:
            match = STRINGS_LINE_RE.match(stripped)
            commented = False
        if match:
            key = _decode_strings_literal(match.group("key"))
            value = _decode_strings_literal(match.group("value"))
            legacy_entries.append(
                StringsEntryAdapter(
                    lines=lines,
                    line_index=idx,
                    line_ending=line_ending,
                    indent=match.group("indent"),
                    key=key,
                    source_text=value,
                    commented=commented,
                    prompt_note=" | ".join(pending_notes),
                )
            )
            pending_notes = []
            continue

        if not lstripped:
            continue

        if lstripped.startswith("//"):
            inline = lstripped[2:].strip()
            if inline:
                pending_notes.append(inline)
            continue

        if "/*" in lstripped:
            _, _, after_open = stripped.partition("/*")
            before_close, sep, _ = after_open.partition("*/")
            block_comment_lines.append(before_close)
            if sep:
                flush_block_comment()
            else:
                in_block_comment = True
            continue

        pending_notes = []

    if in_block_comment:
        flush_block_comment()

    entries = _wrap_legacy_entries(legacy_entries, FileKind.STRINGS)

    output_path = build_output_path(file_path, FileKind.STRINGS)

    def save_strings() -> None:
        for entry in entries:
            entry.commit()
        with open(output_path, "w", encoding=encoding, newline="") as f:
            f.write("".join(lines))

    return entries, save_strings, output_path


def load_txt(file_path: str) -> Tuple[List[Any], Callable[[], None], str]:
    print(f"Processing TXT file: {file_path}")
    encoding = _detect_text_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    entries: List[UnifiedEntry] = []

    for line_index, raw_line in enumerate(lines):
        text = raw_line.rstrip("\r\n")
        line_ending = raw_line[len(text):]
        line_number = line_index + 1
        is_translatable = bool(text.strip())
        status = EntryStatus.UNTRANSLATED if is_translatable else EntryStatus.SKIPPED

        def commit_line(entry: UnifiedEntry, idx: int = line_index, ending: str = line_ending) -> None:
            # Keep source text for entries not translated yet during incremental saves.
            value = entry.msgid
            if entry.status in (EntryStatus.FUZZY, EntryStatus.TRANSLATED) and _is_non_empty_text(entry.msgstr):
                value = entry.msgstr
            lines[idx] = f"{value}{ending}"

        entries.append(
            UnifiedEntry(
                file_kind=FileKind.TXT,
                msgid=text,
                msgstr="",
                msgctxt=f"line:{line_number}",
                flags=[],
                obsolete=False,
                include_in_term_extraction=is_translatable,
                status=status,
                _commit_callback=commit_line,
            )
        )

    output_path = build_output_path(file_path, FileKind.TXT)

    def save_txt() -> None:
        for entry in entries:
            entry.commit()
        with open(output_path, "w", encoding=encoding, newline="") as f:
            f.write("".join(lines))

    return entries, save_txt, output_path


def build_output_path(file_path: str, file_kind: FileKind) -> str:
    root, _ = os.path.splitext(file_path)
    return f"{root}.ai-translated.{file_kind.value}"


def read_optional_text_file(path: str | None, label: str) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return None

    if not content:
        print(f"Warning: {label} file '{path}' is empty.")
        return None
    return content


def _normalize_vocabulary_cell(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _is_translated_vocabulary_entry(entry: Any) -> bool:
    translated_attr = getattr(entry, "translated", None)
    if callable(translated_attr):
        try:
            return bool(translated_attr())
        except TypeError:
            return False
    return False


def parse_vocabulary_line(line: str) -> Tuple[str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#"):
        return None

    separator = " - "
    if separator not in stripped:
        return None

    source_term, target_term = stripped.split(separator, 1)
    source_term = _normalize_vocabulary_cell(source_term)
    target_term = _normalize_vocabulary_cell(target_term)
    if not source_term or not target_term:
        return None
    return source_term, target_term


def load_vocabulary_pairs(path: str | None, label: str = "Vocabulary") -> List[Tuple[str, str]]:
    if not path:
        return []

    if not path.lower().endswith(".po"):
        content = read_optional_text_file(path, label)
        if not content:
            return []

        pairs: List[Tuple[str, str]] = []
        seen_indices: Dict[str, int] = {}
        for raw_line in content.splitlines():
            parsed = parse_vocabulary_line(raw_line)
            if not parsed:
                continue
            source_term, target_term = parsed
            key = source_term.lower()
            if key in seen_indices:
                pairs[seen_indices[key]] = (source_term, target_term)
                continue
            seen_indices[key] = len(pairs)
            pairs.append((source_term, target_term))
        return pairs

    try:
        glossary = polib.pofile(path, wrapwidth=PO_WRAP_WIDTH)
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return []

    pairs: List[Tuple[str, str]] = []
    for entry in glossary:
        if not _is_translated_vocabulary_entry(entry):
            continue
        source_term = _normalize_vocabulary_cell(getattr(entry, "msgid", ""))
        target_term = _normalize_vocabulary_cell(getattr(entry, "msgstr", ""))
        if not source_term or not target_term:
            continue
        pairs.append((source_term, target_term))
    return pairs


def read_optional_vocabulary_file(path: str | None, label: str = "Vocabulary") -> str | None:
    pairs = load_vocabulary_pairs(path, label)
    if not pairs:
        if path and path.lower().endswith(".po"):
            print(f"Warning: {label} file '{path}' has no usable msgid/msgstr glossary pairs.")
        return None
    return "\n".join(f"{source_term} - {target_term}" for source_term, target_term in pairs)


def build_language_code_candidates(target_lang: str) -> List[str]:
    raw = (target_lang or "").strip()
    if not raw:
        return []

    seeds: List[str] = []
    for candidate in (raw, raw.replace("-", "_"), raw.replace("_", "-")):
        if candidate and candidate not in seeds:
            seeds.append(candidate)

    results: List[str] = []

    def add(code: str | None) -> None:
        if code and code not in results:
            results.append(code)

    for seed in seeds:
        add(seed)
        if "_" in seed:
            add(seed.split("_", 1)[0])
        if "-" in seed:
            add(seed.split("-", 1)[0])

    for value in list(results):
        lower = value.lower()
        add(lower)
        if "_" in lower:
            add(lower.split("_", 1)[0])
        if "-" in lower:
            add(lower.split("-", 1)[0])

    return results


def detect_default_text_resource(prefix: str, extension: str, target_lang: str) -> str | None:
    for lang_code in build_language_code_candidates(target_lang):
        candidate_path = os.path.join("data", lang_code, f"{prefix}.{extension}")
        if os.path.isfile(candidate_path):
            return candidate_path

    # Backward compatibility with flat naming at repository root.
    for lang_code in build_language_code_candidates(target_lang):
        legacy_path = f"{prefix}-{lang_code}.{extension}"
        if os.path.isfile(legacy_path):
            return legacy_path
    return None


def resolve_resource_path(
    explicit_path: str | None,
    prefix: str,
    extension: str,
    target_lang: str,
) -> str | None:
    if explicit_path:
        return explicit_path
    return detect_default_text_resource(prefix, extension, target_lang)


def merge_project_rules(file_rules: str | None, inline_rules: str | None) -> str | None:
    parts: List[str] = []
    if file_rules:
        parts.append(file_rules.strip())
    if inline_rules and inline_rules.strip():
        parts.append(inline_rules.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def detect_rules_source(
    rules_path: str | None,
    file_rules: str | None,
    inline_rules: str | None,
) -> str | None:
    sources: List[str] = []
    if file_rules and rules_path:
        sources.append(f"file:{rules_path}")
    if inline_rules and inline_rules.strip():
        sources.append("inline:--rules-str")

    if not sources:
        return None
    return ", ".join(sources)


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


def _entry_has_source_text(entry: Any) -> bool:
    return bool(str(getattr(entry, "msgid", "") or "").strip())


def select_work_items(entries: List[Any], retranslate_all: bool = False) -> List[Any]:
    selected: List[Any] = []
    for entry in entries:
        if bool(getattr(entry, "obsolete", False)):
            continue

        status = getattr(entry, "status", None)
        if status == EntryStatus.SKIPPED:
            continue

        if retranslate_all:
            if _entry_has_source_text(entry):
                selected.append(entry)
            continue

        if not entry.translated():
            selected.append(entry)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process and translate PO, TS, RESX, STRINGS, or TXT files using Google Gemini"
    )
    parser.add_argument("file", help="Input .po, .ts, .resx, .strings, or .txt file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    add_thinking_level_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent Gemini requests (auto if omitted)")
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt). Supports .txt and glossary .po",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional translation rules/instructions file (auto: data/<target-lang>/rules.md)",
    )
    parser.add_argument("--rules-str", default=None, help="Optional inline translation rules/instructions")
    parser.add_argument(
        "--retranslate-all",
        action="store_true",
        help="Force translation of all translatable messages, not only unfinished/fuzzy ones",
    )

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    vocabulary_path = resolve_resource_path(
        explicit_path=args.vocab,
        prefix="vocab",
        extension="txt",
        target_lang=args.target_lang,
    )
    rules_path = resolve_resource_path(
        explicit_path=args.rules,
        prefix="rules",
        extension="md",
        target_lang=args.target_lang,
    )

    vocabulary_text = read_optional_vocabulary_file(vocabulary_path, "Vocabulary")
    rules_text = read_optional_text_file(rules_path, "Rules")
    project_rules = merge_project_rules(rules_text, args.rules_str)
    rules_source = detect_rules_source(rules_path, rules_text, args.rules_str)
    vocabulary_source = f"file:{vocabulary_path}" if vocabulary_text and vocabulary_path else "none"

    file_path = args.file
    try:
        file_kind = detect_file_kind(file_path)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    if file_kind == FileKind.TS:
        entries, save_callback, output_path = load_ts(file_path)
    elif file_kind == FileKind.RESX:
        entries, save_callback, output_path = load_resx(file_path)
    elif file_kind == FileKind.STRINGS:
        entries, save_callback, output_path = load_strings(file_path)
    elif file_kind == FileKind.TXT:
        entries, save_callback, output_path = load_txt(file_path)
    else:
        entries, save_callback, output_path = load_po(file_path)

    work_items = select_work_items(entries, retranslate_all=args.retranslate_all)

    total = len(work_items)
    if total == 0:
        if args.retranslate_all:
            print("No translatable messages found.")
        else:
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

    translation_config = build_translation_generation_config(args.thinking_level)

    print("Startup configuration:")
    print(f"  Model: {args.model}")
    print(f"  Thinking level: {args.thinking_level or 'provider default'}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")
    print(f"  Retranslate all: {'yes' if args.retranslate_all else 'no'}")
    print(f"  Vocabulary source: {vocabulary_source}")
    print(f"  Rules source: {rules_source or 'none'}")

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
            msg_map: Dict[str, Dict[str, Any]] = {}
            for i, entry in enumerate(batch):
                msg_map[str(i)] = build_prompt_message_payload(entry)

            prompt = build_prompt(
                msg_map,
                args.source_lang,
                args.target_lang,
                vocabulary_text,
                project_rules,
            )

            response = await generate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                batch_label=f"batch {batch_index + 1}/{batches}",
                max_attempts=5,
                config=translation_config,
            )
            translations = parse_response(response)

            missing_indices = [
                i
                for i in range(len(batch))
                if not translation_has_content(translations.get(str(i)))
            ]
            if missing_indices:
                print(
                    f"  Warning [batch {batch_index + 1}/{batches}]: "
                    f"{len(missing_indices)} items missing from response. Retrying them..."
                )
                retry_map: Dict[str, Dict[str, Any]] = {}
                for idx in missing_indices:
                    entry = batch[idx]
                    retry_map[str(idx)] = build_prompt_message_payload(entry)

                retry_prompt = build_prompt(
                    retry_map,
                    args.source_lang,
                    args.target_lang,
                    vocabulary_text,
                    project_rules,
                    force_non_empty=True,
                )

                try:
                    retry_resp = await generate_with_retry(
                        client=client,
                        model=args.model,
                        prompt=retry_prompt,
                        batch_label=f"batch {batch_index + 1}/{batches} missing-items",
                        max_attempts=3,
                        config=translation_config,
                    )
                    retry_translations = parse_response(retry_resp)
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
                result = translations.get(key)
                if result and apply_translation_to_entry(entry, result):
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

