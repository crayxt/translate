from __future__ import annotations

from core.entries import apply_translation_to_entry, build_entry_source_text, get_entry_prompt_context_and_note, plural_key_sort_key
from core.formats.base import (
    EntryStatus,
    FileKind,
    UnifiedEntry,
    _build_unified_entry,
    build_output_path,
    detect_file_kind,
    select_work_items,
)
from core.formats.po import PO_WRAP_WIDTH, load_po
from core.formats.resx import ResxEntryAdapter, load_resx
from core.formats.strings import StringsEntryAdapter, _detect_text_encoding, _write_text_with_encoding_fallback, load_strings
from core.formats.ts import TSEntryAdapter, load_ts
from core.formats.txt import load_txt

__all__ = [
    "PO_WRAP_WIDTH",
    "TSEntryAdapter",
    "ResxEntryAdapter",
    "StringsEntryAdapter",
    "FileKind",
    "EntryStatus",
    "UnifiedEntry",
    "apply_translation_to_entry",
    "build_entry_source_text",
    "build_output_path",
    "detect_file_kind",
    "get_entry_prompt_context_and_note",
    "load_po",
    "load_resx",
    "load_strings",
    "load_ts",
    "load_txt",
    "plural_key_sort_key",
    "select_work_items",
    "_build_unified_entry",
    "_detect_text_encoding",
    "_write_text_with_encoding_fallback",
]
