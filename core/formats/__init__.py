from __future__ import annotations

from core.entries import apply_translation_to_entry, build_entry_source_text, get_entry_prompt_context_and_note, plural_key_sort_key
from core.formats.android_xml import load_android_xml, load_paired_android_xml
from core.formats.base import (
    DEFAULT_TRANSLATION_SCOPE,
    EntryStatus,
    FileKind,
    TRANSLATION_SCOPE_ALL,
    TRANSLATION_SCOPE_CHOICES,
    TRANSLATION_SCOPE_UNFINISHED,
    TRANSLATION_SCOPE_UNTRANSLATED,
    UnifiedEntry,
    build_output_path,
    detect_file_kind,
    select_work_items,
)
from core.formats.po import PO_WRAP_WIDTH, load_po
from core.formats.resx import ResxEntryAdapter, load_resx
from core.formats.strings import StringsEntryAdapter, load_strings
from core.formats.ts import TSEntryAdapter, load_ts
from core.formats.txt import load_txt
from core.formats.xliff import load_xliff

__all__ = [
    "PO_WRAP_WIDTH",
    "TSEntryAdapter",
    "ResxEntryAdapter",
    "StringsEntryAdapter",
    "FileKind",
    "EntryStatus",
    "DEFAULT_TRANSLATION_SCOPE",
    "UnifiedEntry",
    "apply_translation_to_entry",
    "build_entry_source_text",
    "build_output_path",
    "detect_file_kind",
    "get_entry_prompt_context_and_note",
    "load_android_xml",
    "load_po",
    "load_paired_android_xml",
    "load_resx",
    "load_strings",
    "load_ts",
    "load_txt",
    "load_xliff",
    "plural_key_sort_key",
    "select_work_items",
    "TRANSLATION_SCOPE_ALL",
    "TRANSLATION_SCOPE_CHOICES",
    "TRANSLATION_SCOPE_UNFINISHED",
    "TRANSLATION_SCOPE_UNTRANSLATED",
]
