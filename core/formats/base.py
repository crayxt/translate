from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from core.entries import (
    PluralKey,
    TranslationResult,
    apply_translation_to_entry,
    translation_has_content,
)


class FileKind(str, Enum):
    PO = "po"
    TS = "ts"
    RESX = "resx"
    STRINGS = "strings"
    TXT = "txt"
    XLIFF = "xliff"
    ANDROID_XML = "xml"


class EntryStatus(str, Enum):
    UNTRANSLATED = "untranslated"
    FUZZY = "fuzzy"
    TRANSLATED = "translated"
    SKIPPED = "skipped"


@dataclass
class UnifiedEntry:
    file_kind: FileKind
    msgid: str
    msgid_plural: str = ""
    msgstr: str = ""
    msgstr_plural: Dict[PluralKey, str] = field(default_factory=dict)
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


def detect_file_kind(file_path: str) -> FileKind:
    lower_path = file_path.lower()
    if lower_path.endswith((".po", ".pot")):
        return FileKind.PO
    if lower_path.endswith((".xlf", ".xliff")):
        return FileKind.XLIFF
    if lower_path.endswith(".ts"):
        return FileKind.TS
    if lower_path.endswith(".resx"):
        return FileKind.RESX
    if lower_path.endswith(".strings"):
        return FileKind.STRINGS
    if lower_path.endswith(".txt"):
        return FileKind.TXT
    if lower_path.endswith(".xml"):
        from core.formats.android_xml import is_android_resources_xml

        if is_android_resources_xml(file_path):
            return FileKind.ANDROID_XML
    raise ValueError("Unsupported file type. Use .po, .xlf/.xliff, .ts, .resx, .strings, .txt, or Android .xml")


def build_output_path(file_path: str, file_kind: FileKind) -> str:
    root, _ = os.path.splitext(file_path)
    if file_kind == FileKind.XLIFF:
        extension = os.path.splitext(file_path)[1] or ".xliff"
        return f"{root}.ai-translated{extension}"
    return f"{root}.ai-translated.{file_kind.value}"


def _copy_legacy_plural_map(entry: Any) -> Dict[PluralKey, str]:
    plural_map = getattr(entry, "msgstr_plural", None)
    if not isinstance(plural_map, dict):
        return {}
    copied: Dict[PluralKey, str] = {}
    for key, value in plural_map.items():
        normalized_key: PluralKey = key if isinstance(key, (int, str)) else str(key)
        copied[normalized_key] = value if isinstance(value, str) else str(value)
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


def _translation_result_from_unified(entry: UnifiedEntry, plural_key_sort_key) -> TranslationResult:
    plural_texts: List[str] = []
    if entry.msgstr_plural:
        for key in sorted(entry.msgstr_plural.keys(), key=plural_key_sort_key):
            plural_texts.append(entry.msgstr_plural[key])
    return TranslationResult(text=entry.msgstr, plural_texts=plural_texts)


def _commit_unified_to_legacy(entry: UnifiedEntry, legacy_entry: Any, plural_key_sort_key) -> None:
    result = _translation_result_from_unified(entry, plural_key_sort_key)
    if translation_has_content(result):
        apply_translation_to_entry(legacy_entry, result)
    legacy_flags = getattr(legacy_entry, "flags", None)
    if isinstance(legacy_flags, list):
        for flag in entry.flags:
            if flag not in legacy_flags:
                legacy_flags.append(flag)


def _build_unified_entry(
    entry: Any,
    file_kind: FileKind,
    status_getter,
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
        status=status_getter(entry),
        _commit_callback=commit_callback,
    )


def wrap_legacy_entries(
    entries: List[Any],
    file_kind: FileKind,
    status_getter,
    plural_key_sort_key,
) -> List[UnifiedEntry]:
    wrapped: List[UnifiedEntry] = []
    for legacy_entry in entries:
        wrapped.append(
            _build_unified_entry(
                entry=legacy_entry,
                file_kind=file_kind,
                status_getter=status_getter,
                commit_callback=lambda unified, le=legacy_entry: _commit_unified_to_legacy(
                    unified, le, plural_key_sort_key
                ),
            )
        )
    return wrapped


def _entry_has_source_text(entry: Any) -> bool:
    return bool(str(getattr(entry, "msgid", "") or "").strip())


def select_work_items(entries: List[Any], retranslate_all: bool = False) -> List[Any]:
    selected: List[Any] = []
    for entry in entries:
        if bool(getattr(entry, "obsolete", False)):
            continue
        if getattr(entry, "status", None) == EntryStatus.SKIPPED:
            continue
        if retranslate_all:
            if _entry_has_source_text(entry):
                selected.append(entry)
            continue
        if not entry.translated():
            selected.append(entry)
    return selected
