from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from core.formats import (
    EntryStatus,
    FileKind,
    UnifiedEntry,
    build_entry_source_text,
    detect_file_kind,
    get_entry_prompt_context_and_note,
    load_paired_android_xml,
    load_po,
    load_resx,
    load_strings,
    load_ts,
    load_xliff,
)
from core.formats.strings import _detect_text_encoding, _write_text_with_encoding_fallback
from core.review_common import plural_key_sort_key
from core.review_flow import has_reviewable_translation as has_shared_reviewable_translation


@dataclass(slots=True)
class ReviewItem:
    """Revision-ready view of one translated entry plus its review context."""

    entry: UnifiedEntry
    source_text: str
    current_text: str
    current_plural_texts: List[str]
    plural_form_count: int
    context: str = ""
    note: str = ""
    pair_key: str = ""


@dataclass(slots=True)
class ReviewBundle:
    """Loaded file state and review items for one revision run."""

    file_kind: FileKind
    entries: List[UnifiedEntry]
    save_callback: Callable[[], None]
    generated_output_path: str
    items: List[ReviewItem]
    warnings: List[str] = field(default_factory=list)


def build_revision_output_path(file_path: str) -> str:
    """Build the default output path for a revised file."""
    root, ext = os.path.splitext(file_path)
    return f"{root}-revised{ext}"


def _clean_text(text: str | None) -> str:
    """Trim a possibly missing string value."""
    return str(text or "").strip()


def get_plural_texts(entry: UnifiedEntry) -> List[str]:
    """Return existing plural translations in stable plural-slot order."""
    if not entry.msgstr_plural:
        return []
    return [
        str(entry.msgstr_plural[key] or "")
        for key in sorted(entry.msgstr_plural.keys(), key=plural_key_sort_key)
    ]


def get_plural_form_count(entry: UnifiedEntry) -> int:
    """Estimate how many plural slots a revised entry should fill."""
    if not entry.msgid_plural:
        return 0
    if entry.msgstr_plural:
        return len(entry.msgstr_plural)
    return 2


def has_reviewable_translation(entry: UnifiedEntry) -> bool:
    """Return whether an entry has translated content worth revising."""
    return has_shared_reviewable_translation(
        entry,
        plural_texts=get_plural_texts(entry),
        allow_context_only=True,
    )


def _join_non_empty(parts: List[str]) -> str:
    """Join unique non-empty note fragments with a stable separator."""
    seen: set[str] = set()
    result: List[str] = []
    for value in parts:
        cleaned = _clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return " | ".join(result)


def build_review_item(entry: UnifiedEntry) -> ReviewItem:
    """Project a translated entry into the normalized revision item shape."""
    context, note = get_entry_prompt_context_and_note(entry)
    current_plural_texts = get_plural_texts(entry)
    current_text = current_plural_texts[0] if current_plural_texts else str(entry.msgstr or "")
    pair_key = _clean_text(context) or _clean_text(entry.msgctxt) or _clean_text(entry.msgid)
    return ReviewItem(
        entry=entry,
        source_text=build_entry_source_text(entry),
        current_text=current_text,
        current_plural_texts=current_plural_texts,
        plural_form_count=get_plural_form_count(entry),
        context=context or "",
        note=note or "",
        pair_key=pair_key,
    )


def _load_file_entries(file_path: str, file_kind: FileKind) -> Tuple[List[UnifiedEntry], Callable[[], None], str]:
    """Load one revisable file kind and return entries plus save metadata."""
    if file_kind == FileKind.PO:
        return load_po(file_path)
    if file_kind == FileKind.XLIFF:
        return load_xliff(file_path)
    if file_kind == FileKind.TS:
        return load_ts(file_path)
    if file_kind == FileKind.RESX:
        return load_resx(file_path)
    if file_kind == FileKind.STRINGS:
        return load_strings(file_path)
    raise ValueError(
        f"Unsupported single-file revision format: {file_kind.value}. "
        "Use a paired workflow for .txt files."
    )


def build_single_file_bundle(file_path: str, file_kind: FileKind) -> ReviewBundle:
    """Build a review bundle for formats that embed their own source text."""
    entries, save_callback, generated_output_path = _load_file_entries(file_path, file_kind)
    items = [build_review_item(entry) for entry in entries if has_reviewable_translation(entry)]
    return ReviewBundle(
        file_kind=file_kind,
        entries=entries,
        save_callback=save_callback,
        generated_output_path=generated_output_path,
        items=items,
    )


def _build_pair_key(entry: UnifiedEntry, index: int) -> str:
    """Build a stable pairing key for source/translated entry alignment."""
    context = _clean_text(entry.msgctxt)
    if context:
        return context
    return f"index:{index}"


def build_paired_bundle(
    source_file: str,
    translated_file: str,
    file_kind: FileKind,
) -> ReviewBundle:
    """Build a review bundle by pairing translated entries with a source file."""
    source_entries, _, _ = _load_file_entries(source_file, file_kind)
    translated_entries, save_callback, generated_output_path = _load_file_entries(translated_file, file_kind)

    buckets: Dict[str, List[UnifiedEntry]] = {}
    for index, source_entry in enumerate(source_entries):
        key = _build_pair_key(source_entry, index)
        buckets.setdefault(key, []).append(source_entry)

    items: List[ReviewItem] = []
    warnings: List[str] = []
    missing_pairs = 0

    for index, translated_entry in enumerate(translated_entries):
        if not has_reviewable_translation(translated_entry):
            continue

        key = _build_pair_key(translated_entry, index)
        source_bucket = buckets.get(key) or []
        if not source_bucket:
            missing_pairs += 1
            continue

        source_entry = source_bucket.pop(0)
        translated_context, translated_note = get_entry_prompt_context_and_note(translated_entry)
        source_context, source_note = get_entry_prompt_context_and_note(source_entry)
        current_plural_texts = get_plural_texts(translated_entry)
        current_text = current_plural_texts[0] if current_plural_texts else str(translated_entry.msgstr or "")

        items.append(
            ReviewItem(
                entry=translated_entry,
                source_text=str(source_entry.msgid or ""),
                current_text=current_text,
                current_plural_texts=current_plural_texts,
                plural_form_count=0,
                context=_clean_text(translated_context or source_context),
                note=_join_non_empty([source_note or "", translated_note or ""]),
                pair_key=key,
            )
        )

    if missing_pairs:
        warnings.append(
            f"Skipped {missing_pairs} translated entries with no matching source pair in {source_file}."
        )

    return ReviewBundle(
        file_kind=file_kind,
        entries=translated_entries,
        save_callback=save_callback,
        generated_output_path=generated_output_path,
        items=items,
        warnings=warnings,
    )


def load_paired_txt_bundle(source_file: str, translated_file: str) -> ReviewBundle:
    """Build a paired revision bundle for line-oriented text files."""
    source_encoding = _detect_text_encoding(source_file)
    translated_encoding = _detect_text_encoding(translated_file)

    with open(source_file, "r", encoding=source_encoding) as handle:
        source_lines = handle.read().splitlines(keepends=True)
    with open(translated_file, "r", encoding=translated_encoding) as handle:
        translated_lines = handle.read().splitlines(keepends=True)

    if len(source_lines) != len(translated_lines):
        raise ValueError(
            "Paired .txt workflow requires the source and translated files to have the same number of lines."
        )

    entries: List[UnifiedEntry] = []
    items: List[ReviewItem] = []

    for index, (source_raw, translated_raw) in enumerate(zip(source_lines, translated_lines)):
        source_text = source_raw.rstrip("\r\n")
        translated_text = translated_raw.rstrip("\r\n")
        translated_line_ending = translated_raw[len(translated_text):]
        line_number = index + 1

        if not _clean_text(source_text):
            status = EntryStatus.SKIPPED
        elif _clean_text(translated_text):
            status = EntryStatus.TRANSLATED
        else:
            status = EntryStatus.UNTRANSLATED

        def commit_line(
            entry: UnifiedEntry,
            idx: int = index,
            ending: str = translated_line_ending,
        ) -> None:
            translated_lines[idx] = f"{entry.msgstr}{ending}"

        entry = UnifiedEntry(
            file_kind=FileKind.TXT,
            msgid=source_text,
            msgstr=translated_text,
            msgctxt=f"line:{line_number}",
            flags=[],
            obsolete=False,
            include_in_term_extraction=bool(_clean_text(source_text)),
            status=status,
            _commit_callback=commit_line,
        )
        entries.append(entry)

        if not has_reviewable_translation(entry):
            continue

        items.append(
            ReviewItem(
                entry=entry,
                source_text=source_text,
                current_text=translated_text,
                current_plural_texts=[],
                plural_form_count=0,
                context=f"line:{line_number}",
                note="",
                pair_key=f"line:{line_number}",
            )
        )

    generated_output_path = build_revision_output_path(translated_file)

    def save_txt() -> None:
        for entry in entries:
            entry.commit()
        _write_text_with_encoding_fallback(
            generated_output_path,
            "".join(translated_lines),
            translated_encoding,
            newline="",
        )

    return ReviewBundle(
        file_kind=FileKind.TXT,
        entries=entries,
        save_callback=save_txt,
        generated_output_path=generated_output_path,
        items=items,
    )


def load_review_bundle(
    translated_file: str,
    source_file: str | None = None,
) -> ReviewBundle:
    """Load the appropriate revision bundle for the translated file and optional source file."""
    file_kind = detect_file_kind(translated_file)

    if source_file:
        source_kind = detect_file_kind(source_file)
        if source_kind != file_kind:
            raise ValueError(
                f"--source-file type mismatch: expected .{file_kind.value}, got .{source_kind.value}"
            )

    if file_kind in (FileKind.PO, FileKind.XLIFF, FileKind.TS):
        return build_single_file_bundle(translated_file, file_kind)

    if file_kind == FileKind.ANDROID_XML:
        if not source_file:
            raise ValueError(
                "--source-file is required for .xml revision runs because the translated file "
                "does not retain the original source text."
            )
        entries, save_callback, generated_output_path, warnings = load_paired_android_xml(
            source_file,
            translated_file,
        )
        items = [build_review_item(entry) for entry in entries if has_reviewable_translation(entry)]
        return ReviewBundle(
            file_kind=file_kind,
            entries=entries,
            save_callback=save_callback,
            generated_output_path=generated_output_path,
            items=items,
            warnings=warnings,
        )

    if file_kind == FileKind.TXT:
        if not source_file:
            raise ValueError("--source-file is required for .txt revision runs.")
        return load_paired_txt_bundle(source_file, translated_file)

    if file_kind in (FileKind.STRINGS, FileKind.RESX):
        if not source_file:
            raise ValueError(
                f"--source-file is required for .{file_kind.value} revision runs because "
                "the translated file does not retain the original source text."
            )
        return build_paired_bundle(source_file, translated_file, file_kind)

    raise ValueError(
        "Unsupported file type. Use .po, .xlf/.xliff, .ts, .resx, .strings, .txt, or Android .xml"
    )
