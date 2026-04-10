from __future__ import annotations

import ast
import re
from typing import Any, Callable, List, Tuple

from core.entries import is_non_empty_text, plural_key_sort_key
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path, wrap_legacy_entries

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
    with open(file_path, "rb") as handle:
        head = handle.read(4)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _write_text_with_encoding_fallback(
    output_path: str,
    content: str,
    preferred_encoding: str,
    *,
    newline: str | None = None,
) -> str:
    try:
        with open(output_path, "w", encoding=preferred_encoding, newline=newline) as handle:
            handle.write(content)
        return preferred_encoding
    except UnicodeEncodeError:
        fallback_encoding = "utf-8-sig"
        with open(output_path, "w", encoding=fallback_encoding, newline=newline) as handle:
            handle.write(content)
        print(
            f"Warning: could not write '{output_path}' using {preferred_encoding}; "
            f"saved as {fallback_encoding} instead."
        )
        return fallback_encoding


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
        return not self._commented

    @property
    def prompt_context(self) -> str:
        return self._key

    @property
    def prompt_note(self) -> str:
        return self._prompt_note


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def load_strings(file_path: str) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
    print(f"Processing STRINGS file: {file_path}")
    encoding = _detect_text_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as handle:
        content = handle.read()

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

    entries = wrap_legacy_entries(legacy_entries, FileKind.STRINGS, _entry_status_from_legacy, plural_key_sort_key)
    output_path = build_output_path(file_path, FileKind.STRINGS)

    def save_strings() -> None:
        for entry in entries:
            entry.commit()
        _write_text_with_encoding_fallback(output_path, "".join(lines), encoding, newline="")

    return entries, save_strings, output_path
