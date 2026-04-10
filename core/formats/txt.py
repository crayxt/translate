from __future__ import annotations

from typing import Callable, List, Tuple

from core.entries import is_non_empty_text
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path
from core.formats.strings import _detect_text_encoding, _write_text_with_encoding_fallback


def load_txt(file_path: str) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
    print(f"Processing TXT file: {file_path}")
    encoding = _detect_text_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as handle:
        content = handle.read()

    lines = content.splitlines(keepends=True)
    entries: List[UnifiedEntry] = []

    for line_index, raw_line in enumerate(lines):
        text = raw_line.rstrip("\r\n")
        line_ending = raw_line[len(text):]
        line_number = line_index + 1
        is_translatable = bool(text.strip())
        status = EntryStatus.UNTRANSLATED if is_translatable else EntryStatus.SKIPPED

        def commit_line(entry: UnifiedEntry, idx: int = line_index, ending: str = line_ending) -> None:
            value = entry.msgid
            if entry.status in (EntryStatus.FUZZY, EntryStatus.TRANSLATED) and is_non_empty_text(entry.msgstr):
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
        _write_text_with_encoding_fallback(output_path, "".join(lines), encoding, newline="")

    return entries, save_txt, output_path
