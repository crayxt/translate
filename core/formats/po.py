from __future__ import annotations

from typing import Any, Callable, Tuple

import polib

from core.entries import plural_key_sort_key
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path, wrap_legacy_entries

PO_WRAP_WIDTH = 78


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def load_po(
    file_path: str,
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
    print(f"Processing PO file: {file_path}")
    loader = pofile_loader or polib.pofile
    po = loader(file_path, wrapwidth=PO_WRAP_WIDTH)
    legacy_entries = [entry for entry in po]
    entries = wrap_legacy_entries(legacy_entries, FileKind.PO, _entry_status_from_legacy, plural_key_sort_key)
    output_path = build_output_path(file_path, FileKind.PO)

    def save_po() -> None:
        for entry in entries:
            entry.commit()
        try:
            po.save(output_path)
        except UnicodeEncodeError:
            original_encoding = po.encoding
            po.encoding = "utf-8"
            po.save(output_path)
            print(
                f"Warning: could not write '{output_path}' using PO encoding "
                f"{original_encoding!r}; saved as utf-8 instead."
            )

    return entries, save_po, output_path
