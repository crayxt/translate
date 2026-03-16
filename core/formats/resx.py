from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Callable, List, Tuple

from core.entries import plural_key_sort_key
from core.formats.base import EntryStatus, FileKind, build_output_path, wrap_legacy_entries


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
        return any(ch.isalpha() for ch in value_text)

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
        return not self._translate

    @property
    def prompt_context(self) -> str:
        return (self.elem.get("name") or "").strip()

    @property
    def prompt_note(self) -> str:
        if self.comment_elem is None or not self.comment_elem.text:
            return ""
        return self.comment_elem.text.strip()


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    if isinstance(entry, ResxEntryAdapter) and not getattr(entry, "_translate", True):
        return EntryStatus.SKIPPED
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def load_resx(file_path: str) -> Tuple[list[Any], Callable[[], None], str]:
    print(f"Processing RESX file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    legacy_entries = [ResxEntryAdapter(data_node) for data_node in root.findall("./data")]
    entries = wrap_legacy_entries(legacy_entries, FileKind.RESX, _entry_status_from_legacy, plural_key_sort_key)
    output_path = build_output_path(file_path, FileKind.RESX)

    def save_resx() -> None:
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_resx, output_path
