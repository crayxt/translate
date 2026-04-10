from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Callable, List, Tuple

from core.entries import plural_key_sort_key
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path, wrap_legacy_entries


class TSEntryAdapter:
    """Adapts a Qt .ts XML <message> element to look like a polib entry."""

    def __init__(self, message_elem: ET.Element, context_name: str | None = None):
        self.elem = message_elem
        self.source_elem = message_elem.find("source")
        self.translation_elem = message_elem.find("translation")
        self.comment_elem = message_elem.find("comment")
        self.extracomment_elem = message_elem.find("numerusform") and None
        self.extracomment_elem = message_elem.find("extracomment")
        self.context_name = context_name or ""
        if self.translation_elem is None:
            self.translation_elem = ET.SubElement(message_elem, "translation")

    @property
    def msgid(self) -> str:
        return self.source_elem.text if self.source_elem is not None else ""

    @property
    def msgid_plural(self) -> str:
        if self.elem.get("numerus") == "yes":
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
        return False

    def translated(self) -> bool:
        t = self.translation_elem
        if t is None:
            return False
        if t.get("type") == "unfinished":
            return False
        if self.elem.get("numerus") == "yes":
            forms = self._numerusform_nodes()
            if not forms:
                return False
            return all(bool(self._clean_form_text(node.text)) for node in forms)
        return bool(t.text)

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


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def load_ts(file_path: str) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
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

    entries = wrap_legacy_entries(legacy_entries, FileKind.TS, _entry_status_from_legacy, plural_key_sort_key)
    output_path = build_output_path(file_path, FileKind.TS)

    def save_ts() -> None:
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_ts, output_path
