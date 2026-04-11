from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable, Iterable, List, Tuple
from xml.sax.saxutils import escape, quoteattr

from core.entries import is_non_empty_text
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path


XLIFF_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"
_TRANSLATED_STATES = frozenset({"translated", "final", "signed-off", "approved"})
_FUZZY_STATES = frozenset(
    {
        "needs-review-translation",
        "needs-review-l10n",
        "needs-adaptation",
        "needs-review",
    }
)
_UNTRANSLATED_STATES = frozenset({"needs-translation", "new", "initial"})


def load_xliff(file_path: str) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
    print(f"Processing XLIFF file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    if _strip_ns(root.tag) != "xliff":
        raise ValueError("Unsupported XML file. Expected XLIFF <xliff> document.")

    namespaces = _collect_namespaces(file_path)
    entries: List[UnifiedEntry] = []

    for file_elem, unit_elem in _iter_trans_units(root):
        source_elem = _find_direct_child(unit_elem, "source")
        if source_elem is None:
            continue

        source_text = _serialize_inner_xml(source_elem)
        target_elem = _find_direct_child(unit_elem, "target")
        target_text = _serialize_inner_xml(target_elem) if target_elem is not None else ""
        original_state = target_elem.get("state") if target_elem is not None else None
        status = _infer_status(unit_elem=unit_elem, source_text=source_text, target_text=target_text, target_elem=target_elem)
        msgstr = target_text if status in (EntryStatus.TRANSLATED, EntryStatus.FUZZY) else ""
        flags = ["fuzzy"] if status == EntryStatus.FUZZY else []
        include_in_term_extraction = _is_translatable_unit(unit_elem) and bool(_clean(source_text))

        target_holder: List[ET.Element | None] = [target_elem]

        def commit_xliff(
            entry: UnifiedEntry,
            unit: ET.Element = unit_elem,
            current_text: str = target_text,
            state_before: str | None = original_state,
            node_namespaces: Tuple[Tuple[str, str], ...] = namespaces,
            target_ref: List[ET.Element | None] = target_holder,
        ) -> None:
            current_target_elem = target_ref[0]
            should_write = is_non_empty_text(entry.msgstr) and (
                entry.status in (EntryStatus.TRANSLATED, EntryStatus.FUZZY) or "fuzzy" in entry.flags
            )
            if should_write:
                if current_target_elem is None:
                    current_target_elem = _ensure_target_element(unit)
                    target_ref[0] = current_target_elem
                _set_inner_xml(current_target_elem, entry.msgstr, node_namespaces)
                if "fuzzy" in entry.flags or entry.status == EntryStatus.FUZZY:
                    current_target_elem.set("state", "needs-review-translation")
                else:
                    current_target_elem.set("state", "translated")
                return

            if current_target_elem is not None:
                _set_inner_xml(current_target_elem, current_text, node_namespaces)
                if state_before:
                    current_target_elem.set("state", state_before)
                else:
                    current_target_elem.attrib.pop("state", None)

        entries.append(
            UnifiedEntry(
                file_kind=FileKind.XLIFF,
                msgid=source_text,
                msgstr=msgstr,
                msgctxt=_build_context(unit_elem),
                prompt_note_text=_build_note(file_elem, unit_elem),
                flags=flags,
                obsolete=False,
                include_in_term_extraction=include_in_term_extraction,
                status=status,
                _commit_callback=commit_xliff,
            )
        )

    output_path = build_output_path(file_path, FileKind.XLIFF)

    def save_xliff() -> None:
        _register_namespaces(namespaces)
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_xliff, output_path


def _collect_namespaces(file_path: str) -> Tuple[Tuple[str, str], ...]:
    namespaces: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    try:
        for _event, value in ET.iterparse(file_path, events=("start-ns",)):
            item = (value[0] or "", value[1] or "")
            if item in seen:
                continue
            seen.add(item)
            namespaces.append(item)
    except (ET.ParseError, OSError):
        return (("", XLIFF_NAMESPACE),)

    if ("", XLIFF_NAMESPACE) not in seen:
        namespaces.append(("", XLIFF_NAMESPACE))
    return tuple(namespaces)


def _register_namespaces(namespaces: Iterable[Tuple[str, str]]) -> None:
    for prefix, uri in namespaces:
        ET.register_namespace(prefix, uri)


def _iter_trans_units(root: ET.Element) -> Iterable[Tuple[ET.Element, ET.Element]]:
    for file_elem in root.findall(f".//{{{XLIFF_NAMESPACE}}}file"):
        for elem in file_elem.iter():
            if _strip_ns(elem.tag) == "trans-unit":
                yield file_elem, elem


def _find_direct_child(parent: ET.Element, tag_name: str) -> ET.Element | None:
    for child in list(parent):
        if _strip_ns(child.tag) == tag_name:
            return child
    return None


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _clean(value: str | None) -> str:
    return str(value or "").strip()


def _is_translatable_unit(unit_elem: ET.Element) -> bool:
    return _clean(unit_elem.get("translate")).lower() != "no"


def _infer_status(
    *,
    unit_elem: ET.Element,
    source_text: str,
    target_text: str,
    target_elem: ET.Element | None,
) -> EntryStatus:
    if not _is_translatable_unit(unit_elem) or not _clean(source_text):
        return EntryStatus.SKIPPED

    state = _clean(target_elem.get("state") if target_elem is not None else "").lower()
    has_target = bool(_clean(target_text))

    if state in _FUZZY_STATES:
        return EntryStatus.FUZZY if has_target else EntryStatus.UNTRANSLATED
    if state in _UNTRANSLATED_STATES:
        return EntryStatus.UNTRANSLATED
    if state in _TRANSLATED_STATES:
        return EntryStatus.TRANSLATED if has_target else EntryStatus.UNTRANSLATED
    if has_target:
        return EntryStatus.TRANSLATED
    return EntryStatus.UNTRANSLATED


def _build_context(unit_elem: ET.Element) -> str:
    return _clean(unit_elem.get("resname")) or _clean(unit_elem.get("id"))


def _build_note(file_elem: ET.Element, unit_elem: ET.Element) -> str:
    parts: List[str] = []

    original = _clean(file_elem.get("original"))
    if original:
        parts.append(f"file: {original}")

    resname = _clean(unit_elem.get("resname"))
    for context_group in list(unit_elem):
        if _strip_ns(context_group.tag) != "context-group":
            continue
        for context in list(context_group):
            if _strip_ns(context.tag) != "context":
                continue
            text = _clean("".join(context.itertext()))
            if not text:
                continue
            if resname:
                if text == resname:
                    continue
                if text.startswith(f"{resname}\n\n"):
                    text = _clean(text[len(resname) + 2 :])
                elif text.startswith(f"{resname}\n"):
                    text = _clean(text[len(resname) + 1 :])
            if text and text not in parts:
                parts.append(text)

    for note in unit_elem.findall(f"{{{XLIFF_NAMESPACE}}}note"):
        if _clean(note.get("from")):
            continue
        text = _clean("".join(note.itertext()))
        if text and text not in parts:
            parts.append(text)

    return " | ".join(parts)


def _serialize_inner_xml(elem: ET.Element | None) -> str:
    if elem is None:
        return ""

    parts: List[str] = []
    if elem.text:
        parts.append(escape(elem.text))
    for child in list(elem):
        parts.append(_serialize_element(child))
        if child.tail:
            parts.append(escape(child.tail))
    return "".join(parts)


def _serialize_element(elem: ET.Element) -> str:
    tag = _strip_ns(elem.tag)
    attrs = "".join(f" { _strip_ns(name) }={quoteattr(str(value))}" for name, value in elem.attrib.items())
    inner = _serialize_inner_xml(elem)
    if inner:
        return f"<{tag}{attrs}>{inner}</{tag}>"
    return f"<{tag}{attrs}/>"


def _ensure_target_element(unit_elem: ET.Element) -> ET.Element:
    target = ET.Element(f"{{{XLIFF_NAMESPACE}}}target")
    children = list(unit_elem)
    for index, child in enumerate(children):
        if _strip_ns(child.tag) == "source":
            unit_elem.insert(index + 1, target)
            return target
    unit_elem.append(target)
    return target


def _set_inner_xml(
    elem: ET.Element,
    fragment: str,
    namespaces: Tuple[Tuple[str, str], ...],
) -> None:
    elem.text = None
    for child in list(elem):
        elem.remove(child)

    raw_fragment = str(fragment or "")
    if not raw_fragment:
        return

    namespace_attrs = "".join(
        f' xmlns{":" + prefix if prefix else ""}="{uri}"'
        for prefix, uri in namespaces
        if uri
    )
    wrapper = f"<wrapper{namespace_attrs}>{raw_fragment}</wrapper>"

    try:
        parsed = ET.fromstring(wrapper)
    except ET.ParseError:
        elem.text = raw_fragment
        return

    elem.text = parsed.text
    for child in list(parsed):
        elem.append(child)
