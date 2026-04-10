from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from core.entries import is_non_empty_text
from core.formats.base import EntryStatus, FileKind, UnifiedEntry, build_output_path


TOOLS_NAMESPACE = "http://schemas.android.com/tools"
XLIFF_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"


@dataclass(frozen=True)
class AndroidResourceNode:
    kind: str
    name: str
    element: ET.Element
    item_elements: Tuple[Tuple[str, ET.Element], ...]
    translatable: bool
    note: str = ""


def is_android_resources_xml(file_path: str) -> bool:
    try:
        tree = _parse_tree(file_path)
    except (ET.ParseError, OSError):
        return False
    return _strip_ns(tree.getroot().tag) == "resources"


def load_android_xml(file_path: str) -> Tuple[list[UnifiedEntry], Callable[[], None], str]:
    print(f"Processing Android XML file: {file_path}")
    tree = _parse_tree(file_path)
    root = tree.getroot()
    if _strip_ns(root.tag) != "resources":
        raise ValueError("Unsupported XML file. Expected Android <resources> XML.")

    namespaces = _collect_namespaces(file_path)
    nodes = _collect_resource_nodes(root)
    entries: List[UnifiedEntry] = []

    for node in nodes:
        if node.kind == "string":
            source_text = _serialize_inner_xml(node.element)
            note = _build_note(node)
            status = EntryStatus.UNTRANSLATED if node.translatable and _clean(source_text) else EntryStatus.SKIPPED

            def commit_string(
                entry: UnifiedEntry,
                elem: ET.Element = node.element,
                original: str = source_text,
                node_namespaces: Tuple[Tuple[str, str], ...] = namespaces,
            ) -> None:
                value = original
                if entry.status in (EntryStatus.FUZZY, EntryStatus.TRANSLATED) and is_non_empty_text(entry.msgstr):
                    value = entry.msgstr
                _set_inner_xml(elem, value, node_namespaces)

            entries.append(
                UnifiedEntry(
                    file_kind=FileKind.ANDROID_XML,
                    msgid=source_text,
                    msgstr="",
                    msgctxt=_build_context(node),
                    prompt_note_text=note,
                    flags=[],
                    obsolete=False,
                    include_in_term_extraction=bool(node.translatable and _clean(source_text)),
                    status=status,
                    _commit_callback=commit_string,
                )
            )
            continue

        source_plural_map = _build_plural_map(node.item_elements)
        source_text, plural_text = _choose_plural_source_texts(source_plural_map)
        note = _build_note(node, source_plural_map)
        status = (
            EntryStatus.UNTRANSLATED
            if node.translatable and any(_clean(text) for text in source_plural_map.values())
            else EntryStatus.SKIPPED
        )

        def commit_plural(
            entry: UnifiedEntry,
            items: Tuple[Tuple[str, ET.Element], ...] = node.item_elements,
            original_map: Dict[str, str] = dict(source_plural_map),
            node_namespaces: Tuple[Tuple[str, str], ...] = namespaces,
        ) -> None:
            for quantity, item_elem in items:
                value = original_map.get(quantity, "")
                if entry.status in (EntryStatus.FUZZY, EntryStatus.TRANSLATED):
                    candidate = str(entry.msgstr_plural.get(quantity, "") or "")
                    if is_non_empty_text(candidate):
                        value = candidate
                _set_inner_xml(item_elem, value, node_namespaces)

        entries.append(
            UnifiedEntry(
                file_kind=FileKind.ANDROID_XML,
                msgid=source_text,
                msgid_plural=plural_text,
                msgstr="",
                msgstr_plural={quantity: "" for quantity in source_plural_map.keys()},
                msgctxt=_build_context(node),
                prompt_note_text=note,
                flags=[],
                obsolete=False,
                include_in_term_extraction=bool(
                    node.translatable and any(_clean(text) for text in source_plural_map.values())
                ),
                status=status,
                _commit_callback=commit_plural,
            )
        )

    output_path = build_output_path(file_path, FileKind.ANDROID_XML)

    def save_android_xml() -> None:
        _register_namespaces(namespaces)
        for entry in entries:
            entry.commit()
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_android_xml, output_path


def load_paired_android_xml(
    source_file: str,
    translated_file: str,
) -> Tuple[list[UnifiedEntry], Callable[[], None], str, List[str]]:
    print(f"Processing paired Android XML files: source={source_file}, translated={translated_file}")

    source_tree = _parse_tree(source_file)
    translated_tree = _parse_tree(translated_file)
    source_root = source_tree.getroot()
    translated_root = translated_tree.getroot()

    if _strip_ns(source_root.tag) != "resources" or _strip_ns(translated_root.tag) != "resources":
        raise ValueError("Paired Android XML workflow requires Android <resources> XML files.")

    translated_namespaces = _collect_namespaces(translated_file)
    source_nodes = _collect_resource_nodes(source_root)
    translated_nodes = _collect_resource_nodes(translated_root)

    source_map: Dict[Tuple[str, str], AndroidResourceNode] = {
        (node.kind, node.name): node
        for node in source_nodes
    }
    translated_keys: set[Tuple[str, str]] = set()

    entries: List[UnifiedEntry] = []
    warnings: List[str] = []
    missing_source_pairs = 0

    for translated_node in translated_nodes:
        key = (translated_node.kind, translated_node.name)
        translated_keys.add(key)
        source_node = source_map.get(key)
        if source_node is None:
            missing_source_pairs += 1
            continue

        if not source_node.translatable or not translated_node.translatable:
            continue

        if translated_node.kind == "string":
            source_text = _serialize_inner_xml(source_node.element)
            current_text = _serialize_inner_xml(translated_node.element)
            note = _join_non_empty(_build_note(source_node), _build_note(translated_node))
            status = EntryStatus.TRANSLATED if _clean(current_text) else EntryStatus.UNTRANSLATED

            def commit_string(
                entry: UnifiedEntry,
                elem: ET.Element = translated_node.element,
                current: str = current_text,
                node_namespaces: Tuple[Tuple[str, str], ...] = translated_namespaces,
            ) -> None:
                _set_inner_xml(elem, entry.msgstr if entry.msgstr is not None else current, node_namespaces)

            entries.append(
                UnifiedEntry(
                    file_kind=FileKind.ANDROID_XML,
                    msgid=source_text,
                    msgstr=current_text,
                    msgctxt=_build_context(translated_node),
                    prompt_note_text=note,
                    flags=[],
                    obsolete=False,
                    include_in_term_extraction=False,
                    status=status,
                    _commit_callback=commit_string,
                )
            )
            continue

        source_plural_map = _build_plural_map(source_node.item_elements)
        current_plural_map = _build_plural_map(translated_node.item_elements)
        source_text, plural_text = _choose_plural_source_texts(source_plural_map)
        note = _join_non_empty(
            _build_note(source_node, source_plural_map),
            _build_note(translated_node, current_plural_map),
        )
        status = (
            EntryStatus.TRANSLATED
            if current_plural_map and all(_clean(text) for text in current_plural_map.values())
            else EntryStatus.UNTRANSLATED
        )

        def commit_plural(
            entry: UnifiedEntry,
            items: Tuple[Tuple[str, ET.Element], ...] = translated_node.item_elements,
            current_map: Dict[str, str] = dict(current_plural_map),
            node_namespaces: Tuple[Tuple[str, str], ...] = translated_namespaces,
        ) -> None:
            for quantity, item_elem in items:
                value = str(entry.msgstr_plural.get(quantity, current_map.get(quantity, "")) or "")
                _set_inner_xml(item_elem, value, node_namespaces)

        entries.append(
            UnifiedEntry(
                file_kind=FileKind.ANDROID_XML,
                msgid=source_text,
                msgid_plural=plural_text,
                msgstr=_pick_plural_primary_text(current_plural_map),
                msgstr_plural=dict(current_plural_map),
                msgctxt=_build_context(translated_node),
                prompt_note_text=note,
                flags=[],
                obsolete=False,
                include_in_term_extraction=False,
                status=status,
                _commit_callback=commit_plural,
            )
        )

    if missing_source_pairs:
        warnings.append(
            f"Skipped {missing_source_pairs} translated Android XML entries with no matching source pair in {source_file}."
        )

    missing_translated_pairs = sum(
        1
        for key, node in source_map.items()
        if node.translatable and key not in translated_keys
    )
    if missing_translated_pairs:
        warnings.append(
            f"Source Android XML contains {missing_translated_pairs} translatable entries missing from {translated_file}."
        )

    output_path = build_output_path(translated_file, FileKind.ANDROID_XML)

    def save_android_xml() -> None:
        _register_namespaces(translated_namespaces)
        for entry in entries:
            entry.commit()
        translated_tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return entries, save_android_xml, output_path, warnings


def _parse_tree(file_path: str) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(file_path, parser=parser)


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
        return (("tools", TOOLS_NAMESPACE), ("xliff", XLIFF_NAMESPACE))

    if ("tools", TOOLS_NAMESPACE) not in seen:
        namespaces.append(("tools", TOOLS_NAMESPACE))
    if ("xliff", XLIFF_NAMESPACE) not in seen:
        namespaces.append(("xliff", XLIFF_NAMESPACE))
    return tuple(namespaces)


def _register_namespaces(namespaces: Tuple[Tuple[str, str], ...]) -> None:
    for prefix, uri in namespaces:
        ET.register_namespace(prefix, uri)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _clean(value: str | None) -> str:
    return str(value or "").strip()


def _join_non_empty(*parts: str) -> str:
    values = [str(part).strip() for part in parts if str(part or "").strip()]
    return " | ".join(values)


def _build_context(node: AndroidResourceNode) -> str:
    return f"{node.kind}:{node.name}"


def _build_note(node: AndroidResourceNode, plural_map: Dict[str, str] | None = None) -> str:
    parts: List[str] = []
    if node.note:
        parts.append(node.note)
    if node.kind == "string":
        parts.append("android string resource")
    else:
        parts.append("android plural resource")
        quantities = ", ".join(quantity for quantity, _ in node.item_elements if quantity)
        if quantities:
            parts.append(f"quantities: {quantities}")
        if plural_map:
            forms = [f"{quantity}={text}" for quantity, text in plural_map.items() if _clean(text)]
            if forms:
                parts.append("forms: " + " | ".join(forms))
    if node.element.get("formatted") == "false":
        parts.append("formatted=false")
    return " | ".join(parts)


def _collect_resource_nodes(root: ET.Element) -> List[AndroidResourceNode]:
    nodes: List[AndroidResourceNode] = []
    pending_notes: List[str] = []

    for child in list(root):
        if not isinstance(child.tag, str):
            comment = _clean(child.text)
            if comment:
                pending_notes.append(comment)
            continue

        tag = _strip_ns(child.tag)
        if tag not in {"string", "plurals"}:
            pending_notes = []
            continue

        name = _clean(child.get("name"))
        if not name:
            pending_notes = []
            continue

        note = " | ".join(pending_notes)
        pending_notes = []
        translatable = str(child.get("translatable", "true")).strip().lower() != "false"

        if tag == "string":
            nodes.append(
                AndroidResourceNode(
                    kind="string",
                    name=name,
                    element=child,
                    item_elements=(),
                    translatable=translatable,
                    note=note,
                )
            )
            continue

        item_elements: List[Tuple[str, ET.Element]] = []
        for item in list(child):
            if not isinstance(item.tag, str) or _strip_ns(item.tag) != "item":
                continue
            quantity = _clean(item.get("quantity"))
            item_elements.append((quantity, item))

        nodes.append(
            AndroidResourceNode(
                kind="plurals",
                name=name,
                element=child,
                item_elements=tuple(item_elements),
                translatable=translatable,
                note=note,
            )
        )

    return nodes


def _serialize_inner_xml(elem: ET.Element) -> str:
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in list(elem):
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts)


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


def _build_plural_map(item_elements: Tuple[Tuple[str, ET.Element], ...]) -> Dict[str, str]:
    return {
        quantity: _serialize_inner_xml(item_elem)
        for quantity, item_elem in item_elements
    }


def _choose_plural_source_texts(plural_map: Dict[str, str]) -> Tuple[str, str]:
    singular = _pick_first_matching_text(plural_map, ("one", "zero", "two", "few", "many", "other"))
    plural = _pick_first_matching_text(plural_map, ("other", "many", "few", "two", "one", "zero"))
    if not plural:
        plural = singular
    return singular, plural


def _pick_first_matching_text(plural_map: Dict[str, str], preferred_keys: Tuple[str, ...]) -> str:
    for key in preferred_keys:
        value = plural_map.get(key)
        if _clean(value):
            return value
    for value in plural_map.values():
        if _clean(value):
            return value
    return ""


def _pick_plural_primary_text(plural_map: Dict[str, str]) -> str:
    return _pick_first_matching_text(plural_map, ("one", "other", "zero", "two", "few", "many"))
