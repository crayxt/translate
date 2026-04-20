from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Tuple

import polib

from core.formats import PO_WRAP_WIDTH

GLOSSARY_FILE_EXTENSIONS = (".po", ".tbx")
XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"


def read_optional_text_file(path: str | None, label: str) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return None
    if not content:
        print(f"Warning: {label} file '{path}' is empty.")
        return None
    return content


def _normalize_glossary_cell(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _is_translated_glossary_entry(entry: Any) -> bool:
    translated_attr = getattr(entry, "translated", None)
    if callable(translated_attr):
        try:
            return bool(translated_attr())
        except TypeError:
            return False
    return False


def parse_glossary_fields(line: str) -> Tuple[str, str, str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "|" not in stripped:
        return None

    raw_parts = stripped.split("|", 3)
    raw_parts += [""] * (4 - len(raw_parts))
    source_term, target_term, part_of_speech, context_note = (
        _normalize_glossary_cell(value) for value in raw_parts[:4]
    )
    if not source_term or not target_term:
        return None
    return source_term, target_term, part_of_speech, context_note


def format_glossary_text_line(
    source_term: str,
    target_term: str,
    part_of_speech: str = "",
    context_note: str = "",
) -> str:
    return "|".join(
        (
            _normalize_glossary_cell(source_term),
            _normalize_glossary_cell(target_term),
            _normalize_glossary_cell(part_of_speech),
            _normalize_glossary_cell(context_note),
        )
    )


def build_glossary_identity_key(
    source_term: str,
    part_of_speech: str = "",
    context_note: str = "",
) -> str:
    return "\u241f".join(
        (
            _normalize_glossary_cell(source_term).lower(),
            _normalize_glossary_cell(part_of_speech).lower(),
            _normalize_glossary_cell(context_note).lower(),
        )
    )


def parse_glossary_line(line: str) -> Tuple[str, str] | None:
    parsed = parse_glossary_fields(line)
    if not parsed:
        return None
    source_term, target_term, _part_of_speech, _context_note = parsed
    return source_term, target_term


def _strip_xml_namespace(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalize_language_code(value: str | None) -> str:
    return str(value or "").strip().replace("_", "-").lower()


def _extract_catalog_glossary_comment_field(comment: str | None, field_name: str) -> str:
    prefix = f"{field_name}:"
    for raw_line in str(comment or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(prefix):
            return _normalize_glossary_cell(stripped[len(prefix):])
    return ""


def _build_language_code_match_set(value: str | None) -> set[str]:
    return {
        _normalize_language_code(candidate)
        for candidate in build_language_code_candidates(str(value or ""))
        if _normalize_language_code(candidate)
    }


def _get_xml_lang(element: ET.Element) -> str:
    return str(
        element.attrib.get(XML_LANG_ATTR)
        or element.attrib.get("xml:lang")
        or ""
    ).strip()


def _iter_xml_children(element: ET.Element, local_name: str) -> List[ET.Element]:
    return [
        child
        for child in list(element)
        if isinstance(child.tag, str) and _strip_xml_namespace(child.tag) == local_name
    ]


def _iter_xml_descendants(element: ET.Element, local_name: str) -> List[ET.Element]:
    return [
        child
        for child in element.iter()
        if child is not element
        and isinstance(child.tag, str)
        and _strip_xml_namespace(child.tag) == local_name
    ]


def _append_unique_text(values: List[str], raw_value: str | None) -> None:
    normalized = _normalize_glossary_cell(raw_value)
    if normalized and normalized not in values:
        values.append(normalized)


def _collect_tbx_document_languages(root: ET.Element) -> List[str]:
    languages: List[str] = []
    for lang_set in _iter_xml_descendants(root, "langSet"):
        lang_code = _normalize_language_code(_get_xml_lang(lang_set))
        if lang_code and lang_code not in languages:
            languages.append(lang_code)
    return languages


def _resolve_tbx_languages(
    root: ET.Element,
    path: str,
    label: str,
    *,
    target_lang: str | None = None,
) -> Tuple[str | None, str | None]:
    document_languages = _collect_tbx_document_languages(root)
    source_lang = _normalize_language_code(_get_xml_lang(root))
    requested_target_matches = _build_language_code_match_set(target_lang)

    if not source_lang and requested_target_matches and len(document_languages) == 2:
        for lang_code in document_languages:
            if lang_code not in requested_target_matches:
                source_lang = lang_code
                break

    if not source_lang and document_languages:
        source_lang = document_languages[0]

    if not source_lang:
        print(f"Warning: {label} file '{path}' has no usable TBX source language metadata.")
        return None, None

    if requested_target_matches:
        for lang_code in document_languages:
            if lang_code in requested_target_matches:
                return source_lang, lang_code
        print(
            f"Warning: {label} file '{path}' has no TBX langSet matching target language "
            f"'{target_lang}'."
        )
        return source_lang, None

    non_source_languages = [
        lang_code for lang_code in document_languages if lang_code and lang_code != source_lang
    ]
    if len(non_source_languages) == 1:
        return source_lang, non_source_languages[0]
    if len(non_source_languages) > 1:
        print(
            f"Warning: {label} file '{path}' contains multiple TBX target languages. "
            "Pass the task target language so the correct glossary language can be selected."
        )
        return source_lang, None
    print(f"Warning: {label} file '{path}' has no TBX target language entries.")
    return source_lang, None


def _collect_tbx_terms(lang_set: ET.Element) -> List[str]:
    values: List[str] = []
    for term_element in _iter_xml_descendants(lang_set, "term"):
        _append_unique_text(values, "".join(term_element.itertext()))
    return values


def _collect_tbx_context_notes(
    term_entry: ET.Element,
    *,
    source_lang: str | None,
    target_lang: str | None,
) -> str:
    notes: List[str] = []
    for child in list(term_entry):
        local_name = _strip_xml_namespace(child.tag)
        if local_name in ("descrip", "note"):
            _append_unique_text(notes, "".join(child.itertext()))
            continue
        if local_name != "langSet":
            continue
        lang_code = _normalize_language_code(_get_xml_lang(child))
        if lang_code not in {source_lang, target_lang}:
            continue
        for note_element in _iter_xml_descendants(child, "descrip") + _iter_xml_descendants(child, "note"):
            _append_unique_text(notes, "".join(note_element.itertext()))
    return " ".join(notes)


def _extract_tbx_part_of_speech(
    term_entry: ET.Element,
    *,
    source_lang: str | None,
    target_lang: str | None,
) -> str:
    for scope in [term_entry] + _iter_xml_children(term_entry, "langSet"):
        if scope is not term_entry:
            lang_code = _normalize_language_code(_get_xml_lang(scope))
            if lang_code not in {source_lang, target_lang}:
                continue
        for term_note in _iter_xml_descendants(scope, "termNote"):
            note_type = _normalize_glossary_cell(
                term_note.attrib.get("type")
                or term_note.attrib.get("termNoteType")
                or ""
            ).lower().replace(" ", "")
            if note_type not in {"pos", "partofspeech"} and "speech" not in note_type:
                continue
            value = _normalize_glossary_cell("".join(term_note.itertext()))
            if value:
                return value
    return ""


def _load_tbx_glossary_records(
    path: str,
    label: str,
    *,
    target_lang: str | None = None,
) -> List[Tuple[str, str, str, str]]:
    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return []
    except ET.ParseError as exc:
        print(f"Warning: {label} file '{path}' is not valid TBX XML: {exc}")
        return []
    except OSError as exc:
        print(f"Warning: Could not read {label} file '{path}': {exc}")
        return []

    source_lang, resolved_target_lang = _resolve_tbx_languages(
        root,
        path,
        label,
        target_lang=target_lang,
    )
    if not source_lang or not resolved_target_lang:
        return []

    records: List[Tuple[str, str, str, str]] = []
    for term_entry in _iter_xml_descendants(root, "termEntry"):
        source_terms: List[str] = []
        target_terms: List[str] = []
        for lang_set in _iter_xml_children(term_entry, "langSet"):
            lang_code = _normalize_language_code(_get_xml_lang(lang_set))
            terms = _collect_tbx_terms(lang_set)
            if not terms:
                continue
            if lang_code == source_lang:
                for term in terms:
                    _append_unique_text(source_terms, term)
            elif lang_code == resolved_target_lang:
                for term in terms:
                    _append_unique_text(target_terms, term)

        if not source_terms or not target_terms:
            continue

        part_of_speech = _extract_tbx_part_of_speech(
            term_entry,
            source_lang=source_lang,
            target_lang=resolved_target_lang,
        )
        context_note = _collect_tbx_context_notes(
            term_entry,
            source_lang=source_lang,
            target_lang=resolved_target_lang,
        )
        for source_term in source_terms:
            for target_term in target_terms:
                records.append((source_term, target_term, part_of_speech, context_note))

    if not records:
        print(f"Warning: {label} file '{path}' has no usable TBX term pairs.")
    return records


def _resolve_glossary_source_paths(
    path: str | None,
    label: str,
) -> List[str]:
    if not path:
        return []

    if os.path.isdir(path):
        source_paths = [
            os.path.join(path, name)
            for name in sorted(os.listdir(path), key=str.lower)
            if os.path.isfile(os.path.join(path, name))
            and os.path.splitext(name)[1].lower() in GLOSSARY_FILE_EXTENSIONS
        ]
        if not source_paths:
            print(
                f"Warning: {label} directory '{path}' has no supported glossary files "
                "(.po or .tbx)."
            )
        return source_paths

    if not os.path.exists(path):
        print(f"Warning: {label} file or directory '{path}' not found.")
        return []

    if not os.path.isfile(path):
        print(f"Warning: {label} path '{path}' is not a file or directory.")
        return []

    return [path]


def _load_glossary_records_from_path(
    path: str,
    label: str,
    *,
    pofile_loader: Callable[..., Any] | None = None,
    target_lang: str | None = None,
) -> List[Tuple[str, str, str, str]]:
    lowered_path = path.lower()
    if lowered_path.endswith(".tbx"):
        return _load_tbx_glossary_records(path, label, target_lang=target_lang)

    if not lowered_path.endswith(".po"):
        print(
            f"Warning: {label} file '{path}' uses an unsupported glossary format. "
            "Only .po and .tbx glossary files are supported."
        )
        return []

    loader = pofile_loader or polib.pofile
    try:
        glossary = loader(path, wrapwidth=PO_WRAP_WIDTH)
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return []

    records: List[Tuple[str, str, str, str]] = []
    for entry in glossary:
        if not _is_translated_glossary_entry(entry):
            continue
        source_term = _normalize_glossary_cell(getattr(entry, "msgid", ""))
        target_term = _normalize_glossary_cell(getattr(entry, "msgstr", ""))
        extracted_comment = getattr(entry, "comment", "")
        catalog_part_of_speech = _extract_catalog_glossary_comment_field(extracted_comment, "POS")
        part_of_speech = catalog_part_of_speech or _normalize_glossary_cell(getattr(entry, "msgctxt", ""))
        if catalog_part_of_speech:
            context_note = _normalize_glossary_cell(getattr(entry, "msgctxt", ""))
        else:
            context_note = _normalize_glossary_cell(getattr(entry, "tcomment", ""))
        if not source_term or not target_term:
            continue
        records.append((source_term, target_term, part_of_speech, context_note))

    if not records:
        print(f"Warning: {label} file '{path}' has no usable msgid/msgstr glossary pairs.")
    return records


def _load_glossary_records(
    path: str | None,
    label: str = "Glossary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
    target_lang: str | None = None,
) -> List[Tuple[str, str, str, str]]:
    records: List[Tuple[str, str, str, str]] = []
    seen_indices: Dict[str, int] = {}
    for source_path in _resolve_glossary_source_paths(path, label):
        for source_term, target_term, part_of_speech, context_note in _load_glossary_records_from_path(
            source_path,
            label,
            pofile_loader=pofile_loader,
            target_lang=target_lang,
        ):
            key = build_glossary_identity_key(source_term, part_of_speech, context_note)
            record = (source_term, target_term, part_of_speech, context_note)
            if key in seen_indices:
                records[seen_indices[key]] = record
                continue
            seen_indices[key] = len(records)
            records.append(record)
    return records


def load_glossary_pairs(
    path: str | None,
    label: str = "Glossary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
    target_lang: str | None = None,
) -> List[Tuple[str, str]]:
    return [
        (source_term, target_term)
        for source_term, target_term, _part_of_speech, _context_note in _load_glossary_records(
            path,
            label,
            pofile_loader=pofile_loader,
            target_lang=target_lang,
        )
    ]


def read_optional_glossary_file(
    path: str | None,
    label: str = "Glossary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
    target_lang: str | None = None,
) -> str | None:
    records = _load_glossary_records(
        path,
        label,
        pofile_loader=pofile_loader,
        target_lang=target_lang,
    )
    if not records:
        return None

    normalized_lines = [
        format_glossary_text_line(
            source_term,
            target_term,
            part_of_speech,
            context_note,
        )
        for source_term, target_term, part_of_speech, context_note in records
    ]
    return "\n".join(normalized_lines)


def build_language_code_candidates(target_lang: str) -> List[str]:
    raw = (target_lang or "").strip()
    if not raw:
        return []

    results: List[str] = []

    def add(code: str | None) -> None:
        if code and code not in results:
            results.append(code)

    for candidate in (raw, raw.replace("-", "_"), raw.replace("_", "-")):
        add(candidate)

    for value in list(results):
        add(value.lower())

    return results


def detect_default_text_resource(
    prefix: str,
    extension: str,
    target_lang: str,
    *,
    base_dir: str | None = None,
    allow_directory: bool = False,
) -> str | None:
    resource_root = os.path.abspath(base_dir) if base_dir else None

    def build_candidate(*parts: str) -> str:
        if resource_root:
            return os.path.join(resource_root, *parts)
        return os.path.join(*parts)

    def iter_candidate_specs() -> List[Tuple[str, str | None, bool]]:
        if prefix == "glossary":
            specs: List[Tuple[str, str | None, bool]] = [("glossary", "po", False)]
            if allow_directory:
                specs.append((prefix, None, True))
            specs.append((prefix, "tbx", False))
            return specs
        return [(prefix, extension, False)] + ([(prefix, None, True)] if allow_directory else [])

    candidate_specs = iter_candidate_specs()

    for lang_code in build_language_code_candidates(target_lang):
        for candidate_prefix, candidate_extension, is_dir in candidate_specs:
            if is_dir:
                candidate_dir = build_candidate("data", "locales", lang_code, candidate_prefix)
                if os.path.isdir(candidate_dir):
                    return candidate_dir
                continue
            candidate_path = build_candidate(
                "data",
                "locales",
                lang_code,
                f"{candidate_prefix}.{candidate_extension}",
            )
            if os.path.isfile(candidate_path):
                return candidate_path
    for lang_code in build_language_code_candidates(target_lang):
        for candidate_prefix, candidate_extension, is_dir in candidate_specs:
            if is_dir:
                candidate_dir = build_candidate("data", lang_code, candidate_prefix)
                if os.path.isdir(candidate_dir):
                    return candidate_dir
                continue
            candidate_path = build_candidate(
                "data",
                lang_code,
                f"{candidate_prefix}.{candidate_extension}",
            )
            if os.path.isfile(candidate_path):
                return candidate_path
    for lang_code in build_language_code_candidates(target_lang):
        if prefix == "glossary":
            glossary_legacy_path = build_candidate(f"glossary-{lang_code}.po")
            if os.path.isfile(glossary_legacy_path):
                return glossary_legacy_path
        legacy_path = build_candidate(f"{prefix}-{lang_code}.{extension}")
        if os.path.isfile(legacy_path):
            return legacy_path
    return None


def resolve_resource_path(
    explicit_path: str | None,
    prefix: str,
    extension: str,
    target_lang: str,
    *,
    base_dir: str | None = None,
    allow_directory: bool = False,
) -> str | None:
    if explicit_path:
        return explicit_path
    return detect_default_text_resource(
        prefix,
        extension,
        target_lang,
        base_dir=base_dir,
        allow_directory=allow_directory,
    )


def merge_project_rules(file_rules: str | None, inline_rules: str | None) -> str | None:
    parts: List[str] = []
    if file_rules:
        parts.append(file_rules.strip())
    if inline_rules and inline_rules.strip():
        parts.append(inline_rules.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def detect_rules_source(
    rules_path: str | None,
    file_rules: str | None,
    inline_rules: str | None,
) -> str | None:
    sources: List[str] = []
    if file_rules and rules_path:
        sources.append(f"file:{rules_path}")
    if inline_rules and inline_rules.strip():
        sources.append("inline:--rules-str")
    if not sources:
        return None
    return ", ".join(sources)


__all__ = [
    "build_language_code_candidates",
    "detect_default_text_resource",
    "build_glossary_identity_key",
    "format_glossary_text_line",
    "detect_rules_source",
    "load_glossary_pairs",
    "merge_project_rules",
    "parse_glossary_fields",
    "parse_glossary_line",
    "read_optional_text_file",
    "read_optional_glossary_file",
    "resolve_resource_path",
]
