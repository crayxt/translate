from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

import polib

from core.formats import PO_WRAP_WIDTH

VOCABULARY_FILE_EXTENSIONS = (".txt", ".po")


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


def _normalize_vocabulary_cell(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _is_translated_vocabulary_entry(entry: Any) -> bool:
    translated_attr = getattr(entry, "translated", None)
    if callable(translated_attr):
        try:
            return bool(translated_attr())
        except TypeError:
            return False
    return False


def parse_vocabulary_fields(line: str) -> Tuple[str, str, str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "|" not in stripped:
        return None

    raw_parts = stripped.split("|", 3)
    raw_parts += [""] * (4 - len(raw_parts))
    source_term, target_term, part_of_speech, context_note = (
        _normalize_vocabulary_cell(value) for value in raw_parts[:4]
    )
    if not source_term or not target_term:
        return None
    return source_term, target_term, part_of_speech, context_note


def format_vocabulary_text_line(
    source_term: str,
    target_term: str,
    part_of_speech: str = "",
    context_note: str = "",
) -> str:
    return "|".join(
        (
            _normalize_vocabulary_cell(source_term),
            _normalize_vocabulary_cell(target_term),
            _normalize_vocabulary_cell(part_of_speech),
            _normalize_vocabulary_cell(context_note),
        )
    )


def build_vocabulary_identity_key(
    source_term: str,
    part_of_speech: str = "",
    context_note: str = "",
) -> str:
    return "\u241f".join(
        (
            _normalize_vocabulary_cell(source_term).lower(),
            _normalize_vocabulary_cell(part_of_speech).lower(),
            _normalize_vocabulary_cell(context_note).lower(),
        )
    )


def parse_vocabulary_line(line: str) -> Tuple[str, str] | None:
    parsed = parse_vocabulary_fields(line)
    if not parsed:
        return None
    source_term, target_term, _part_of_speech, _context_note = parsed
    return source_term, target_term


def _resolve_vocabulary_source_paths(
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
            and os.path.splitext(name)[1].lower() in VOCABULARY_FILE_EXTENSIONS
        ]
        if not source_paths:
            print(
                f"Warning: {label} directory '{path}' has no supported glossary files "
                "(.txt or .po)."
            )
        return source_paths

    if not os.path.exists(path):
        print(f"Warning: {label} file or directory '{path}' not found.")
        return []

    if not os.path.isfile(path):
        print(f"Warning: {label} path '{path}' is not a file or directory.")
        return []

    return [path]


def _load_vocabulary_records_from_path(
    path: str,
    label: str,
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> List[Tuple[str, str, str, str]]:
    if not path.lower().endswith(".po"):
        content = read_optional_text_file(path, label)
        if not content:
            return []
        records: List[Tuple[str, str, str, str]] = []
        for raw_line in content.splitlines():
            parsed = parse_vocabulary_fields(raw_line)
            if not parsed:
                continue
            records.append(parsed)
        return records

    loader = pofile_loader or polib.pofile
    try:
        glossary = loader(path, wrapwidth=PO_WRAP_WIDTH)
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return []

    records: List[Tuple[str, str, str, str]] = []
    for entry in glossary:
        if not _is_translated_vocabulary_entry(entry):
            continue
        source_term = _normalize_vocabulary_cell(getattr(entry, "msgid", ""))
        target_term = _normalize_vocabulary_cell(getattr(entry, "msgstr", ""))
        part_of_speech = _normalize_vocabulary_cell(getattr(entry, "msgctxt", ""))
        context_note = _normalize_vocabulary_cell(getattr(entry, "tcomment", ""))
        if not source_term or not target_term:
            continue
        records.append((source_term, target_term, part_of_speech, context_note))

    if not records:
        print(f"Warning: {label} file '{path}' has no usable msgid/msgstr glossary pairs.")
    return records


def _load_vocabulary_records(
    path: str | None,
    label: str = "Vocabulary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> List[Tuple[str, str, str, str]]:
    records: List[Tuple[str, str, str, str]] = []
    seen_indices: Dict[str, int] = {}
    for source_path in _resolve_vocabulary_source_paths(path, label):
        for source_term, target_term, part_of_speech, context_note in _load_vocabulary_records_from_path(
            source_path,
            label,
            pofile_loader=pofile_loader,
        ):
            key = build_vocabulary_identity_key(source_term, part_of_speech, context_note)
            record = (source_term, target_term, part_of_speech, context_note)
            if key in seen_indices:
                records[seen_indices[key]] = record
                continue
            seen_indices[key] = len(records)
            records.append(record)
    return records


def load_vocabulary_pairs(
    path: str | None,
    label: str = "Vocabulary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> List[Tuple[str, str]]:
    return [
        (source_term, target_term)
        for source_term, target_term, _part_of_speech, _context_note in _load_vocabulary_records(
            path,
            label,
            pofile_loader=pofile_loader,
        )
    ]


def read_optional_vocabulary_file(
    path: str | None,
    label: str = "Vocabulary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> str | None:
    records = _load_vocabulary_records(
        path,
        label,
        pofile_loader=pofile_loader,
    )
    if not records:
        return None

    normalized_lines = [
        format_vocabulary_text_line(
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

    seeds: List[str] = []
    for candidate in (raw, raw.replace("-", "_"), raw.replace("_", "-")):
        if candidate and candidate not in seeds:
            seeds.append(candidate)

    results: List[str] = []

    def add(code: str | None) -> None:
        if code and code not in results:
            results.append(code)

    for seed in seeds:
        add(seed)
        if "_" in seed:
            add(seed.split("_", 1)[0])
        if "-" in seed:
            add(seed.split("-", 1)[0])

    for value in list(results):
        lower = value.lower()
        add(lower)
        if "_" in lower:
            add(lower.split("_", 1)[0])
        if "-" in lower:
            add(lower.split("-", 1)[0])

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

    for lang_code in build_language_code_candidates(target_lang):
        candidate_path = build_candidate("data", "locales", lang_code, f"{prefix}.{extension}")
        if os.path.isfile(candidate_path):
            return candidate_path
        if allow_directory:
            candidate_dir = build_candidate("data", "locales", lang_code, prefix)
            if os.path.isdir(candidate_dir):
                return candidate_dir
    for lang_code in build_language_code_candidates(target_lang):
        candidate_path = build_candidate("data", lang_code, f"{prefix}.{extension}")
        if os.path.isfile(candidate_path):
            return candidate_path
        if allow_directory:
            candidate_dir = build_candidate("data", lang_code, prefix)
            if os.path.isdir(candidate_dir):
                return candidate_dir
    for lang_code in build_language_code_candidates(target_lang):
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
    "build_vocabulary_identity_key",
    "format_vocabulary_text_line",
    "detect_rules_source",
    "load_vocabulary_pairs",
    "merge_project_rules",
    "parse_vocabulary_fields",
    "parse_vocabulary_line",
    "read_optional_text_file",
    "read_optional_vocabulary_file",
    "resolve_resource_path",
]
