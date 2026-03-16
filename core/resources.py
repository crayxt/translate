from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

import polib

from core.formats import PO_WRAP_WIDTH


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


def parse_vocabulary_line(line: str) -> Tuple[str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#"):
        return None
    separator = " - "
    if separator not in stripped:
        return None
    source_term, target_term = stripped.split(separator, 1)
    source_term = _normalize_vocabulary_cell(source_term)
    target_term = _normalize_vocabulary_cell(target_term)
    if not source_term or not target_term:
        return None
    return source_term, target_term


def load_vocabulary_pairs(
    path: str | None,
    label: str = "Vocabulary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> List[Tuple[str, str]]:
    if not path:
        return []
    if not path.lower().endswith(".po"):
        content = read_optional_text_file(path, label)
        if not content:
            return []
        pairs: List[Tuple[str, str]] = []
        seen_indices: Dict[str, int] = {}
        for raw_line in content.splitlines():
            parsed = parse_vocabulary_line(raw_line)
            if not parsed:
                continue
            source_term, target_term = parsed
            key = source_term.lower()
            if key in seen_indices:
                pairs[seen_indices[key]] = (source_term, target_term)
                continue
            seen_indices[key] = len(pairs)
            pairs.append((source_term, target_term))
        return pairs

    loader = pofile_loader or polib.pofile
    try:
        glossary = loader(path, wrapwidth=PO_WRAP_WIDTH)
    except FileNotFoundError:
        print(f"Warning: {label} file '{path}' not found.")
        return []

    pairs: List[Tuple[str, str]] = []
    for entry in glossary:
        if not _is_translated_vocabulary_entry(entry):
            continue
        source_term = _normalize_vocabulary_cell(getattr(entry, "msgid", ""))
        target_term = _normalize_vocabulary_cell(getattr(entry, "msgstr", ""))
        if not source_term or not target_term:
            continue
        pairs.append((source_term, target_term))
    return pairs


def read_optional_vocabulary_file(
    path: str | None,
    label: str = "Vocabulary",
    *,
    pofile_loader: Callable[..., Any] | None = None,
) -> str | None:
    pairs = load_vocabulary_pairs(path, label, pofile_loader=pofile_loader)
    if not pairs:
        if path and path.lower().endswith(".po"):
            print(f"Warning: {label} file '{path}' has no usable msgid/msgstr glossary pairs.")
        return None
    return "\n".join(f"{source_term} - {target_term}" for source_term, target_term in pairs)


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
) -> str | None:
    resource_root = os.path.abspath(base_dir) if base_dir else None

    def build_candidate(*parts: str) -> str:
        if resource_root:
            return os.path.join(resource_root, *parts)
        return os.path.join(*parts)

    for lang_code in build_language_code_candidates(target_lang):
        candidate_path = build_candidate("data", lang_code, f"{prefix}.{extension}")
        if os.path.isfile(candidate_path):
            return candidate_path
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
) -> str | None:
    if explicit_path:
        return explicit_path
    return detect_default_text_resource(
        prefix,
        extension,
        target_lang,
        base_dir=base_dir,
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
    "detect_rules_source",
    "load_vocabulary_pairs",
    "merge_project_rules",
    "parse_vocabulary_line",
    "read_optional_text_file",
    "read_optional_vocabulary_file",
    "resolve_resource_path",
]
