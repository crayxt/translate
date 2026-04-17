from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

import polib

from core.formats import PO_WRAP_WIDTH


@dataclass(frozen=True)
class GlossarySourceTerm:
    """One canonical source glossary entry keyed by a stable sense identifier."""

    source_term: str
    part_of_speech: str = ""
    sense: str = ""
    context_note: str = ""
    id: str = ""
    example: str = ""


def _normalize_field(value: object) -> str:
    return " ".join(str(value or "").split())


def _format_gettext_timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M%z")


def _build_glossary_metadata(
    *,
    source_tag: str,
    language: str | None = None,
) -> dict[str, str]:
    timestamp = _format_gettext_timestamp()
    metadata = {
        "Project-Id-Version": "Glossary",
        "POT-Creation-Date": timestamp,
        "PO-Revision-Date": timestamp if language else "YEAR-MO-DA HO:MI+ZONE",
        "MIME-Version": "1.0",
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Transfer-Encoding": "8bit",
        "Generated-By": "scripts/sync_glossary_catalog.py",
        "X-Glossary-Source": source_tag,
    }
    if language:
        metadata["Language"] = language
    return metadata


def _slugify_component(value: str) -> str:
    lowered = _normalize_field(value).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "term"


_FALLBACK_CONTEXT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ui",
        (
            "bookmark",
            "browser",
            "button",
            "click",
            "control",
            "cursor",
            "dialog",
            "menu",
            "window",
        ),
    ),
    (
        "document",
        (
            "chapter",
            "column",
            "directory",
            "document",
            "endnote",
            "file",
            "filename",
            "folder",
            "footer",
            "footnote",
            "form",
            "page",
        ),
    ),
    (
        "visual",
        (
            "background",
            "bevel",
            "black",
            "blue",
            "bold",
            "brightness",
            "brush",
            "color",
            "curve",
            "effect",
            "emboss",
            "font",
            "fullscreen",
        ),
    ),
    (
        "media_data",
        (
            "bandwidth",
            "bit",
            "bitrate",
            "buffer",
            "byte",
            "caching",
            "channel",
            "checksum",
            "codec",
            "compression",
            "connection",
            "data",
            "database",
            "decode",
            "decoder",
            "demuxer",
            "device",
            "disc",
            "download",
            "dump",
            "encoder",
            "encoding",
            "encryption",
            "frequency",
            "record",
        ),
    ),
)


def _classify_fallback_context(source_term: str) -> str:
    lowered = _normalize_field(source_term).lower()
    for label, keywords in _FALLBACK_CONTEXT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "general"


def build_fallback_context_note(source_term: str, part_of_speech: str = "") -> str:
    normalized_source = _normalize_field(source_term)
    normalized_pos = _normalize_field(part_of_speech)
    category = _classify_fallback_context(normalized_source)

    if normalized_pos == "noun":
        if category == "ui":
            return f"{normalized_source} as a UI control, window, or interaction term"
        if category == "document":
            return f"{normalized_source} as a file, document, or content-structure term"
        if category == "visual":
            return f"{normalized_source} as a visual style, graphics, or display term"
        if category == "media_data":
            return f"{normalized_source} as a media, data, or technical processing term"
        return f"{normalized_source} as a UI, document, or technical noun"

    if normalized_pos == "verb":
        if category == "ui":
            return f"action to {normalized_source} in the UI or application workflow"
        if category in {"document", "media_data"}:
            return f"action to {normalized_source} files, data, media, or application state"
        return f"action to {normalized_source} in the UI or technical workflow"

    if normalized_pos == "adjective":
        if category == "visual":
            return f"{normalized_source} appearance, style, or display option"
        return f"{normalized_source} attribute, state, or option in UI or technical context"

    if normalized_pos == "adverb":
        return f"{normalized_source} manner or direction in UI or technical behavior"

    if normalized_pos == "conjunction":
        return f"{normalized_source} as a connective term in UI or technical text"

    if normalized_pos == "verb phrase":
        return f"{normalized_source} as an application or technical action phrase"

    return normalized_source or normalized_pos


def build_glossary_entry_id(
    source_term: str,
    part_of_speech: str = "",
    sense: str = "",
) -> str:
    """Build a readable glossary id from source term, POS, and sense."""
    normalized_source = _normalize_field(source_term)
    normalized_pos = _normalize_field(part_of_speech)
    pieces = [_slugify_component(normalized_source)]
    if normalized_pos:
        pieces.append(_slugify_component(normalized_pos))
    normalized_sense = _normalize_field(sense)
    pieces.append(_slugify_component(normalized_sense) if normalized_sense else "default")
    return ".".join(pieces)


_SHORT_SENSE_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("command line", "command-line", "cli", "shell"), "cli"),
    (("line of text", "text line", "text layout", "editor text", "line in a file"), "text"),
    (("geometry", "border", "drawing", "stroke", "graphic"), "geometry"),
    (("abbreviation", "short form"), "abbreviation"),
    (("summary",), "summary"),
    (("abstract art",), "art"),
)

_SENSE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "generated",
    "used",
}


def suggest_glossary_sense(context_note: str | None, source_term: str = "") -> str:
    """Suggest a short readable sense label from a context note."""
    normalized = _normalize_field(context_note)
    if not normalized:
        return "default"

    lowered = normalized.lower()
    for needles, sense in _SHORT_SENSE_PATTERNS:
        if any(needle in lowered for needle in needles):
            return sense

    first_clause = re.split(r"[,;]", normalized, maxsplit=1)[0].strip()
    if not first_clause:
        first_clause = normalized

    clause_tokens = re.findall(r"[a-z0-9]+", first_clause.lower())
    source_tokens = set(re.findall(r"[a-z0-9]+", _normalize_field(source_term).lower()))
    significant_tokens = [
        token
        for token in clause_tokens
        if token not in _SENSE_STOPWORDS and token not in source_tokens
    ]
    if significant_tokens:
        return _slugify_component(significant_tokens[0])

    candidate = _slugify_component(first_clause)
    return candidate or "default"


def _sort_key(item: GlossarySourceTerm) -> tuple[str, str, str, str]:
    return (
        item.source_term.lower(),
        item.part_of_speech.lower(),
        item.sense.lower(),
        item.context_note.lower(),
        item.id.lower(),
    )


def load_glossary_source_terms(path: str | Path) -> list[GlossarySourceTerm]:
    """Load canonical glossary entries from a JSONL file."""
    source_path = Path(path)
    entries: List[GlossarySourceTerm] = []
    seen_ids: set[str] = set()

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source_path}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Line {line_number} of {source_path} must be a JSON object."
                )

            entry = GlossarySourceTerm(
                source_term=_normalize_field(payload.get("source_term")),
                part_of_speech=_normalize_field(payload.get("part_of_speech")),
                sense=_normalize_field(payload.get("sense")),
                context_note=_normalize_field(payload.get("context_note")),
                id=_normalize_field(payload.get("id")),
                example=_normalize_field(payload.get("example")),
            )
            if not entry.source_term:
                raise ValueError(
                    f"Line {line_number} of {source_path} is missing `source_term`."
                )
            if not entry.sense:
                raise ValueError(f"Line {line_number} of {source_path} is missing `sense`.")
            if not entry.id:
                raise ValueError(f"Line {line_number} of {source_path} is missing `id`.")
            if entry.id in seen_ids:
                raise ValueError(f"Duplicate glossary id `{entry.id}` in {source_path}.")

            seen_ids.add(entry.id)
            entries.append(entry)

    entries.sort(key=_sort_key)
    return entries


def build_glossary_source_terms_from_records(
    records: Sequence[Tuple[str, str, str, str]],
) -> list[GlossarySourceTerm]:
    """Convert rich glossary records into canonical source glossary entries."""
    entries: List[GlossarySourceTerm] = []
    seen_ids: set[str] = set()
    for source_term, _target_term, part_of_speech, context_note, sense in _iter_records_with_senses(records):
        normalized_source = _normalize_field(source_term)
        normalized_pos = _normalize_field(part_of_speech)
        normalized_context = _normalize_field(context_note) or build_fallback_context_note(
            normalized_source,
            normalized_pos,
        )
        entry = GlossarySourceTerm(
            source_term=normalized_source,
            part_of_speech=normalized_pos,
            sense=sense,
            context_note=normalized_context,
            id=build_glossary_entry_id(normalized_source, normalized_pos, sense),
        )
        if entry.id in seen_ids:
            raise ValueError(f"Duplicate generated glossary id `{entry.id}`.")
        seen_ids.add(entry.id)
        entries.append(entry)
    entries.sort(key=_sort_key)
    return entries


def _build_extracted_comment(entry: GlossarySourceTerm) -> str:
    lines = [f"ID: {entry.id}"]
    if entry.part_of_speech:
        lines.append(f"POS: {entry.part_of_speech}")
    if entry.sense:
        lines.append(f"Sense: {entry.sense}")
    if entry.example:
        lines.append(f"Example: {entry.example}")
    return "\n".join(lines)


def _build_msgctxt_candidate(entry: GlossarySourceTerm) -> str:
    return entry.context_note or entry.part_of_speech or entry.sense or entry.id


def _build_msgctxt_disambiguator(entry: GlossarySourceTerm) -> str:
    parts: list[str] = []
    if entry.part_of_speech:
        parts.append(entry.part_of_speech)
    if entry.sense and entry.sense not in parts:
        parts.append(entry.sense)
    return "/".join(parts) or entry.id


def _build_msgctxt_map(entries: Iterable[GlossarySourceTerm]) -> dict[str, str]:
    grouped: dict[tuple[str, str], list[GlossarySourceTerm]] = defaultdict(list)
    for entry in sorted(entries, key=_sort_key):
        grouped[(entry.source_term, _build_msgctxt_candidate(entry))].append(entry)

    msgctxt_by_id: dict[str, str] = {}
    for (_source_term, base_msgctxt), group_entries in grouped.items():
        if len(group_entries) == 1:
            msgctxt_by_id[group_entries[0].id] = base_msgctxt
            continue

        seen_contexts: set[str] = set()
        for entry in group_entries:
            disambiguator = _build_msgctxt_disambiguator(entry)
            msgctxt = f"{base_msgctxt} [{disambiguator}]"
            suffix = 2
            while msgctxt in seen_contexts:
                msgctxt = f"{base_msgctxt} [{disambiguator}] #{suffix}"
                suffix += 1
            seen_contexts.add(msgctxt)
            msgctxt_by_id[entry.id] = msgctxt

    return msgctxt_by_id


def _extract_glossary_id(entry: polib.POEntry) -> str:
    extracted_comment = str(getattr(entry, "comment", "") or "")
    for raw_line in extracted_comment.splitlines():
        line = raw_line.strip()
        if line.startswith("ID:"):
            return _normalize_field(line[3:])
    return _normalize_field(getattr(entry, "msgctxt", ""))


def build_glossary_pot(entries: Iterable[GlossarySourceTerm]) -> polib.POFile:
    """Build a POT catalog from canonical source glossary entries."""
    sorted_entries = sorted(entries, key=_sort_key)
    msgctxt_by_id = _build_msgctxt_map(sorted_entries)
    catalog = polib.POFile()
    catalog.wrapwidth = PO_WRAP_WIDTH
    catalog.metadata = _build_glossary_metadata(source_tag="jsonl")

    for entry in sorted_entries:
        catalog.append(
            polib.POEntry(
                msgctxt=msgctxt_by_id[entry.id],
                msgid=entry.source_term,
                msgstr="",
                comment=_build_extracted_comment(entry),
            )
        )
    return catalog


def build_locale_glossary_po_from_records(
    records: Sequence[Tuple[str, str, str, str]],
    *,
    locale: str,
) -> polib.POFile:
    """Seed one locale glossary catalog directly from rich glossary records."""
    entries: list[tuple[GlossarySourceTerm, str]] = []
    catalog = polib.POFile()
    catalog.wrapwidth = PO_WRAP_WIDTH
    catalog.metadata = _build_glossary_metadata(
        source_tag="imported-glossary",
        language=locale,
    )

    for source_term, target_term, part_of_speech, context_note, sense in _iter_records_with_senses(records):
        normalized_source = _normalize_field(source_term)
        normalized_pos = _normalize_field(part_of_speech)
        normalized_context = _normalize_field(context_note) or build_fallback_context_note(
            normalized_source,
            normalized_pos,
        )
        entry = GlossarySourceTerm(
            source_term=normalized_source,
            part_of_speech=normalized_pos,
            sense=sense,
            context_note=normalized_context,
            id=build_glossary_entry_id(normalized_source, normalized_pos, sense),
        )
        entries.append((entry, _normalize_field(target_term)))

    msgctxt_by_id = _build_msgctxt_map(entry for entry, _target in entries)
    for entry, target_term in sorted(entries, key=lambda item: _sort_key(item[0])):
        catalog.append(
            polib.POEntry(
                msgctxt=msgctxt_by_id[entry.id],
                msgid=entry.source_term,
                msgstr=target_term,
                comment=_build_extracted_comment(entry),
            )
        )
    return catalog


def _iter_records_with_senses(
    records: Sequence[Tuple[str, str, str, str]],
) -> list[Tuple[str, str, str, str, str]]:
    """Assign readable, collision-safe senses to rich glossary records."""
    sorted_records = sorted(
        records,
        key=lambda item: (
            _normalize_field(item[0]).lower(),
            _normalize_field(item[2]).lower(),
            _normalize_field(item[3]).lower(),
        ),
    )
    sense_counters: dict[tuple[str, str, str], int] = {}
    results: list[Tuple[str, str, str, str, str]] = []

    for source_term, target_term, part_of_speech, context_note in sorted_records:
        normalized_source = _normalize_field(source_term)
        normalized_pos = _normalize_field(part_of_speech)
        normalized_context = _normalize_field(context_note)
        base_sense = suggest_glossary_sense(normalized_context, normalized_source)
        counter_key = (normalized_source.lower(), normalized_pos.lower(), base_sense)
        sense_index = sense_counters.get(counter_key, 0) + 1
        sense_counters[counter_key] = sense_index
        sense = base_sense if sense_index == 1 else f"{base_sense}-{sense_index}"
        results.append((source_term, target_term, part_of_speech, context_note, sense))

    return results


def sync_locale_glossary_po(
    entries: Iterable[GlossarySourceTerm],
    *,
    locale: str,
    existing_catalog: polib.POFile | None = None,
) -> polib.POFile:
    """Sync one locale glossary catalog from canonical source entries."""
    sorted_entries = sorted(entries, key=_sort_key)
    msgctxt_by_id = _build_msgctxt_map(sorted_entries)
    catalog = polib.POFile()
    catalog.wrapwidth = PO_WRAP_WIDTH
    catalog.metadata = _build_glossary_metadata(
        source_tag="jsonl",
        language=locale,
    )

    existing_by_id: dict[str, polib.POEntry] = {}
    for item in existing_catalog or []:
        glossary_id = _extract_glossary_id(item)
        if glossary_id:
            existing_by_id[glossary_id] = item

    for entry in sorted_entries:
        previous = existing_by_id.get(entry.id)
        flags = list(getattr(previous, "flags", []) or [])
        if previous is not None and _normalize_field(previous.msgid) != entry.source_term:
            if "fuzzy" not in flags:
                flags.append("fuzzy")

        catalog.append(
            polib.POEntry(
                msgctxt=msgctxt_by_id[entry.id],
                msgid=entry.source_term,
                msgstr=_normalize_field(getattr(previous, "msgstr", "")),
                comment=_build_extracted_comment(entry),
                tcomment=_normalize_field(getattr(previous, "tcomment", "")),
                flags=flags,
            )
        )

    return catalog


def write_catalog(catalog: polib.POFile, path: str | Path) -> None:
    """Persist a POT/PO catalog to disk, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.save(str(output_path))


__all__ = [
    "GlossarySourceTerm",
    "build_fallback_context_note",
    "build_glossary_entry_id",
    "build_glossary_source_terms_from_records",
    "build_glossary_pot",
    "build_locale_glossary_po_from_records",
    "load_glossary_source_terms",
    "suggest_glossary_sense",
    "sync_locale_glossary_po",
    "write_catalog",
]
