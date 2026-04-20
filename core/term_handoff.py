from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import polib

from core.formats import PO_WRAP_WIDTH
from core.term_extraction import DiscoveryMode, ExtractionResult, normalize_space, validate_max_length


@dataclass
class TermTranslationCandidate:
    """One local term candidate prepared for glossary translation handoff."""

    source_term: str
    decision: str = "accepted"
    score: int = 0
    reasons: List[str] | None = None
    contexts: List[str] | None = None
    examples: List[str] | None = None
    notes: List[str] | None = None
    file_count: int = 0
    files: List[str] | None = None
    location_files: List[str] | None = None
    location_scopes: List[str] | None = None
    known_translation: str = ""


def build_output_path(file_path: str, mode: DiscoveryMode) -> str:
    """Derive the default JSON output path for a local extraction run."""
    root, _ = os.path.splitext(file_path)
    suffix = "prototype-glossary" if mode == "all" else "prototype-missing-terms"
    return f"{root}.{suffix}.json"


def build_po_output_path(json_path: str) -> str:
    """Convert a local extraction JSON path into the default PO handoff path."""
    root, _ = os.path.splitext(json_path)
    return f"{root}.po"


def build_translation_candidate_payload(
    result: ExtractionResult,
    *,
    include_borderline: bool = False,
) -> List[Dict[str, Any]]:
    """Project extracted terms into a compact translation-ready JSON payload."""
    items = list(result.accepted_terms)
    if include_borderline:
        items.extend(result.borderline_terms)

    payload: List[Dict[str, Any]] = []
    for item in items:
        payload.append(
            {
                "source_term": item.source_term,
                "decision": item.decision,
                "score": item.score,
                "reasons": list(item.reasons),
                "contexts": list(item.contexts),
                "examples": list(item.examples),
                "notes": list(item.notes),
                "file_count": item.file_count,
                "files": list(item.files),
                "location_files": list(item.location_files),
                "location_scopes": list(item.location_scopes),
                "known_translation": item.known_translation,
            }
        )
    return payload


def build_json_payload(
    *,
    file_path: str,
    out_path: str,
    source_lang: str,
    target_lang: str | None,
    mode: DiscoveryMode,
    max_length: int,
    glossary_source_path: str | None,
    total_source_messages: int,
    result: ExtractionResult,
    include_rejected: bool,
) -> Dict[str, Any]:
    """Serialize a local extraction run into the review-friendly JSON report."""
    translation_candidates = build_translation_candidate_payload(result)
    payload: Dict[str, Any] = {
        "prototype": "local_term_extractor_v2",
        "source_file": file_path,
        "output_file": out_path,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "mode": mode,
        "max_length": validate_max_length(max_length),
        "glossary_source_path": glossary_source_path,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_source_messages": total_source_messages,
        "accepted_candidate_count": len(result.accepted_terms),
        "borderline_candidate_count": len(result.borderline_terms),
        "rejected_candidate_count": len(result.rejected_terms),
        "translation_candidate_count": len(translation_candidates),
        "terms": [asdict(item) for item in result.accepted_terms],
        "borderline_terms": [asdict(item) for item in result.borderline_terms],
        "translation_candidates": translation_candidates,
    }
    if include_rejected:
        payload["rejected_terms"] = [asdict(item) for item in result.rejected_terms]
    return payload


def coerce_translation_candidate(
    item: Dict[str, Any],
    *,
    default_decision: str,
) -> TermTranslationCandidate | None:
    """Normalize a JSON candidate object into the PO-conversion dataclass."""
    source_term = normalize_space(item.get("source_term"))
    if not source_term:
        return None

    score_raw = item.get("score", 0)
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = 0
    file_count_raw = item.get("file_count", 0)
    try:
        file_count = int(file_count_raw)
    except (TypeError, ValueError):
        file_count = 0

    return TermTranslationCandidate(
        source_term=source_term,
        decision=normalize_space(item.get("decision")) or default_decision,
        score=score,
        reasons=[normalize_space(value) for value in item.get("reasons", []) if normalize_space(value)],
        contexts=[normalize_space(value) for value in item.get("contexts", []) if normalize_space(value)],
        examples=[normalize_space(value) for value in item.get("examples", []) if normalize_space(value)],
        notes=[normalize_space(value) for value in item.get("notes", []) if normalize_space(value)],
        file_count=file_count,
        files=[normalize_space(value) for value in item.get("files", []) if normalize_space(value)],
        location_files=[
            normalize_space(value)
            for value in item.get("location_files", [])
            if normalize_space(value)
        ],
        location_scopes=[
            normalize_space(value)
            for value in item.get("location_scopes", [])
            if normalize_space(value)
        ],
        known_translation=normalize_space(item.get("known_translation")),
    )


def collect_json_translation_candidates(
    payload: Dict[str, Any],
    *,
    include_borderline: bool = False,
) -> List[TermTranslationCandidate]:
    """Load accepted and optional borderline translation candidates from JSON."""
    results: List[TermTranslationCandidate] = []
    seen: set[str] = set()

    def add_items(raw_items: Iterable[Dict[str, Any]], *, default_decision: str) -> None:
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            candidate = coerce_translation_candidate(raw_item, default_decision=default_decision)
            if candidate is None:
                continue
            key = candidate.source_term.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(candidate)

    add_items(payload.get("translation_candidates", []), default_decision="accepted")
    if include_borderline:
        add_items(payload.get("borderline_terms", []), default_decision="borderline")
    return results


def join_contexts(contexts: Sequence[str] | None, *, limit: int = 3) -> str | None:
    """Join a few candidate contexts into one PO msgctxt string."""
    values = [normalize_space(value) for value in contexts or [] if normalize_space(value)]
    if not values:
        return None
    return " | ".join(values[:limit])


def parse_occurrence(location: str) -> Tuple[str, str] | None:
    """Split a normalized file:line location into a PO occurrence tuple."""
    normalized = normalize_space(location)
    if not normalized or ":" not in normalized:
        return None
    file_path, line = normalized.rsplit(":", 1)
    file_path = normalize_space(file_path.replace("\\", "/"))
    line = normalize_space(line)
    if not file_path or not line:
        return None
    return file_path, line


def build_translation_entry_note(candidate: TermTranslationCandidate) -> str | None:
    """Build a compact translator note from local extraction evidence."""
    lines: List[str] = []
    if candidate.decision:
        lines.append(f"Decision: {candidate.decision}")
    lines.append(f"Score: {candidate.score}")
    if candidate.reasons:
        lines.append(f"Reasons: {', '.join(candidate.reasons)}")
    if candidate.examples:
        lines.append(f"Example: {candidate.examples[0]}")
    if candidate.notes:
        lines.extend(f"Note: {note}" for note in candidate.notes[:3])
    if candidate.files:
        lines.append(f"Files: {', '.join(candidate.files[:5])}")
    if candidate.location_scopes:
        lines.append(f"Location scopes: {', '.join(candidate.location_scopes[:5])}")
    if candidate.known_translation:
        lines.append(f"Known translation: {candidate.known_translation}")
    return "\n".join(lines) if lines else None


def build_po_from_translation_candidates(
    candidates: Sequence[TermTranslationCandidate],
    *,
    source_file: str,
    source_lang: str,
    target_lang: str | None,
) -> polib.POFile:
    """Build a glossary-style PO file for later translation."""
    po = polib.POFile()
    po.wrapwidth = PO_WRAP_WIDTH
    po.metadata = {
        "Project-Id-Version": "Prototype Glossary",
        "X-Source-Language": source_lang,
        "Content-Type": "text/plain; charset=utf-8",
        "MIME-Version": "1.0",
        "Generated-By": "extract_terms_local.py",
        "X-Prototype-Source": source_file,
    }
    if target_lang:
        po.metadata["Language"] = target_lang

    for candidate in candidates:
        entry = polib.POEntry(msgid=candidate.source_term, msgstr="")
        context = join_contexts(candidate.contexts)
        if context:
            entry.msgctxt = context
        note = build_translation_entry_note(candidate)
        if note:
            entry.tcomment = note

        occurrences = [
            occurrence
            for occurrence in (
                parse_occurrence(value) for value in (candidate.location_files or [])
            )
            if occurrence is not None
        ]
        if occurrences:
            entry.occurrences = occurrences

        po.append(entry)

    return po


def convert_json_to_po(
    json_path: str,
    *,
    out_path: str | None = None,
    include_borderline: bool = False,
) -> str:
    """Convert a local extraction JSON report into a translation-ready PO glossary."""
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Prototype JSON payload must be an object.")

    final_out_path = out_path or build_po_output_path(json_path)
    candidates = collect_json_translation_candidates(
        payload,
        include_borderline=include_borderline,
    )
    if not candidates:
        raise ValueError("No translation candidates found in prototype JSON.")

    po = build_po_from_translation_candidates(
        candidates,
        source_file=json_path,
        source_lang=normalize_space(payload.get("source_lang")) or "en",
        target_lang=normalize_space(payload.get("target_lang")) or None,
    )
    po.save(final_out_path)
    return final_out_path


__all__ = [
    "TermTranslationCandidate",
    "build_json_payload",
    "build_output_path",
    "build_po_from_translation_candidates",
    "build_po_output_path",
    "build_translation_candidate_payload",
    "build_translation_entry_note",
    "collect_json_translation_candidates",
    "convert_json_to_po",
    "coerce_translation_candidate",
    "join_contexts",
    "parse_occurrence",
]
