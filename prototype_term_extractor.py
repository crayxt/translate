#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Tuple

from core.entries import get_entry_prompt_context_and_note
from core.formats import (
    FileKind,
    detect_file_kind,
    load_po,
    load_resx,
    load_strings,
    load_ts,
    load_txt,
)
from core.resources import load_vocabulary_pairs, resolve_resource_path


DiscoveryMode = Literal["all", "missing"]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
STRUCTURAL_CONTEXT_RE = re.compile(r"^line:\d+$", re.IGNORECASE)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
    "without",
}

LOW_VALUE_SINGLE_WORDS = STOP_WORDS | {
    "available",
    "current",
    "currently",
    "default",
    "disabled",
    "enable",
    "enabled",
    "failed",
    "invalid",
    "new",
    "old",
    "please",
    "previous",
    "selected",
}

FIXED_MULTIWORD_ALLOWLIST = {
    "access token",
    "command line",
    "dark mode",
    "file system",
    "light mode",
    "log in",
    "log out",
    "menu bar",
    "side panel",
    "sign in",
    "sign out",
    "status bar",
    "tool bar",
    "user interface",
}


@dataclass
class SourceMessage:
    source: str
    context: str = ""
    note: str = ""


@dataclass
class CandidateEvidence:
    source_term: str
    occurrence_count: int = 0
    message_count: int = 0
    exact_source_match_count: int = 0
    examples: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    known_translation: str = ""
    accepted: bool = False
    score: int = 0
    reasons: List[str] = field(default_factory=list)


@dataclass
class PrototypeExtractionResult:
    accepted_terms: List[CandidateEvidence]
    rejected_terms: List[CandidateEvidence]


def normalize_space(text: str | None) -> str:
    return " ".join(str(text or "").split())


def normalize_candidate_key(text: str | None) -> str:
    cleaned = normalize_space(text).strip(".,:;!?\"'()[]{}")
    return cleaned.lower()


def build_output_path(
    file_path: str,
    mode: DiscoveryMode,
) -> str:
    root, _ = os.path.splitext(file_path)
    suffix = "prototype-glossary" if mode == "all" else "prototype-missing-terms"
    return f"{root}.{suffix}.json"


def load_entries_for_file(file_path: str, file_kind: FileKind) -> List[Any]:
    if file_kind == FileKind.TS:
        entries, _, _ = load_ts(file_path)
        return entries
    if file_kind == FileKind.RESX:
        entries, _, _ = load_resx(file_path)
        return entries
    if file_kind == FileKind.STRINGS:
        entries, _, _ = load_strings(file_path)
        return entries
    if file_kind == FileKind.TXT:
        entries, _, _ = load_txt(file_path)
        return entries
    entries, _, _ = load_po(file_path)
    return entries


def should_include_source_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalpha() for ch in stripped)


def collect_source_messages(entries: Iterable[Any]) -> List[SourceMessage]:
    results: List[SourceMessage] = []
    seen: set[Tuple[str, str, str]] = set()

    def add_message(text: str, *, context: str | None, note: str | None) -> None:
        normalized = normalize_space(text)
        if not should_include_source_text(normalized):
            return
        normalized_context = normalize_space(context)
        normalized_note = normalize_space(note)
        key = (
            normalized.lower(),
            normalized_context.lower(),
            normalized_note.lower(),
        )
        if key in seen:
            return
        seen.add(key)
        results.append(
            SourceMessage(
                source=normalized,
                context=normalized_context,
                note=normalized_note,
            )
        )

    for entry in entries:
        if getattr(entry, "obsolete", False):
            continue
        if not getattr(entry, "include_in_term_extraction", True):
            continue
        context, note = get_entry_prompt_context_and_note(entry)
        add_message(getattr(entry, "msgid", "") or "", context=context, note=note)
        plural_text = getattr(entry, "msgid_plural", None)
        if plural_text:
            add_message(plural_text, context=context, note=note)

    return results


def tokenize_source_text(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def is_placeholder_like(term: str) -> bool:
    if not term:
        return True
    if "%" in term or "{" in term or "}" in term or "<" in term or ">" in term:
        return True
    if "&" in term:
        return True
    return False


def is_valid_single_token(token: str) -> bool:
    if not token:
        return False
    if token in LOW_VALUE_SINGLE_WORDS:
        return False
    if token.isdigit():
        return False
    if len(token) <= 1:
        return False
    return not is_placeholder_like(token)


def is_valid_phrase_tokens(tokens: List[str]) -> bool:
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    phrase = " ".join(tokens)
    if phrase in FIXED_MULTIWORD_ALLOWLIST:
        return True
    if tokens[0] in STOP_WORDS or tokens[-1] in STOP_WORDS:
        return False
    return all(is_valid_single_token(token) for token in tokens)


def extract_candidate_counts(source_text: str) -> Counter[str]:
    tokens = tokenize_source_text(source_text)
    counts: Counter[str] = Counter()

    for token in tokens:
        if is_valid_single_token(token):
            counts[token] += 1

    for size in (2, 3):
        for index in range(0, len(tokens) - size + 1):
            phrase_tokens = tokens[index : index + size]
            if not is_valid_phrase_tokens(phrase_tokens):
                continue
            counts[" ".join(phrase_tokens)] += 1

    return counts


def is_meaningful_context(context: str) -> bool:
    normalized = normalize_space(context)
    if not normalized:
        return False
    return STRUCTURAL_CONTEXT_RE.fullmatch(normalized) is None


def add_unique_limited(values: List[str], value: str, limit: int = 3) -> None:
    cleaned = normalize_space(value)
    if not cleaned or cleaned in values:
        return
    if len(values) >= limit:
        return
    values.append(cleaned)


def collect_candidate_evidence(
    messages: List[SourceMessage],
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> Dict[str, CandidateEvidence]:
    vocabulary_map = {
        normalize_candidate_key(source): normalize_space(target)
        for source, target in vocabulary_pairs or []
        if normalize_candidate_key(source) and normalize_space(target)
    }
    evidence: Dict[str, CandidateEvidence] = {}

    for message in messages:
        candidate_counts = extract_candidate_counts(message.source)
        source_tokens = tokenize_source_text(message.source)
        exact_candidate = " ".join(source_tokens) if source_tokens else ""
        seen_terms_in_message: set[str] = set()

        for term, count in candidate_counts.items():
            item = evidence.setdefault(
                term,
                CandidateEvidence(
                    source_term=term,
                    known_translation=vocabulary_map.get(term, ""),
                ),
            )
            item.occurrence_count += count
            if term not in seen_terms_in_message:
                item.message_count += 1
                seen_terms_in_message.add(term)
            add_unique_limited(item.examples, message.source)
            if is_meaningful_context(message.context):
                add_unique_limited(item.contexts, message.context)
            if message.note:
                add_unique_limited(item.notes, message.note)

        if exact_candidate and exact_candidate in candidate_counts:
            evidence[exact_candidate].exact_source_match_count += 1

    return evidence


def decide_candidate(
    item: CandidateEvidence,
    *,
    mode: DiscoveryMode,
    vocabulary_keys: set[str],
) -> CandidateEvidence:
    decided = CandidateEvidence(
        source_term=item.source_term,
        occurrence_count=item.occurrence_count,
        message_count=item.message_count,
        exact_source_match_count=item.exact_source_match_count,
        examples=list(item.examples),
        contexts=list(item.contexts),
        notes=list(item.notes),
        known_translation=item.known_translation,
    )

    tokens = item.source_term.split()
    if not tokens or is_placeholder_like(item.source_term):
        decided.reasons.append("placeholder_or_empty")
        return decided

    if mode == "missing" and item.source_term in vocabulary_keys:
        decided.reasons.append("already_in_vocabulary")
        return decided

    if len(tokens) == 1:
        token = tokens[0]
        if token in LOW_VALUE_SINGLE_WORDS:
            decided.reasons.append("low_value_single_word")
            return decided

        if item.exact_source_match_count > 0:
            decided.score += 3
            decided.reasons.append("exact_ui_label")
        if item.message_count > 1:
            decided.score += 2
            decided.reasons.append("repeated_across_messages")
        if len(token) >= 4:
            decided.score += 1
            decided.reasons.append("content_word_candidate")
        if item.contexts:
            decided.score += 1
            decided.reasons.append("has_meaningful_context")

        decided.accepted = decided.score >= 1
        if not decided.accepted:
            decided.reasons.append("insufficient_single_word_evidence")
        return decided

    if item.source_term in FIXED_MULTIWORD_ALLOWLIST:
        decided.score += 4
        decided.reasons.append("fixed_multiword_allowlist")
    if item.message_count > 1:
        decided.score += 2
        decided.reasons.append("repeated_phrase_across_messages")
    if item.exact_source_match_count > 1:
        decided.score += 1
        decided.reasons.append("repeated_exact_label")
    if item.contexts:
        decided.score += 1
        decided.reasons.append("has_meaningful_context")

    decided.accepted = decided.score >= 3
    if not decided.accepted:
        decided.reasons.append("single_occurrence_multiword_phrase")
    return decided


def sort_candidates(items: Iterable[CandidateEvidence]) -> List[CandidateEvidence]:
    return sorted(
        items,
        key=lambda item: (
            -item.score,
            -item.message_count,
            -item.occurrence_count,
            item.source_term,
        ),
    )


def extract_terms_locally(
    messages: List[SourceMessage],
    *,
    mode: DiscoveryMode = "missing",
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> PrototypeExtractionResult:
    evidence = collect_candidate_evidence(messages, vocabulary_pairs=vocabulary_pairs)
    vocabulary_keys = {
        normalize_candidate_key(source_term)
        for source_term, _ in vocabulary_pairs or []
        if normalize_candidate_key(source_term)
    }

    accepted: List[CandidateEvidence] = []
    rejected: List[CandidateEvidence] = []
    for key, item in evidence.items():
        decided = decide_candidate(item, mode=mode, vocabulary_keys=vocabulary_keys)
        if decided.accepted:
            accepted.append(decided)
        else:
            rejected.append(decided)

    return PrototypeExtractionResult(
        accepted_terms=sort_candidates(accepted),
        rejected_terms=sort_candidates(rejected),
    )


def build_json_payload(
    *,
    file_path: str,
    out_path: str,
    source_lang: str,
    target_lang: str,
    mode: DiscoveryMode,
    vocabulary_path: str | None,
    messages: List[SourceMessage],
    result: PrototypeExtractionResult,
    include_rejected: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prototype": "local_term_extractor_v1",
        "source_file": file_path,
        "output_file": out_path,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "mode": mode,
        "vocabulary_path": vocabulary_path,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_source_messages": len(messages),
        "accepted_candidate_count": len(result.accepted_terms),
        "rejected_candidate_count": len(result.rejected_terms),
        "terms": [asdict(item) for item in result.accepted_terms],
    }
    if include_rejected:
        payload["rejected_terms"] = [asdict(item) for item in result.rejected_terms]
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype local terminology extractor. "
            "Uses contextual source messages plus conservative local filtering; no model calls."
        )
    )
    parser.add_argument("file", help="Input .po, .ts, .resx, .strings, or .txt file")
    parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    parser.add_argument("--target-lang", default="kk", help="Target language code (default: kk)")
    parser.add_argument(
        "--mode",
        choices=["all", "missing"],
        default="missing",
        help="all: keep all local candidates; missing: drop terms already in vocabulary",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt). Supports .txt and glossary .po",
    )
    parser.add_argument("--out", default=None, help="Output path (default: <input>.prototype-*.json)")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected candidates in the JSON output for debugging",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        file_kind = detect_file_kind(args.file)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    vocabulary_path = resolve_resource_path(
        explicit_path=args.vocab,
        prefix="vocab",
        extension="txt",
        target_lang=args.target_lang,
    )
    vocabulary_pairs = load_vocabulary_pairs(vocabulary_path, "Vocabulary")

    entries = load_entries_for_file(args.file, file_kind)
    messages = collect_source_messages(entries)
    if not messages:
        print("No source messages found for terminology extraction.")
        return

    out_path = args.out or build_output_path(args.file, args.mode)
    result = extract_terms_locally(
        messages,
        mode=args.mode,
        vocabulary_pairs=vocabulary_pairs,
    )
    payload = build_json_payload(
        file_path=args.file,
        out_path=out_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        mode=args.mode,
        vocabulary_path=vocabulary_path,
        messages=messages,
        result=result,
        include_rejected=args.include_rejected,
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("Prototype term extraction complete.")
    print(f"Saved file: {out_path}")
    print(f"Accepted terms: {len(result.accepted_terms)}")
    print(f"Rejected terms: {len(result.rejected_terms)}")


if __name__ == "__main__":
    main()
