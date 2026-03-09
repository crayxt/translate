#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

import polib
from google import genai
from google.genai import types as genai_types

from process import (
    FileKind,
    detect_file_kind,
    generate_with_retry,
    load_po,
    load_resx,
    load_strings,
    load_txt,
    load_ts,
    read_optional_text_file,
    resolve_resource_path,
    resolve_runtime_limits,
)


DiscoveryMode = Literal["all", "missing"]
OutputFormat = Literal["po", "json"]


TERM_DISCOVERY_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "terms": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "source_term": genai_types.Schema(type=genai_types.Type.STRING),
                    "suggested_translation": genai_types.Schema(type=genai_types.Type.STRING),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                    "example_source": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["source_term", "suggested_translation", "reason"],
            ),
        )
    },
    required=["terms"],
)


@dataclass
class TermCandidate:
    source_term: str
    suggested_translation: str
    reason: str
    example_source: str = ""


def build_term_output_path(
    file_path: str,
    output_format: OutputFormat = "po",
    mode: DiscoveryMode = "all",
) -> str:
    root, _ = os.path.splitext(file_path)
    suffix = "glossary" if mode == "all" else "missing-terms"
    return f"{root}.{suffix}.{output_format}"


def should_include_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalpha() for ch in stripped)


def collect_source_messages(entries: List[Any]) -> List[str]:
    results: List[str] = []
    seen: set[str] = set()

    def add_message(text: str) -> None:
        normalized = " ".join(text.split())
        if not should_include_text(normalized):
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        results.append(normalized)

    for entry in entries:
        if getattr(entry, "obsolete", False):
            continue
        if not getattr(entry, "include_in_term_extraction", True):
            continue
        add_message(getattr(entry, "msgid", "") or "")
        plural_text = getattr(entry, "msgid_plural", None)
        if plural_text:
            add_message(plural_text)

    return results


def build_terms_prompt(
    messages: Dict[str, str],
    source_lang: str,
    target_lang: str,
    mode: DiscoveryMode,
    vocabulary: str | None,
    max_terms_per_batch: int,
) -> str:
    if vocabulary:
        vocab_block = vocabulary
    else:
        vocab_block = "(No vocabulary provided)"

    if mode == "all":
        task_block = """
Task:
- Inspect source UI messages and extract a comprehensive project glossary.
- Focus on product/domain/UI terms that should be standardized.
- Also extract generic IT terminology as we are updating our glossary.
- Exclude generic words and stop words.
- If vocabulary is provided, use it for consistency but do not limit extraction to only missing terms.
"""
    else:
        task_block = """
Task:
- Inspect source UI messages and find terms that are missing from the provided project vocabulary.
- Focus on product/domain/UI terms that should be standardized in a glossary.
- Exclude generic words, stop words, and terms already present in the vocabulary.
"""
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    return f"""
You are a software localization terminology analyst.

{task_block}

Output requirements:
- Return ONLY valid JSON (no markdown).
- Use this schema exactly: {{"terms": [{{"source_term": "...", "suggested_translation": "...", "reason": "...", "example_source": "..."}}]}}
- Keep "source_term" concise and canonical.
- Keep source term and translation lowercase, in singular form.
- Provide at most {max_terms_per_batch} terms for this batch.

Project context:
- Source language: {source_lang}
- Target language: {target_lang}

Known vocabulary:
{vocab_block}

Messages (id -> source text):
{messages_json}
"""


def _json_load_maybe(text: str) -> Any:
    payload = text.strip()
    if not payload:
        return None

    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def parse_term_response(response_payload: Any) -> List[TermCandidate]:
    if isinstance(response_payload, dict):
        payload = response_payload
    else:
        payload = getattr(response_payload, "parsed", None)
        if payload is None:
            text_payload = getattr(response_payload, "text", None) or ""
            payload = _json_load_maybe(text_payload)

    if not isinstance(payload, dict):
        return []

    terms = payload.get("terms")
    if not isinstance(terms, list):
        return []

    results: List[TermCandidate] = []
    for item in terms:
        if not isinstance(item, dict):
            continue
        source_term = str(item.get("source_term", "")).strip()
        suggested = str(item.get("suggested_translation", "")).strip()
        reason = str(item.get("reason", "")).strip()
        example = str(item.get("example_source", "")).strip()
        if not source_term or not suggested or not reason:
            continue
        results.append(
            TermCandidate(
                source_term=source_term,
                suggested_translation=suggested,
                reason=reason,
                example_source=example,
            )
        )

    return results


def merge_term_candidates(candidates: List[TermCandidate]) -> List[TermCandidate]:
    merged: Dict[str, TermCandidate] = {}
    order: List[str] = []

    for item in candidates:
        key = " ".join(item.source_term.lower().split())
        if not key:
            continue
        if key not in merged:
            merged[key] = item
            order.append(key)
            continue

        existing = merged[key]
        if not existing.suggested_translation and item.suggested_translation:
            existing.suggested_translation = item.suggested_translation
        if not existing.reason and item.reason:
            existing.reason = item.reason
        if not existing.example_source and item.example_source:
            existing.example_source = item.example_source

    return [merged[k] for k in order]


def save_terms_as_po(
    terms: List[TermCandidate],
    out_path: str,
    source_lang: str,
    target_lang: str,
) -> None:
    po = polib.POFile()
    po.metadata = {
        "Project-Id-Version": "Glossary",
        "Language": target_lang,
        "X-Source-Language": source_lang,
        "Content-Type": "text/plain; charset=utf-8",
        "MIME-Version": "1.0",
        "Generated-By": "extract_terms.py",
    }

    for item in terms:
        entry = polib.POEntry(
            msgid=item.source_term,
            msgstr=item.suggested_translation,
            flags=["fuzzy"],
        )

        notes: List[str] = []
        if item.reason:
            notes.append(f"Reason: {item.reason}")
        if item.example_source:
            notes.append(f"Example: {item.example_source}")
        if notes:
            entry.tcomment = "\n".join(notes)

        po.append(entry)

    po.save(out_path)


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


def normalize_limits(
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
) -> Tuple[int, int, str]:
    if batch_size_arg is None and parallel_arg is None:
        batch_size, parallel, _ = resolve_runtime_limits(
            total_items=total_items,
            batch_size_arg=250,
            parallel_arg=6,
        )
        return batch_size, parallel, "term defaults"

    return resolve_runtime_limits(
        total_items=total_items,
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract glossary term candidates from PO/TS/RESX/STRINGS/TXT files using Gemini "
            "and save as PO or JSON"
        )
    )
    parser.add_argument("file", help="Input .po, .ts, .resx, .strings, or .txt file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent Gemini requests (auto if omitted)")
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "missing"],
        default="missing",
        help="all: build full glossary; missing: only terms missing from vocabulary",
    )
    parser.add_argument(
        "--out-format",
        choices=["po", "json"],
        default="po",
        help="Output format (default: po)",
    )
    parser.add_argument("--out", default=None, help="Output path (default depends on --mode and --out-format)")
    parser.add_argument("--max-terms-per-batch", type=int, default=80, help="Max term suggestions requested per batch")
    parser.add_argument("--max-attempts", type=int, default=5, help="Retry attempts per batch")
    args = parser.parse_args()

    if args.max_terms_per_batch <= 0:
        sys.exit("ERROR: --max-terms-per-batch must be greater than 0")
    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    vocabulary_path = resolve_resource_path(
        explicit_path=args.vocab,
        prefix="vocab",
        extension="txt",
        target_lang=args.target_lang,
    )
    vocabulary_text = read_optional_text_file(vocabulary_path, "Vocabulary")
    vocab_source = f"file:{vocabulary_path}" if vocabulary_text and vocabulary_path else "none"

    try:
        file_kind = detect_file_kind(args.file)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    entries = load_entries_for_file(args.file, file_kind)
    source_messages = collect_source_messages(entries)

    total = len(source_messages)
    if total == 0:
        print("No source messages found for terminology extraction.")
        return

    try:
        batch_size, parallel_requests, limits_mode = normalize_limits(
            total_items=total,
            batch_size_arg=args.batch_size,
            parallel_arg=args.parallel_requests,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    batches = [
        source_messages[i * batch_size: (i + 1) * batch_size]
        for i in range((total + batch_size - 1) // batch_size)
    ]

    out_path = args.out or build_term_output_path(
        args.file,
        output_format=args.out_format,
        mode=args.mode,
    )
    term_config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TERM_DISCOVERY_SCHEMA,
    )

    print("Startup configuration:")
    print(f"  Model: {args.model}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")
    print(f"  Discovery mode: {args.mode}")
    print(f"  Output format: {args.out_format}")
    print(f"  Vocabulary source: {vocab_source}")
    print(f"  Total source messages: {total}")
    print(f"  Total batches: {len(batches)}")

    async def process_batch(batch_index: int, batch: List[str], sem: asyncio.Semaphore):
        async with sem:
            msg_map = {str(i): text for i, text in enumerate(batch)}
            prompt = build_terms_prompt(
                messages=msg_map,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                mode=args.mode,
                vocabulary=vocabulary_text,
                max_terms_per_batch=args.max_terms_per_batch,
            )
            response = await generate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                batch_label=f"terms batch {batch_index + 1}/{len(batches)}",
                max_attempts=args.max_attempts,
                config=term_config,
            )
            return batch_index, parse_term_response(response)

    async def run_extraction() -> List[TermCandidate]:
        sem = asyncio.Semaphore(parallel_requests)
        tasks = [
            asyncio.create_task(process_batch(i, batch, sem))
            for i, batch in enumerate(batches)
        ]

        all_candidates: List[TermCandidate] = []
        completed = 0
        for finished in asyncio.as_completed(tasks):
            batch_index, terms = await finished
            all_candidates.extend(terms)
            completed += 1
            print(
                f"Progress: completed batches {completed}/{len(batches)} "
                f"(latest: {batch_index + 1}/{len(batches)}), "
                f"raw terms collected: {len(all_candidates)}"
            )

        return all_candidates

    try:
        raw_candidates = asyncio.run(run_extraction())
    except RuntimeError as e:
        sys.exit(str(e))

    merged_terms = merge_term_candidates(raw_candidates)
    merged_terms.sort(key=lambda x: x.source_term.lower())

    payload = {
        "source_file": args.file,
        "output_file": out_path,
        "discovery_mode": args.mode,
        "output_format": args.out_format,
        "model": args.model,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "vocabulary_source": vocab_source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_source_messages": total,
        "raw_candidate_count": len(raw_candidates),
        "deduped_candidate_count": len(merged_terms),
        "terms": [asdict(item) for item in merged_terms],
    }

    if args.out_format == "po":
        save_terms_as_po(
            terms=merged_terms,
            out_path=out_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nTerm extraction complete.")
    print(f"Saved file: {out_path}")
    print(f"Candidate terms: {len(merged_terms)} (from {len(raw_candidates)} raw suggestions)")


if __name__ == "__main__":
    main()
