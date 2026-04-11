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

from core.review_common import build_target_script_guidance as build_shared_target_script_guidance
from core.formats import (
    FileKind,
    PO_WRAP_WIDTH,
    UnifiedEntry,
    detect_file_kind,
    load_android_xml,
    load_xliff,
    load_po,
    load_resx,
    load_strings,
    load_txt,
    load_ts,
)
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, TranslationProvider, get_translation_provider
from core.request_contents import TaskRequestSpec, build_task_request_contents, render_text_fallback_prompt
from core.task_cli import (
    add_language_arguments,
    add_max_attempts_argument,
    add_provider_arguments,
    add_runtime_limit_arguments,
    add_vocabulary_argument,
    build_task_parser,
    resolve_provider_model,
    run_task_main,
)
from core.resources import load_vocabulary_pairs, read_optional_vocabulary_file, resolve_resource_path
from core.runtime import DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL_REQUESTS, resolve_runtime_limits
from core.task_batches import build_fixed_batches, build_indexed_batch_map, run_model_batches
from core.task_runtime import build_task_runtime_context, print_startup_configuration
from core.term_extraction import collect_source_messages as collect_shared_source_messages


DiscoveryMode = Literal["all", "missing"]
OutputFormat = Literal["po", "json"]

TERM_SYSTEM_INSTRUCTION = """
You are a software localization terminology analyst.

MUST:
- Extract canonical product, domain, and UI terms rather than full-sentence translations
- Prefer concise glossary-ready source terms and concise target-language equivalents
- Prefer atomic reusable terms over message-specific phrases
- If a phrase is only a loose modifier+noun or verb+object combination, split it into standalone terms when each part is reusable on its own
- Example: prefer `audio` and `channel` over `audio channel`; prefer `save` and `file` over `save file`
- Keep a multi-word source term only when it is a fixed concept whose meaning would change if split
- Example: keep `access token`, `command line`, or `dark mode` as whole terms
- Skip generic stop words, obvious function words, and low-value noise
- Keep term casing and number canonical unless the source clearly requires otherwise
- When vocabulary is provided, use it for consistency and do not invent conflicting alternatives
"""


TERM_DISCOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_term": {"type": "string"},
                    "suggested_translation": {"type": "string"},
                    "reason": {"type": "string"},
                    "example_source": {"type": "string"},
                },
                "required": ["source_term", "suggested_translation", "reason"],
            },
        }
    },
    "required": ["terms"],
}


@dataclass
class TermCandidate:
    """One glossary candidate returned by the model-driven extractor."""
    source_term: str
    suggested_translation: str
    reason: str
    example_source: str = ""


def build_term_output_path(
    file_path: str,
    output_format: OutputFormat = "po",
    mode: DiscoveryMode = "all",
) -> str:
    """Build the default glossary output path for the selected mode and format."""
    root, _ = os.path.splitext(file_path)
    suffix = "glossary" if mode == "all" else "missing-terms"
    return f"{root}.{suffix}.{output_format}"


def collect_source_messages(entries: List[UnifiedEntry]) -> List[Dict[str, str]]:
    """Project entries through the shared extractor and preserve the existing payload shape."""
    results: List[Dict[str, str]] = []
    for item in collect_shared_source_messages(entries):
        payload: Dict[str, str] = {"source": item.source}
        if item.context:
            payload["context"] = item.context
        if item.note:
            payload["note"] = item.note
        results.append(payload)
    return results


def build_term_request_spec(mode: DiscoveryMode, max_terms_per_batch: int) -> TaskRequestSpec:
    """Describe the model contract for glossary-term extraction batches."""
    task_lines: tuple[str, ...]
    if mode == "all":
        task_lines = (
            "Inspect source UI messages and extract a comprehensive project glossary.",
            "Focus on product, domain, and UI terms that should be standardized.",
            "Also extract generic IT terminology as we are updating the glossary.",
            "Prefer atomic reusable terms over message-specific collocations.",
            "Exclude generic words and stop words.",
            "If vocabulary is provided, use it for consistency but do not limit extraction to only missing terms.",
        )
    else:
        task_lines = (
            "Inspect source UI messages and find terms that are missing from the provided project vocabulary.",
            "Focus on product, domain, and UI terms that should be standardized in a glossary.",
            "Prefer atomic reusable terms over message-specific collocations.",
            "Exclude generic words, stop words, and terms already present in the vocabulary.",
        )

    return TaskRequestSpec(
        task_intro="Extract glossary term candidates from software localization source messages.",
        task_lines=task_lines,
        payload_lines=(
            "The payload contains source language, target language, discovery mode, optional known vocabulary, per-batch term limit, and a `messages` map.",
            "Each message item may include `source`, `context`, and `note`.",
            "Inspect only the provided source UI messages.",
        ),
        output_lines=(
            "Return only valid JSON, with no markdown or commentary.",
            "Return glossary-ready terms, not full-sentence translations.",
            "Keep `source_term` concise and canonical.",
            "Default to the smallest standalone reusable term; prefer one-word entries when they preserve the meaning.",
            "Do not return loose UI phrases like `audio channel` when `audio` and `channel` should be separate glossary entries.",
            "Keep a multi-word term only for fixed concepts such as `access token`, `command line`, or `dark mode`.",
            "Keep source terms and suggested translations lowercase and singular where natural.",
            f"Provide at most {max_terms_per_batch} terms for this batch.",
            "If vocabulary is provided, use it for consistency and avoid conflicting alternatives.",
        ),
    )


def build_terms_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    mode: DiscoveryMode,
    vocabulary: str | None,
    max_terms_per_batch: int,
) -> str:
    """Render the plain-text fallback prompt for one extraction batch."""
    return render_text_fallback_prompt(
        task_spec=build_term_request_spec(mode=mode, max_terms_per_batch=max_terms_per_batch),
        payload=build_term_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,
            vocabulary=vocabulary,
            max_terms_per_batch=max_terms_per_batch,
        ),
    )


def _json_load_maybe(text: str) -> Any:
    """Parse raw or fenced JSON model output, returning None on failure."""
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
    """Normalize a provider response into validated term candidates."""
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
    """Deduplicate extracted terms by normalized source text while keeping first-seen order."""
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


def build_term_generation_config(
    thinking_level: str | None = None,
    *,
    provider: TranslationProvider = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
    flex_mode: bool = False,
) -> Any:
    """Build the provider generation config for glossary extraction."""
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=TERM_DISCOVERY_SCHEMA,
        system_instruction=TERM_SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
        flex_mode=flex_mode,
    )


def build_term_system_instruction(target_lang: str) -> str:
    """Augment the base terminology instruction with target-script guidance."""
    parts = [TERM_SYSTEM_INSTRUCTION.strip()]
    script_guidance = build_shared_target_script_guidance(
        target_lang,
        update_wording=lambda: "suggested translations",
    )
    if script_guidance:
        parts.append(f"- {script_guidance}")
    return "\n\n".join(parts)


def build_term_request_payload(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    mode: DiscoveryMode,
    vocabulary: str | None,
    max_terms_per_batch: int,
) -> dict[str, Any]:
    """Build the structured payload sent to the glossary extractor."""
    return {
        "project_type": "software_ui_term_extraction",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "mode": mode,
        "vocabulary": vocabulary,
        "max_terms_per_batch": max_terms_per_batch,
        "messages": messages,
    }


def build_term_request_contents(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    mode: DiscoveryMode,
    vocabulary: str | None,
    max_terms_per_batch: int,
    *,
    provider: TranslationProvider = DEFAULT_PROVIDER,
) -> Any:
    """Build provider-native request contents for a glossary extraction batch."""
    return build_task_request_contents(
        provider=provider,
        task_spec=build_term_request_spec(mode=mode, max_terms_per_batch=max_terms_per_batch),
        function_name="term_extraction_batch",
        payload=build_term_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,
            vocabulary=vocabulary,
            max_terms_per_batch=max_terms_per_batch,
        ),
    )


def save_terms_as_po(
    terms: List[TermCandidate],
    out_path: str,
    source_lang: str,
    target_lang: str,
    base_vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> None:
    """Write extracted glossary candidates to a fuzzy PO handoff file."""
    po = polib.POFile()
    po.wrapwidth = PO_WRAP_WIDTH
    po.metadata = {
        "Project-Id-Version": "Glossary",
        "Language": target_lang,
        "X-Source-Language": source_lang,
        "Content-Type": "text/plain; charset=utf-8",
        "MIME-Version": "1.0",
        "Generated-By": "extract_terms.py",
    }

    seen_terms: set[str] = set()

    def remember_key(text: str) -> str:
        return " ".join(str(text or "").split()).lower()

    for source_term, target_term in base_vocabulary_pairs or []:
        key = remember_key(source_term)
        if not key or key in seen_terms:
            continue
        po.append(
            polib.POEntry(
                msgid=source_term,
                msgstr=target_term,
            )
        )
        seen_terms.add(key)

    for item in terms:
        key = remember_key(item.source_term)
        if not key or key in seen_terms:
            continue
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
        seen_terms.add(key)

    po.save(out_path)


def load_entries_for_file(file_path: str, file_kind: FileKind) -> List[UnifiedEntry]:
    """Load source entries for any format supported by model-based term extraction."""
    if file_kind == FileKind.ANDROID_XML:
        entries, _, _ = load_android_xml(file_path)
        return entries
    if file_kind == FileKind.XLIFF:
        entries, _, _ = load_xliff(file_path)
        return entries
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
    """Resolve extraction batch limits, applying task defaults when omitted."""
    if batch_size_arg is None and parallel_arg is None:
        batch_size, parallel, _ = resolve_runtime_limits(
            total_items=total_items,
            batch_size_arg=DEFAULT_BATCH_SIZE,
            parallel_arg=DEFAULT_PARALLEL_REQUESTS,
        )
        return batch_size, parallel, "term defaults"

    return resolve_runtime_limits(
        total_items=total_items,
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
    )


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure the standalone CLI for model-based glossary extraction."""
    parser.description = (
        "Extract glossary term candidates from PO/XLIFF/TS/RESX/STRINGS/TXT/Android XML files using the configured provider "
        "and save as PO or JSON"
    )
    parser.add_argument("file", help="Input .po, .xlf/.xliff, .ts, .resx, .strings, .txt, or Android .xml file")
    add_language_arguments(parser)
    add_provider_arguments(
        parser,
        default_provider_name=DEFAULT_PROVIDER_NAME,
        default_model=DEFAULT_PROVIDER.default_model,
    )
    add_runtime_limit_arguments(parser)
    add_vocabulary_argument(parser)
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
    add_max_attempts_argument(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for model-based glossary extraction."""
    return build_task_parser(configure_parser)


def run_from_args(args: argparse.Namespace) -> None:
    """Execute model-based glossary extraction from parsed CLI arguments."""
    model_name = resolve_provider_model(args.provider, args.model)

    if args.max_terms_per_batch <= 0:
        sys.exit("ERROR: --max-terms-per-batch must be greater than 0")
    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

    runtime_context = build_task_runtime_context(
        provider_name=args.provider,
        target_lang=args.target_lang,
        flex_mode=args.flex_mode,
        explicit_vocab_path=args.vocab,
        include_rules=False,
        load_vocab_pairs_flag=args.mode == "missing" and args.out_format == "po",
        get_translation_provider_fn=get_translation_provider,
        resolve_resource_path_fn=resolve_resource_path,
        read_optional_vocabulary_file_fn=read_optional_vocabulary_file,
        load_vocabulary_pairs_fn=load_vocabulary_pairs,
    )
    provider = runtime_context.provider
    client = runtime_context.client
    resource_context = runtime_context.resources

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

    batches = build_fixed_batches(source_messages, batch_size)

    out_path = args.out or build_term_output_path(
        args.file,
        output_format=args.out_format,
        mode=args.mode,
    )
    term_config = build_term_generation_config(
        args.thinking_level,
        provider=provider,
        system_instruction=build_term_system_instruction(args.target_lang),
        flex_mode=args.flex_mode,
    )

    print_startup_configuration(
        ("Provider", provider.name),
        ("Model", model_name),
        ("Flex mode", "yes" if args.flex_mode and getattr(provider, "supports_flex_mode", False) else "no"),
        ("Thinking level", args.thinking_level or "provider default"),
        ("Parallel requests", parallel_requests),
        ("Batch size", batch_size),
        ("Limits mode", limits_mode),
        ("Discovery mode", args.mode),
        ("Output format", args.out_format),
        ("Vocabulary source", resource_context.vocabulary_source),
        ("Total source messages", total),
        ("Total batches", len(batches)),
    )

    async def run_extraction() -> List[TermCandidate]:
        all_candidates: List[TermCandidate] = []
        completed = 0

        def build_contents(_batch_index: int, batch: List[Dict[str, str]]) -> Any:
            return build_term_request_contents(
                messages=build_indexed_batch_map(batch, lambda item: item),
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                mode=args.mode,
                vocabulary=resource_context.vocabulary_text,
                max_terms_per_batch=args.max_terms_per_batch,
                provider=provider,
            )

        def on_batch_completed(
            batch_index: int,
            batch: List[Dict[str, str]],
            terms: List[TermCandidate],
        ) -> None:
            nonlocal completed
            all_candidates.extend(terms)
            completed += 1
            print(
                f"Progress: completed batches {completed}/{len(batches)} "
                f"(latest: {batch_index + 1}/{len(batches)}), "
                f"raw terms collected: {len(all_candidates)}"
            )

        await run_model_batches(
            batches=batches,
            parallel_requests=parallel_requests,
            provider=provider,
            client=client,
            model=model_name,
            config=term_config,
            max_attempts=args.max_attempts,
            build_contents=build_contents,
            parse_response=parse_term_response,
            on_batch_completed=on_batch_completed,
            build_batch_label=lambda batch_index: f"terms batch {batch_index + 1}/{len(batches)}",
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
        "provider": provider.name,
        "model": model_name,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "vocabulary_source": resource_context.vocabulary_source,
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
            base_vocabulary_pairs=resource_context.vocabulary_pairs,
        )
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nTerm extraction complete.")
    print(f"Saved file: {out_path}")
    print(f"Candidate terms: {len(merged_terms)} (from {len(raw_candidates)} raw suggestions)")


def main(argv: list[str] | None = None) -> None:
    """Run the glossary extraction CLI."""
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
