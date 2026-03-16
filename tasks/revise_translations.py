#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from core.review_common import (
    build_target_script_guidance as build_shared_target_script_guidance,
    json_load_maybe,
    plural_key_sort_key,
)
from core.formats import (
    EntryStatus,
    FileKind,
    UnifiedEntry,
    build_entry_source_text,
    detect_file_kind,
    get_entry_prompt_context_and_note,
    load_po,
    load_resx,
    load_strings,
    load_ts,
)
from core.resources import (
    detect_rules_source,
    merge_project_rules,
    read_optional_text_file,
    read_optional_vocabulary_file,
    resolve_resource_path,
)
from core.review_flow import (
    has_reviewable_translation as has_shared_reviewable_translation,
    limit_items,
    normalize_limits as normalize_review_limits,
)
from google import genai
from google.genai import types as genai_types

from tasks import translate as process
from tasks.translate import add_thinking_level_argument, build_thinking_config, generate_with_retry


DEFAULT_REVISION_BATCH_SIZE = 120
DEFAULT_REVISION_PARALLEL = 6


REVISION_RESPONSE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "revisions": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "id": genai_types.Schema(type=genai_types.Type.STRING),
                    "action": genai_types.Schema(type=genai_types.Type.STRING),
                    "text": genai_types.Schema(type=genai_types.Type.STRING),
                    "plural_texts": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["id", "action"],
            ),
        ),
    },
    required=["revisions"],
)


@dataclass(slots=True)
class RevisionResult:
    action: str
    text: str = ""
    plural_texts: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class ReviewItem:
    entry: UnifiedEntry
    source_text: str
    current_text: str
    current_plural_texts: List[str]
    plural_form_count: int
    context: str = ""
    note: str = ""
    pair_key: str = ""


@dataclass(slots=True)
class ReviewBundle:
    file_kind: FileKind
    entries: List[UnifiedEntry]
    save_callback: Callable[[], None]
    generated_output_path: str
    items: List[ReviewItem]
    warnings: List[str] = field(default_factory=list)


def build_revision_generation_config(
    thinking_level: str | None = None,
) -> genai_types.GenerateContentConfig:
    config_kwargs: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": REVISION_RESPONSE_SCHEMA,
    }
    thinking_config = build_thinking_config(thinking_level)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    return genai_types.GenerateContentConfig(**config_kwargs)


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def build_revision_output_path(file_path: str) -> str:
    root, ext = os.path.splitext(file_path)
    return f"{root}-revised{ext}"


def normalize_limits(
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
) -> Tuple[int, int, str]:
    return normalize_review_limits(
        total_items=total_items,
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
        default_batch_size=DEFAULT_REVISION_BATCH_SIZE,
        default_parallel=DEFAULT_REVISION_PARALLEL,
        label="revision",
    )


def _clean(text: str | None) -> str:
    return str(text or "").strip()


def _json_load_maybe(text: str) -> Any:
    return json_load_maybe(text)


def _normalize_revision_payload(payload: Any) -> Dict[str, RevisionResult]:
    results: Dict[str, RevisionResult] = {}
    if not isinstance(payload, dict):
        return results

    items = payload.get("revisions")
    if not isinstance(items, list):
        return results

    for item in items:
        if not isinstance(item, dict):
            continue
        msg_id = item.get("id")
        action = item.get("action")
        if msg_id is None or action is None:
            continue

        text = item.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        reason = item.get("reason")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            reason = str(reason)

        plural_texts: List[str] = []
        plural_texts_raw = item.get("plural_texts")
        if isinstance(plural_texts_raw, list):
            for value in plural_texts_raw:
                if value is None:
                    continue
                plural_texts.append(value if isinstance(value, str) else str(value))

        results[str(msg_id)] = RevisionResult(
            action=str(action).strip().lower(),
            text=text,
            plural_texts=plural_texts,
            reason=reason,
        )

    return results


def parse_revision_response(response_payload: Any) -> Dict[str, RevisionResult]:
    if isinstance(response_payload, dict):
        return _normalize_revision_payload(response_payload)

    if isinstance(response_payload, str):
        return _normalize_revision_payload(_json_load_maybe(response_payload))

    parsed_payload = getattr(response_payload, "parsed", None)
    if parsed_payload is not None:
        return _normalize_revision_payload(parsed_payload)

    text_payload = getattr(response_payload, "text", None) or ""
    return _normalize_revision_payload(_json_load_maybe(text_payload))


def _plural_key_sort_key(value: Any) -> Tuple[int, Any]:
    return plural_key_sort_key(value)


def get_plural_texts(entry: UnifiedEntry) -> List[str]:
    if not entry.msgstr_plural:
        return []
    return [
        str(entry.msgstr_plural[key] or "")
        for key in sorted(entry.msgstr_plural.keys(), key=_plural_key_sort_key)
    ]


def get_plural_form_count(entry: UnifiedEntry) -> int:
    if not entry.msgid_plural:
        return 0
    if entry.msgstr_plural:
        return max(2, len(entry.msgstr_plural))
    return 2


def has_reviewable_translation(entry: UnifiedEntry) -> bool:
    return has_shared_reviewable_translation(
        entry,
        plural_texts=get_plural_texts(entry),
        allow_context_only=True,
    )


def limit_review_items(items: List[ReviewItem], num_messages: int | None) -> List[ReviewItem]:
    return limit_items(items, num_messages)


def build_target_script_guidance(target_lang: str) -> str | None:
    return build_shared_target_script_guidance(
        target_lang,
        update_wording=lambda: "updated target text",
    )


def build_revision_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> str:
    vocab_block = f"\nApproved vocabulary/glossary (mandatory):\n{vocabulary}\n" if vocabulary else ""
    rules_block = (
        f"\nProject translation rules/instructions (mandatory when present):\n{translation_rules}\n"
        if translation_rules
        else ""
    )
    script_guidance = build_target_script_guidance(target_lang)
    script_block = f"- {script_guidance}\n" if script_guidance else ""
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    return f"""
You are revising existing software localization translations.

User instruction:
{instruction}

STRICT MUST:
- Review each item against the source text, current translation, and user instruction
- Keep the current translation unchanged when it already satisfies the instruction
- Change only entries where the instruction clearly applies and the current translation needs an update
- If the instruction is ambiguous or not clearly applicable to a specific item, keep that item unchanged
- Preserve placeholders exactly (%s, %d, %(name)s, %1, %n, {{var}}, {{{{var}}}})
- Preserve HTML/XML tags exactly and keep them well-formed
- Preserve keyboard accelerators/hotkeys exactly (`_`, `&`)
- Preserve leading/trailing spaces, escapes, entities, and meaningful punctuation
- Preserve approved vocabulary and project rules when supplied
- Never return an empty updated translation
- Do not rewrite unrelated wording just because a different phrasing is possible
- Do not translate or rewrite context/note metadata
{script_block}

Project context:
This is a software UI localization revision pass.
Source language: {source_lang}
Target language: {target_lang}
{vocab_block}
{rules_block}

Output requirements:
- Return ONLY valid JSON with this schema:
  {{"revisions": [{{"id": "...", "action": "keep|update", "text": "...", "plural_texts": ["..."], "reason": "..."}}]}}
- Return one result for every input item id
- Keep each result id exactly the same as the input key
- Use action "keep" when no change is needed
- Use action "update" only when the current translation should change to satisfy the instruction
- If action is "update", provide the full corrected target text
- For plural items, if action is "update", provide exactly item.plural_forms plural_texts
- If the target language effectively uses one plural wording, repeat it in all required plural slots
- Keep reason short and concrete

Items:
{messages_json}
""".strip()


def build_review_message_payload(item: ReviewItem) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": item.source_text,
        "current_translation": item.current_text,
    }
    if item.current_plural_texts:
        payload["current_plural_texts"] = item.current_plural_texts
    if item.plural_form_count:
        payload["plural_forms"] = item.plural_form_count
    if item.context:
        payload["context"] = item.context
    if item.note:
        payload["note"] = item.note
    return payload


def _join_non_empty(parts: List[str]) -> str:
    seen: set[str] = set()
    result: List[str] = []
    for value in parts:
        cleaned = _clean(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return " | ".join(result)


def build_review_item(entry: UnifiedEntry) -> ReviewItem:
    context, note = get_entry_prompt_context_and_note(entry)
    current_plural_texts = get_plural_texts(entry)
    current_text = current_plural_texts[0] if current_plural_texts else str(entry.msgstr or "")
    pair_key = _clean(context) or _clean(entry.msgctxt) or _clean(entry.msgid)
    return ReviewItem(
        entry=entry,
        source_text=build_entry_source_text(entry),
        current_text=current_text,
        current_plural_texts=current_plural_texts,
        plural_form_count=get_plural_form_count(entry),
        context=context or "",
        note=note or "",
        pair_key=pair_key,
    )


def _load_file_entries(file_path: str, file_kind: FileKind) -> Tuple[List[UnifiedEntry], Callable[[], None], str]:
    if file_kind == FileKind.PO:
        return load_po(file_path)
    if file_kind == FileKind.TS:
        return load_ts(file_path)
    if file_kind == FileKind.RESX:
        return load_resx(file_path)
    if file_kind == FileKind.STRINGS:
        return load_strings(file_path)
    raise ValueError(
        f"Unsupported single-file revision format: {file_kind.value}. "
        "Use a paired workflow for .txt files."
    )


def build_single_file_bundle(file_path: str, file_kind: FileKind) -> ReviewBundle:
    entries, save_callback, generated_output_path = _load_file_entries(file_path, file_kind)
    items = [build_review_item(entry) for entry in entries if has_reviewable_translation(entry)]
    return ReviewBundle(
        file_kind=file_kind,
        entries=entries,
        save_callback=save_callback,
        generated_output_path=generated_output_path,
        items=items,
    )


def _build_pair_key(entry: UnifiedEntry, index: int) -> str:
    context = _clean(entry.msgctxt)
    if context:
        return context
    return f"index:{index}"


def build_paired_bundle(
    source_file: str,
    translated_file: str,
    file_kind: FileKind,
) -> ReviewBundle:
    source_entries, _, _ = _load_file_entries(source_file, file_kind)
    translated_entries, save_callback, generated_output_path = _load_file_entries(translated_file, file_kind)

    buckets: Dict[str, List[UnifiedEntry]] = {}
    for index, source_entry in enumerate(source_entries):
        key = _build_pair_key(source_entry, index)
        buckets.setdefault(key, []).append(source_entry)

    items: List[ReviewItem] = []
    warnings: List[str] = []
    missing_pairs = 0

    for index, translated_entry in enumerate(translated_entries):
        if not has_reviewable_translation(translated_entry):
            continue

        key = _build_pair_key(translated_entry, index)
        source_bucket = buckets.get(key) or []
        if not source_bucket:
            missing_pairs += 1
            continue

        source_entry = source_bucket.pop(0)
        translated_context, translated_note = get_entry_prompt_context_and_note(translated_entry)
        source_context, source_note = get_entry_prompt_context_and_note(source_entry)
        current_plural_texts = get_plural_texts(translated_entry)
        current_text = current_plural_texts[0] if current_plural_texts else str(translated_entry.msgstr or "")

        items.append(
            ReviewItem(
                entry=translated_entry,
                source_text=str(source_entry.msgid or ""),
                current_text=current_text,
                current_plural_texts=current_plural_texts,
                plural_form_count=0,
                context=_clean(translated_context or source_context),
                note=_join_non_empty([source_note or "", translated_note or ""]),
                pair_key=key,
            )
        )

    if missing_pairs:
        warnings.append(
            f"Skipped {missing_pairs} translated entries with no matching source pair in {source_file}."
        )

    return ReviewBundle(
        file_kind=file_kind,
        entries=translated_entries,
        save_callback=save_callback,
        generated_output_path=generated_output_path,
        items=items,
        warnings=warnings,
    )


def load_paired_txt_bundle(source_file: str, translated_file: str) -> ReviewBundle:
    source_encoding = process._detect_text_encoding(source_file)
    translated_encoding = process._detect_text_encoding(translated_file)

    with open(source_file, "r", encoding=source_encoding) as handle:
        source_lines = handle.read().splitlines(keepends=True)
    with open(translated_file, "r", encoding=translated_encoding) as handle:
        translated_lines = handle.read().splitlines(keepends=True)

    if len(source_lines) != len(translated_lines):
        raise ValueError(
            "Paired .txt workflow requires the source and translated files to have the same number of lines."
        )

    entries: List[UnifiedEntry] = []
    items: List[ReviewItem] = []

    for index, (source_raw, translated_raw) in enumerate(zip(source_lines, translated_lines)):
        source_text = source_raw.rstrip("\r\n")
        translated_text = translated_raw.rstrip("\r\n")
        translated_line_ending = translated_raw[len(translated_text):]
        line_number = index + 1

        if not _clean(source_text):
            status = EntryStatus.SKIPPED
        elif _clean(translated_text):
            status = EntryStatus.TRANSLATED
        else:
            status = EntryStatus.UNTRANSLATED

        def commit_line(
            entry: UnifiedEntry,
            idx: int = index,
            ending: str = translated_line_ending,
        ) -> None:
            translated_lines[idx] = f"{entry.msgstr}{ending}"

        entry = UnifiedEntry(
            file_kind=FileKind.TXT,
            msgid=source_text,
            msgstr=translated_text,
            msgctxt=f"line:{line_number}",
            flags=[],
            obsolete=False,
            include_in_term_extraction=bool(_clean(source_text)),
            status=status,
            _commit_callback=commit_line,
        )
        entries.append(entry)

        if not has_reviewable_translation(entry):
            continue

        items.append(
            ReviewItem(
                entry=entry,
                source_text=source_text,
                current_text=translated_text,
                current_plural_texts=[],
                plural_form_count=0,
                context=f"line:{line_number}",
                note="",
                pair_key=f"line:{line_number}",
            )
        )

    generated_output_path = build_revision_output_path(translated_file)

    def save_txt() -> None:
        for entry in entries:
            entry.commit()
        process._write_text_with_encoding_fallback(
            generated_output_path,
            "".join(translated_lines),
            translated_encoding,
            newline="",
        )

    return ReviewBundle(
        file_kind=FileKind.TXT,
        entries=entries,
        save_callback=save_txt,
        generated_output_path=generated_output_path,
        items=items,
    )


def load_review_bundle(
    translated_file: str,
    source_file: str | None = None,
) -> ReviewBundle:
    file_kind = detect_file_kind(translated_file)

    if source_file:
        source_kind = detect_file_kind(source_file)
        if source_kind != file_kind:
            raise ValueError(
                f"--source-file type mismatch: expected .{file_kind.value}, got .{source_kind.value}"
            )

    if file_kind in (FileKind.PO, FileKind.TS):
        return build_single_file_bundle(translated_file, file_kind)

    if file_kind == FileKind.TXT:
        if not source_file:
            raise ValueError("--source-file is required for .txt revision runs.")
        return load_paired_txt_bundle(source_file, translated_file)

    if file_kind in (FileKind.STRINGS, FileKind.RESX):
        if not source_file:
            raise ValueError(
                f"--source-file is required for .{file_kind.value} revision runs because "
                "the translated file does not retain the original source text."
            )
        return build_paired_bundle(source_file, translated_file, file_kind)

    raise ValueError(
        "Unsupported file type. Use .po, .ts, .resx, .strings, or .txt"
    )


def build_candidate_plural_forms(item: ReviewItem, result: RevisionResult) -> List[str]:
    usable_forms = [
        process._normalize_model_escaped_text(item.source_text, text)
        for text in result.plural_texts
        if _clean(text)
    ]
    if usable_forms:
        if len(usable_forms) >= item.plural_form_count:
            return usable_forms[: item.plural_form_count]
        return usable_forms + [usable_forms[-1]] * (item.plural_form_count - len(usable_forms))

    if _clean(result.text):
        repeated = process._normalize_model_escaped_text(item.source_text, result.text)
        return [repeated] * item.plural_form_count

    return []


def apply_revision_to_item(item: ReviewItem, result: RevisionResult) -> bool:
    if result.action != "update":
        return False

    if item.plural_form_count:
        candidate_forms = build_candidate_plural_forms(item, result)
        if not candidate_forms:
            return False
        if candidate_forms == item.current_plural_texts:
            return False

        plural_keys = sorted(item.entry.msgstr_plural.keys(), key=_plural_key_sort_key)
        if not plural_keys:
            plural_keys = list(range(item.plural_form_count))
        for index, key in enumerate(plural_keys):
            item.entry.msgstr_plural[key] = candidate_forms[index]
        item.entry.msgstr = candidate_forms[0]
    else:
        candidate = process._normalize_model_escaped_text(item.source_text, result.text)
        if not _clean(candidate):
            return False
        if candidate == item.current_text:
            return False
        item.entry.msgstr = candidate

    if "fuzzy" not in item.entry.flags:
        item.entry.flags.append("fuzzy")
    item.entry.status = EntryStatus.FUZZY
    return True


def move_output_file(generated_output_path: str, final_output_path: str) -> None:
    if generated_output_path == final_output_path:
        return
    os.replace(generated_output_path, final_output_path)


def build_final_output_path(
    translated_file: str,
    explicit_out: str | None = None,
    in_place: bool = False,
) -> str:
    if explicit_out:
        return explicit_out
    if in_place:
        return translated_file
    return build_revision_output_path(translated_file)


def print_change_samples(changes: List[Tuple[ReviewItem, RevisionResult]], limit: int = 10) -> None:
    if not changes:
        return

    print("Sample updated entries:")
    for item, result in changes[:limit]:
        label = item.pair_key or item.context or item.source_text
        before = item.current_text.replace("\n", "\\n")
        after = (
            result.plural_texts[0]
            if result.plural_texts
            else result.text
        ).replace("\n", "\\n")
        print(f"  - {label}: {before} -> {after}")


def main(argv: list[str] | None = None) -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Review existing translations against a natural-language instruction and "
            "update only the entries that need a change."
        )
    )
    parser.add_argument("file", help="Current translated .po, .ts, .resx, .strings, or .txt file")
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural-language revision instruction, for example: change the translation of 'Save' to 'Store'",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Required for .resx, .strings, and .txt revision runs; optional otherwise.",
    )
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    add_thinking_level_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent Gemini requests (auto if omitted)")
    parser.add_argument(
        "--probe",
        "--num-messages",
        dest="num_messages",
        type=int,
        default=None,
        help="Review only the first N reviewable entries.",
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Gemini retry attempts per batch")
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt). Supports .txt and glossary .po",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional translation rules file (auto: data/<target-lang>/rules.md)",
    )
    parser.add_argument("--rules-str", default=None, help="Optional inline translation rules")
    parser.add_argument("--out", default=None, help="Output path (default: <input>.revised.<ext>)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the translated input file")
    parser.add_argument("--dry-run", action="store_true", help="Review and report changes without writing output")

    args = parser.parse_args(argv)

    if args.out and args.in_place:
        sys.exit("ERROR: --out and --in-place cannot be used together")
    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY environment variable is not set")

    try:
        review_bundle = load_review_bundle(args.file, source_file=args.source_file)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    try:
        review_items = limit_review_items(review_bundle.items, args.num_messages)
        batch_size, parallel_requests, limits_mode = normalize_limits(
            total_items=len(review_items),
            batch_size_arg=args.batch_size,
            parallel_arg=args.parallel_requests,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    if not review_items:
        print("No translated entries found to review.")
        return

    vocabulary_path = resolve_resource_path(
        explicit_path=args.vocab,
        prefix="vocab",
        extension="txt",
        target_lang=args.target_lang,
    )
    rules_path = resolve_resource_path(
        explicit_path=args.rules,
        prefix="rules",
        extension="md",
        target_lang=args.target_lang,
    )
    vocabulary_text = read_optional_vocabulary_file(vocabulary_path, "Vocabulary")
    rules_text = read_optional_text_file(rules_path, "Rules")
    project_rules = merge_project_rules(rules_text, args.rules_str)
    rules_source = detect_rules_source(rules_path, rules_text, args.rules_str)
    vocabulary_source = f"file:{vocabulary_path}" if vocabulary_text and vocabulary_path else "none"

    client = genai.Client(api_key=api_key)
    revision_config = build_revision_generation_config(args.thinking_level)
    final_output_path = build_final_output_path(
        translated_file=args.file,
        explicit_out=args.out,
        in_place=args.in_place,
    )

    print("Startup configuration:")
    print(f"  Model: {args.model}")
    print(f"  Thinking level: {args.thinking_level or 'provider default'}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")
    print(f"  Review items: {len(review_items)}")
    print(f"  Source file: {args.source_file or 'embedded in translated file'}")
    print(f"  Vocabulary source: {vocabulary_source}")
    print(f"  Rules source: {rules_source or 'none'}")
    print(f"  Output path: {final_output_path}")
    print(f"  Dry run: {'yes' if args.dry_run else 'no'}")
    for warning in review_bundle.warnings:
        print(f"Warning: {warning}")

    all_batches = [
        review_items[index: index + batch_size]
        for index in range(0, len(review_items), batch_size)
    ]
    total_batches = len(all_batches)
    print(f"Total batches: {total_batches}")

    async def process_batch(
        batch_index: int,
        batch: List[ReviewItem],
        sem: asyncio.Semaphore,
    ) -> Tuple[int, List[ReviewItem], Dict[str, RevisionResult]]:
        async with sem:
            msg_map: Dict[str, Dict[str, Any]] = {}
            for index, item in enumerate(batch):
                msg_map[str(index)] = build_review_message_payload(item)

            prompt = build_revision_prompt(
                messages=msg_map,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                instruction=args.instruction,
                vocabulary=vocabulary_text,
                translation_rules=project_rules,
            )

            response = await generate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                batch_label=f"revision batch {batch_index + 1}/{total_batches}",
                max_attempts=args.max_attempts,
                config=revision_config,
            )
            revisions = parse_revision_response(response)

            missing_indices = [
                index
                for index in range(len(batch))
                if str(index) not in revisions
            ]
            if missing_indices:
                print(
                    f"  Warning [batch {batch_index + 1}/{total_batches}]: "
                    f"{len(missing_indices)} items missing from response. Retrying them..."
                )
                retry_map: Dict[str, Dict[str, Any]] = {}
                for index in missing_indices:
                    retry_map[str(index)] = build_review_message_payload(batch[index])

                retry_prompt = build_revision_prompt(
                    messages=retry_map,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    instruction=args.instruction,
                    vocabulary=vocabulary_text,
                    translation_rules=project_rules,
                )
                try:
                    retry_response = await generate_with_retry(
                        client=client,
                        model=args.model,
                        prompt=retry_prompt,
                        batch_label=f"revision batch {batch_index + 1}/{total_batches} missing-items",
                        max_attempts=max(2, min(args.max_attempts, 3)),
                        config=revision_config,
                    )
                    revisions.update(parse_revision_response(retry_response))
                except Exception as exc:
                    print(
                        f"  Retry failed [batch {batch_index + 1}/{total_batches}]: {exc}"
                    )

            return batch_index, batch, revisions

    async def run_revision() -> Tuple[int, List[Tuple[ReviewItem, RevisionResult]]]:
        sem = asyncio.Semaphore(parallel_requests)
        tasks = [
            asyncio.create_task(process_batch(batch_index, batch, sem))
            for batch_index, batch in enumerate(all_batches)
        ]

        changed_total = 0
        changed_items: List[Tuple[ReviewItem, RevisionResult]] = []
        completed_batches = 0

        for finished in asyncio.as_completed(tasks):
            batch_index, batch, revisions = await finished
            batch_changed = 0

            for index, item in enumerate(batch):
                result = revisions.get(str(index))
                if result is None:
                    continue
                if apply_revision_to_item(item, result):
                    batch_changed += 1
                    changed_items.append((item, result))

            changed_total += batch_changed
            completed_batches += 1
            percent = (completed_batches / total_batches) * 100.0
            print(
                f"Progress: {percent:.1f}% ({completed_batches}/{total_batches} batches), "
                f"changed so far: {changed_total}"
            )

        return changed_total, changed_items

    try:
        changed_count, changed_items = asyncio.run(run_revision())
    except RuntimeError as exc:
        sys.exit(str(exc))

    print("")
    print(f"Reviewed entries: {len(review_items)}")
    print(f"Changed entries: {changed_count}")

    if not changed_count:
        print("No changes were needed.")
        return

    print_change_samples(changed_items)

    if args.dry_run:
        print("Dry run complete. No file was written.")
        return

    review_bundle.save_callback()
    move_output_file(review_bundle.generated_output_path, final_output_path)

    print("")
    print("Revision complete.")
    print(f"Saved file: {final_output_path}")
    print("Changed AI-reviewed entries are marked as fuzzy/unfinished for human review.")


if __name__ == "__main__":
    main()
