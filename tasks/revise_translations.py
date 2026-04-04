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
    load_paired_android_xml,
    load_po,
    load_resx,
    load_strings,
    load_ts,
)
from core.formats.strings import _detect_text_encoding, _write_text_with_encoding_fallback
from core.entries import normalize_model_escaped_text
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, get_translation_provider
from core.request_contents import TaskRequestSpec, build_task_request_contents, render_text_fallback_prompt
from core.task_cli import (
    add_language_arguments,
    add_max_attempts_argument,
    add_probe_argument,
    add_provider_arguments,
    add_rules_arguments,
    add_runtime_limit_arguments,
    add_vocabulary_argument,
    build_task_parser,
    resolve_provider_model,
    run_task_main,
)
from core.review_flow import (
    has_reviewable_translation as has_shared_reviewable_translation,
    limit_items,
    normalize_limits as normalize_review_limits,
)
from core.task_batches import (
    build_fixed_batches,
    build_indexed_batch_map,
    find_missing_index_keys,
    merge_mapping_results,
    run_model_batches,
)
from core.task_runtime import build_task_runtime_context, print_startup_configuration


DEFAULT_REVISION_BATCH_SIZE = 120
DEFAULT_REVISION_PARALLEL = 6

REVISION_SYSTEM_INSTRUCTION = """
You are revising existing software localization translations.

STRICT MUST:
- Review each item against the source text, current translation, and user instruction
- Keep the current translation unchanged when it already satisfies the instruction
- Change only entries where the instruction clearly applies and the current translation needs an update
- If the instruction is ambiguous or not clearly applicable to a specific item, keep that item unchanged
- Preserve placeholders exactly (%s, %d, %(name)s, %1, %n, {var}, {{var}})
- Preserve HTML/XML tags exactly and keep them well-formed
- Preserve keyboard accelerators/hotkeys exactly (`_`, `&`)
- Preserve leading/trailing spaces, escapes, entities, and meaningful punctuation
- Preserve literal escape sequences such as `\\n` and `\\t` as literal backslash sequences when the source uses them
- Preserve approved vocabulary and project rules when supplied
- Never return an empty updated translation
- Do not rewrite unrelated wording just because a different phrasing is possible
- Do not translate or rewrite context/note metadata
"""


REVISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "action": {"type": "string"},
                    "text": {"type": "string"},
                    "plural_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "action"],
            },
        },
    },
    "required": ["revisions"],
}


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
    *,
    provider: Any = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
    flex_mode: bool = False,
) -> Any:
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=REVISION_RESPONSE_SCHEMA,
        system_instruction=REVISION_SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
        flex_mode=flex_mode,
    )


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
        return _normalize_revision_payload(json_load_maybe(response_payload))

    parsed_payload = getattr(response_payload, "parsed", None)
    if parsed_payload is not None:
        return _normalize_revision_payload(parsed_payload)

    text_payload = getattr(response_payload, "text", None) or ""
    return _normalize_revision_payload(json_load_maybe(text_payload))


def get_plural_texts(entry: UnifiedEntry) -> List[str]:
    if not entry.msgstr_plural:
        return []
    return [
        str(entry.msgstr_plural[key] or "")
        for key in sorted(entry.msgstr_plural.keys(), key=plural_key_sort_key)
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


def build_revision_system_instruction(target_lang: str) -> str:
    parts = [REVISION_SYSTEM_INSTRUCTION.strip()]
    script_guidance = build_target_script_guidance(target_lang)
    if script_guidance:
        parts.append(f"- {script_guidance}")
    return "\n\n".join(parts)


def build_revision_request_spec() -> TaskRequestSpec:
    return TaskRequestSpec(
        task_intro="Revise each software localization translation item.",
        task_lines=(
            "Review each item against the source text, current translation, and user instruction.",
        ),
        payload_lines=(
            "The payload contains the user instruction, source language, target language, optional approved vocabulary/glossary, optional project translation rules/instructions, and an `items` map.",
        ),
        output_lines=(
            "Return only valid JSON, with no markdown or commentary.",
            "Return one result for every input item id.",
            "Keep each result `id` exactly the same as the input key.",
            "Use action `keep` when no change is needed.",
            "Use action `update` only when the current translation should change to satisfy the instruction.",
            "If action is `update`, provide the full corrected target text.",
            "For plural items, if action is `update`, provide exactly `item.plural_forms` plural_texts.",
            "If the target language effectively uses one plural wording, repeat it in all required plural slots.",
            "Keep `reason` short and concrete.",
        ),
    )


def build_revision_request_payload(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> dict[str, Any]:
    return {
        "project_type": "software_ui_revision",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "instruction": instruction,
        "vocabulary": vocabulary,
        "translation_rules": translation_rules,
        "items": messages,
    }


def build_revision_request_contents(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
    *,
    provider: Any = DEFAULT_PROVIDER,
) -> Any:
    return build_task_request_contents(
        provider=provider,
        task_spec=build_revision_request_spec(),
        function_name="translation_revision_batch",
        payload=build_revision_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            instruction=instruction,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
        ),
    )


def build_revision_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> str:
    return render_text_fallback_prompt(
        task_spec=build_revision_request_spec(),
        payload=build_revision_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            instruction=instruction,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
        ),
    )


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
    source_encoding = _detect_text_encoding(source_file)
    translated_encoding = _detect_text_encoding(translated_file)

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
        _write_text_with_encoding_fallback(
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

    if file_kind == FileKind.ANDROID_XML:
        if not source_file:
            raise ValueError(
                "--source-file is required for .xml revision runs because the translated file "
                "does not retain the original source text."
            )
        entries, save_callback, generated_output_path, warnings = load_paired_android_xml(
            source_file,
            translated_file,
        )
        items = [build_review_item(entry) for entry in entries if has_reviewable_translation(entry)]
        return ReviewBundle(
            file_kind=file_kind,
            entries=entries,
            save_callback=save_callback,
            generated_output_path=generated_output_path,
            items=items,
            warnings=warnings,
        )

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
        "Unsupported file type. Use .po, .ts, .resx, .strings, .txt, or Android .xml"
    )


def build_candidate_plural_forms(item: ReviewItem, result: RevisionResult) -> List[str]:
    usable_forms = [
            normalize_model_escaped_text(item.source_text, text)
        for text in result.plural_texts
        if _clean(text)
    ]
    if usable_forms:
        if len(usable_forms) >= item.plural_form_count:
            return usable_forms[: item.plural_form_count]
        return usable_forms + [usable_forms[-1]] * (item.plural_form_count - len(usable_forms))

    if _clean(result.text):
        repeated = normalize_model_escaped_text(item.source_text, result.text)
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

        plural_keys = sorted(item.entry.msgstr_plural.keys(), key=plural_key_sort_key)
        if not plural_keys:
            plural_keys = list(range(item.plural_form_count))
        for index, key in enumerate(plural_keys):
            item.entry.msgstr_plural[key] = candidate_forms[index]
        item.entry.msgstr = candidate_forms[0]
    else:
        candidate = normalize_model_escaped_text(item.source_text, result.text)
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


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = (
        "Review existing translations against a natural-language instruction and "
        "update only the entries that need a change."
    )
    parser.add_argument(
        "file",
        help="Current translated .po, .ts, .resx, .strings, .txt, or Android .xml file",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural-language revision instruction, for example: change the translation of 'Save' to 'Store'",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Required for Android .xml, .resx, .strings, and .txt revision runs; optional otherwise.",
    )
    add_language_arguments(parser)
    add_provider_arguments(
        parser,
        default_provider_name=DEFAULT_PROVIDER_NAME,
        default_model=DEFAULT_PROVIDER.default_model,
    )
    add_runtime_limit_arguments(parser)
    add_probe_argument(
        parser,
        help_text="Review only the first N reviewable entries.",
    )
    add_max_attempts_argument(parser)
    add_vocabulary_argument(parser)
    add_rules_arguments(
        parser,
        rules_help="Optional translation rules file (auto: data/locales/<target-lang>/rules.md)",
        rules_str_help="Optional inline translation rules",
    )
    parser.add_argument("--out", default=None, help="Output path (default: <input>.revised.<ext>)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the translated input file")
    parser.add_argument("--dry-run", action="store_true", help="Review and report changes without writing output")
    return parser


def build_parser() -> argparse.ArgumentParser:
    return build_task_parser(configure_parser)


def run_from_args(args: argparse.Namespace) -> None:
    configure_stdio()
    model_name = resolve_provider_model(args.provider, args.model)

    if args.out and args.in_place:
        sys.exit("ERROR: --out and --in-place cannot be used together")
    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

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

    runtime_context = build_task_runtime_context(
        provider_name=args.provider,
        target_lang=args.target_lang,
        flex_mode=args.flex_mode,
        explicit_vocab_path=args.vocab,
        explicit_rules_path=args.rules,
        inline_rules=args.rules_str,
        get_translation_provider_fn=get_translation_provider,
    )
    provider = runtime_context.provider
    client = runtime_context.client
    resource_context = runtime_context.resources
    revision_config = build_revision_generation_config(
        args.thinking_level,
        provider=provider,
        system_instruction=build_revision_system_instruction(args.target_lang),
        flex_mode=args.flex_mode,
    )
    final_output_path = build_final_output_path(
        translated_file=args.file,
        explicit_out=args.out,
        in_place=args.in_place,
    )

    print_startup_configuration(
        ("Provider", provider.name),
        ("Model", model_name),
        ("Flex mode", "yes" if args.flex_mode and getattr(provider, "supports_flex_mode", False) else "no"),
        ("Thinking level", args.thinking_level or "provider default"),
        ("Parallel requests", parallel_requests),
        ("Batch size", batch_size),
        ("Limits mode", limits_mode),
        ("Review items", len(review_items)),
        ("Source file", args.source_file or "embedded in translated file"),
        ("Vocabulary source", resource_context.vocabulary_source),
        ("Rules source", resource_context.rules_source or "none"),
        ("Output path", final_output_path),
        ("Dry run", "yes" if args.dry_run else "no"),
    )
    for warning in review_bundle.warnings:
        print(f"Warning: {warning}")

    all_batches = build_fixed_batches(review_items, batch_size)
    total_batches = len(all_batches)
    print(f"Total batches: {total_batches}")

    async def run_revision() -> Tuple[int, List[Tuple[ReviewItem, RevisionResult]]]:
        changed_total = 0
        changed_items: List[Tuple[ReviewItem, RevisionResult]] = []
        completed_batches = 0

        def build_contents(_batch_index: int, batch: List[ReviewItem]) -> Any:
            return build_revision_request_contents(
                messages=build_indexed_batch_map(batch, build_review_message_payload),
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                instruction=args.instruction,
                vocabulary=resource_context.vocabulary_text,
                translation_rules=resource_context.project_rules,
                provider=provider,
            )

        def build_retry_contents(
            _batch_index: int,
            batch: List[ReviewItem],
            missing_indices: List[int],
        ) -> Any:
            return build_revision_request_contents(
                messages=build_indexed_batch_map(
                    missing_indices,
                    lambda item_index: build_review_message_payload(batch[item_index]),
                    key_builder=lambda _retry_index, item_index: str(item_index),
                ),
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                instruction=args.instruction,
                vocabulary=resource_context.vocabulary_text,
                translation_rules=resource_context.project_rules,
                provider=provider,
            )

        def on_batch_completed(
            batch_index: int,
            batch: List[ReviewItem],
            revisions: Dict[str, RevisionResult],
        ) -> None:
            nonlocal changed_total, completed_batches
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

        await run_model_batches(
            batches=all_batches,
            parallel_requests=parallel_requests,
            provider=provider,
            client=client,
            model=model_name,
            config=revision_config,
            max_attempts=args.max_attempts,
            build_contents=build_contents,
            parse_response=parse_revision_response,
            on_batch_completed=on_batch_completed,
            build_batch_label=lambda batch_index: f"revision batch {batch_index + 1}/{total_batches}",
            find_missing_indices=lambda batch, revisions: find_missing_index_keys(len(batch), revisions),
            build_retry_contents=build_retry_contents,
            build_retry_label=lambda batch_index: f"revision batch {batch_index + 1}/{total_batches} missing-items",
            retry_max_attempts=lambda _batch_index: max(2, min(args.max_attempts, 3)),
            merge_retry_result=merge_mapping_results,
            on_missing_indices=lambda batch_index, _batch, missing_indices: print(
                f"  Warning [batch {batch_index + 1}/{total_batches}]: "
                f"{len(missing_indices)} items missing from response. Retrying them..."
            ),
            on_retry_error=lambda batch_index, _batch, _missing_indices, exc: print(
                f"  Retry failed [batch {batch_index + 1}/{total_batches}]: {exc}"
            ),
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


def main(argv: list[str] | None = None) -> None:
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
