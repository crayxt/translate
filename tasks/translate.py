#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import polib

from core.entries import (
    TranslationResult,
    apply_translation_to_entry,
    build_entry_source_text,
    build_prompt_message_payload,
    build_prompt_message_payload as _build_prompt_message_payload,
    get_entry_prompt_context_and_note,
    get_plural_form_count,
    is_non_empty_text as _is_non_empty_text,
    is_plural_entry,
    json_load_maybe as _json_load_maybe,
    normalize_model_escaped_text as _normalize_model_escaped_text,
    parse_response,
    plural_key_sort_key as _plural_key_sort_key,
    translation_has_content,
)
from core.formats import (
    PO_WRAP_WIDTH,
    EntryStatus,
    FileKind,
    ResxEntryAdapter,
    StringsEntryAdapter,
    TSEntryAdapter,
    UnifiedEntry,
    build_output_path,
    detect_file_kind,
    load_resx,
    load_strings,
    load_ts,
    load_txt,
    select_work_items,
)
from core.formats.base import _build_unified_entry as _core_build_unified_entry
from core.formats.po import load_po as _core_load_po
from core.formats.strings import (
    _detect_text_encoding,
    _write_text_with_encoding_fallback,
)
from core.resources import (
    build_language_code_candidates as _core_build_language_code_candidates,
    detect_default_text_resource as _core_detect_default_text_resource,
    detect_rules_source,
    load_vocabulary_pairs as _core_load_vocabulary_pairs,
    merge_project_rules,
    parse_vocabulary_line,
    read_optional_text_file,
    read_optional_vocabulary_file as _core_read_optional_vocabulary_file,
    resolve_resource_path as _core_resolve_resource_path,
)
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, get_translation_provider
from core.request_contents import TaskRequestSpec, build_task_request_contents, render_text_fallback_prompt
from core.task_cli import (
    add_language_arguments,
    add_provider_arguments,
    add_rules_arguments,
    add_runtime_limit_arguments,
    add_vocabulary_argument,
    build_task_parser,
    resolve_provider_model,
    run_task_main,
)
from core.runtime import (
    MIN_ITEMS_PER_WORKER,
    build_thinking_config,
    resolve_runtime_limits,
)
from core.task_batches import build_fixed_batches, run_parallel_batches
from core.task_runtime import build_task_runtime_context, print_startup_configuration

build_prompt_message_payload = _build_prompt_message_payload

SYSTEM_INSTRUCTION = """
You are a professional software localization translator.

MUST:
- Preserve all placeholders EXACTLY (%s, %d, %(name)s, {var}, {{var}})
- Preserve HTML/XML tags EXACTLY
- Preserve keyboard accelerators/hotkeys EXACTLY (`_`, `&`) and keep them usable in target text
- Preserve escapes, entities, and line-break markers exactly when they carry formatting or structure
- Preserve leading and trailing spaces exactly
- Do NOT reorder placeholders
- Do NOT add, remove, soften, intensify, or reinterpret meaning
- Do NOT translate protected tokens such as product names, brand names, API names, code identifiers, command flags, file extensions, paths, or variable-like strings
- Keep original punctuation and capitalization style unless the target language requires a minimal grammatical adjustment
- If source text is ALL CAPS, keep translation ALL CAPS
- Translate ONLY the message text, not metadata
- When approved vocabulary or project rules are provided, follow them exactly

FORMATTING:
- If the source text contains \\n line-wrapping markers, preserve them in the translation and try to keep lines at roughly similar lengths
- Avoid unnatural CamelCase in the target language unless source uses intentional branded casing

PLURALS:
- If the input contains 'Singular:' and 'Plural:', provide a natural plural-aware translation for the target language.
"""

TRANSLATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "plural_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id", "text"],
            },
        },
    },
    "required": ["translations"],
}


def build_translation_generation_config(
    thinking_level: str | None = None,
    *,
    provider: Any = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
    flex_mode: bool = False,
) -> Any:
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=TRANSLATION_RESPONSE_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
        flex_mode=flex_mode,
    )


def build_translation_request_spec(force_non_empty: bool = False) -> TaskRequestSpec:
    non_empty_block = (
        (
            "Every translation must be non-empty.",
            "Never return empty strings. If uncertain, provide your best translation.",
        )
        if force_non_empty
        else ()
    )
    return TaskRequestSpec(
        task_intro="Translate each software localization message item.",
        task_lines=(
            "Translate each message independently.",
            "Do not merge messages.",
        ),
        payload_lines=(
            "The payload includes source language, target language, optional approved vocabulary/glossary, optional project translation rules/instructions, and a `messages` map.",
        ),
        output_lines=(
            "Return only the corrected final JSON.",
            "Keep each translation item's `id` exactly the same as the input key.",
            "Use `plural_texts` only for plural entries when you can provide explicit forms.",
            "For plural entries (source contains `Singular:`/`Plural:` or `item.plural_forms` is present), return non-empty `plural_texts` with exactly `item.plural_forms` forms (or at least 2 if absent).",
            "If the target language effectively has one plural form but multiple slots are required, repeat the same wording in all required plural slots.",
            "For target languages with no plural difference, prefer the source plural form as the basis for translation.",
            "Run a silent vocabulary audit before finalizing each translation and prefer approved terminology when applicable.",
            "If project rules are present, follow them exactly; they are mandatory, not advisory.",
            "Preserve every numeric placeholder like `%d` or `%n` exactly.",
            *non_empty_block,
        ),
    )


def build_translation_request_payload(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    force_non_empty: bool = False,
) -> dict[str, Any]:
    return {
        "project_type": "software_ui_localization",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "vocabulary": vocabulary,
        "translation_rules": translation_rules,
        "force_non_empty": force_non_empty,
        "messages": messages,
    }


def build_translation_request_contents(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    *,
    provider: Any = DEFAULT_PROVIDER,
    force_non_empty: bool = False,
) -> Any:
    task_spec = build_translation_request_spec(force_non_empty=force_non_empty)
    payload = build_translation_request_payload(
        messages=messages,
        source_lang=source_lang,
        target_lang=target_lang,
        vocabulary=vocabulary,
        translation_rules=translation_rules,
        force_non_empty=force_non_empty,
    )
    return build_task_request_contents(
        provider=provider,
        task_spec=task_spec,
        function_name="translation_batch",
        payload=payload,
    )


def build_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    force_non_empty: bool = False,
) -> str:
    return render_text_fallback_prompt(
        task_spec=build_translation_request_spec(force_non_empty=force_non_empty),
        payload=build_translation_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
            force_non_empty=force_non_empty,
        ),
    )


def load_po(file_path: str):
    return _core_load_po(file_path, pofile_loader=polib.pofile)


def load_vocabulary_pairs(path: str | None, label: str = "Vocabulary"):
    return _core_load_vocabulary_pairs(path, label, pofile_loader=polib.pofile)


def read_optional_vocabulary_file(path: str | None, label: str = "Vocabulary"):
    return _core_read_optional_vocabulary_file(path, label, pofile_loader=polib.pofile)


def build_language_code_candidates(target_lang: str) -> List[str]:
    return _core_build_language_code_candidates(target_lang)


def detect_default_text_resource(prefix: str, extension: str, target_lang: str) -> str | None:
    return _core_detect_default_text_resource(prefix, extension, target_lang)


def resolve_resource_path(
    explicit_path: str | None,
    prefix: str,
    extension: str,
    target_lang: str,
) -> str | None:
    return _core_resolve_resource_path(explicit_path, prefix, extension, target_lang)


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    if isinstance(entry, ResxEntryAdapter) and not getattr(entry, "_translate", True):
        return EntryStatus.SKIPPED
    flags = getattr(entry, "flags", None)
    if isinstance(flags, list) and "fuzzy" in flags:
        return EntryStatus.FUZZY
    return EntryStatus.TRANSLATED if entry.translated() else EntryStatus.UNTRANSLATED


def _build_unified_entry(
    entry: Any,
    file_kind: FileKind,
    commit_callback,
) -> UnifiedEntry:
    return _core_build_unified_entry(
        entry=entry,
        file_kind=file_kind,
        status_getter=_entry_status_from_legacy,
        commit_callback=commit_callback,
    )


@dataclass(frozen=True)
class TranslationRunConfig:
    files: List[str]
    source_lang: str
    target_lang: str
    provider: str
    model: str
    thinking_level: str | None
    batch_size: int | None
    parallel_requests: int | None
    vocab: str | None
    rules: str | None
    rules_str: str | None
    retranslate_all: bool
    flex_mode: bool


@dataclass
class TranslationFileJob:
    file_path: str
    file_kind: FileKind
    entries: List[Any]
    save_callback: Callable[[], None] | None
    output_path: str


@dataclass(frozen=True)
class TranslationQueueItem:
    job: TranslationFileJob
    entry: Any


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = (
        "Pre-process and translate PO, TS, RESX, STRINGS, or TXT files using a provider adapter"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input .po, .ts, .resx, .strings, or .txt file(s)",
    )
    add_language_arguments(parser)
    add_provider_arguments(
        parser,
        default_provider_name=DEFAULT_PROVIDER_NAME,
        default_model=DEFAULT_PROVIDER.default_model,
    )
    add_runtime_limit_arguments(parser)
    add_vocabulary_argument(parser)
    add_rules_arguments(
        parser,
        rules_help="Optional translation rules/instructions file (auto: data/locales/<target-lang>/rules.md)",
        rules_str_help="Optional inline translation rules/instructions",
    )
    parser.add_argument(
        "--retranslate-all",
        action="store_true",
        help="Force translation of all translatable messages, not only unfinished/fuzzy ones",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return build_task_parser(configure_parser)


def config_from_args(args: argparse.Namespace) -> TranslationRunConfig:
    return TranslationRunConfig(
        files=list(args.files),
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        provider=args.provider,
        model=resolve_provider_model(args.provider, args.model),
        thinking_level=args.thinking_level,
        batch_size=args.batch_size,
        parallel_requests=args.parallel_requests,
        vocab=args.vocab,
        rules=args.rules,
        rules_str=args.rules_str,
        retranslate_all=args.retranslate_all,
        flex_mode=args.flex_mode,
    )


def load_entries_for_translation(file_path: str):
    try:
        file_kind = detect_file_kind(file_path)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    if file_kind == FileKind.TS:
        return file_kind, *load_ts(file_path)
    if file_kind == FileKind.RESX:
        return file_kind, *load_resx(file_path)
    if file_kind == FileKind.STRINGS:
        return file_kind, *load_strings(file_path)
    if file_kind == FileKind.TXT:
        return file_kind, *load_txt(file_path)
    return file_kind, *load_po(file_path)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def validate_translation_files(file_paths: List[str]) -> FileKind:
    expected_kind: FileKind | None = None
    seen_inputs: set[str] = set()
    seen_outputs: dict[str, str] = {}

    for file_path in file_paths:
        normalized_input = _normalize_path(file_path)
        if normalized_input in seen_inputs:
            raise ValueError(f"Duplicate input file: {file_path}")
        seen_inputs.add(normalized_input)

        file_kind = detect_file_kind(file_path)
        if expected_kind is None:
            expected_kind = file_kind
        elif file_kind != expected_kind:
            raise ValueError("Multi-file translation requires all input files to use the same format.")

        output_path = build_output_path(file_path, file_kind)
        normalized_output = _normalize_path(output_path)
        conflict_input = seen_outputs.get(normalized_output)
        if conflict_input is not None:
            raise ValueError(
                f"Conflicting output path '{output_path}' for inputs '{conflict_input}' and '{file_path}'."
            )
        seen_outputs[normalized_output] = file_path

    if expected_kind is None:
        raise ValueError("At least one input file is required.")
    return expected_kind


def load_translation_jobs(file_paths: List[str]) -> List[TranslationFileJob]:
    jobs: List[TranslationFileJob] = []
    for file_path in file_paths:
        file_kind, entries, save_callback, output_path = load_entries_for_translation(file_path)
        jobs.append(
            TranslationFileJob(
                file_path=file_path,
                file_kind=file_kind,
                entries=entries,
                save_callback=save_callback,
                output_path=output_path,
            )
        )
    return jobs


def build_translation_queue(
    jobs: List[TranslationFileJob],
    *,
    retranslate_all: bool,
) -> List[TranslationQueueItem]:
    queue: List[TranslationQueueItem] = []
    for job in jobs:
        for entry in select_work_items(job.entries, retranslate_all=retranslate_all):
            queue.append(TranslationQueueItem(job=job, entry=entry))
    return queue


def build_batches(work_items: List[Any], parallel_requests: int, batch_size: int) -> List[List[Any]]:
    total = len(work_items)
    all_batches: List[List[Any]] = []
    small_file_threshold = parallel_requests * batch_size
    if total < small_file_threshold:
        worker_count = parallel_requests if total >= parallel_requests * MIN_ITEMS_PER_WORKER else total // MIN_ITEMS_PER_WORKER
        if worker_count < 2:
            print(f"Found {total} items to translate. Small file mode: using 1 batch (minimum items/worker not met).")
            return [work_items]

        base = total // worker_count
        remainder = total % worker_count
        start = 0
        for worker_index in range(worker_count):
            size = base + (1 if worker_index < remainder else 0)
            end = start + size
            all_batches.append(work_items[start:end])
            start = end
        print(
            f"Found {total} items to translate. Small file mode: split evenly into {worker_count} parallel batches "
            f"(min batch size: {min(len(batch) for batch in all_batches)})."
        )
        return all_batches

    batches = math.ceil(total / batch_size)
    all_batches = build_fixed_batches(work_items, batch_size)
    print(f"Found {total} items to translate. Running up to {parallel_requests} batch requests in parallel.")
    return all_batches


async def run_translation_batches(
    *,
    provider,
    client: Any,
    model: str,
    translation_config: Any,
    all_batches: List[List[TranslationQueueItem]],
    total: int,
    parallel_requests: int,
    source_lang: str,
    target_lang: str,
    vocabulary_text: str | None,
    project_rules: str | None,
) -> int:
    batches = len(all_batches)

    translated_count = 0
    completed_batches = 0

    async def process_batch(batch_index: int, batch: List[TranslationQueueItem]) -> Dict[str, TranslationResult]:
        msg_map = {
            str(i): build_prompt_message_payload(item.entry)
            for i, item in enumerate(batch)
        }
        contents = build_translation_request_contents(
            msg_map,
            source_lang,
            target_lang,
            vocabulary_text,
            project_rules,
            provider=provider,
        )
        response = await provider.generate_with_retry(
            client=client,
            model=model,
            contents=contents,
            batch_label=f"batch {batch_index + 1}/{batches}",
            max_attempts=5,
            config=translation_config,
        )
        translations = parse_response(response)
        missing_indices = [i for i in range(len(batch)) if not translation_has_content(translations.get(str(i)))]
        if missing_indices:
            print(
                f"  Warning [batch {batch_index + 1}/{batches}]: "
                f"{len(missing_indices)} items missing from response. Retrying them..."
            )
            retry_map = {
                str(idx): build_prompt_message_payload(batch[idx].entry)
                for idx in missing_indices
            }
            retry_contents = build_translation_request_contents(
                retry_map,
                source_lang,
                target_lang,
                vocabulary_text,
                project_rules,
                provider=provider,
                force_non_empty=True,
            )
            try:
                retry_resp = await provider.generate_with_retry(
                    client=client,
                    model=model,
                    contents=retry_contents,
                    batch_label=f"batch {batch_index + 1}/{batches} missing-items",
                    max_attempts=3,
                    config=translation_config,
                )
                translations.update(parse_response(retry_resp))
            except Exception as exc:
                print(f"  Retry failed [batch {batch_index + 1}/{batches}]: {exc}")
        return translations

    def on_batch_completed(
        batch_index: int,
        batch: List[TranslationQueueItem],
        translations: Dict[str, TranslationResult],
    ) -> None:
        nonlocal translated_count, completed_batches
        batch_translated = 0
        touched_jobs: dict[str, TranslationFileJob] = {}
        for i, item in enumerate(batch):
            result = translations.get(str(i))
            entry = item.entry
            if result and apply_translation_to_entry(entry, result):
                if "fuzzy" not in entry.flags:
                    entry.flags.append("fuzzy")
                batch_translated += 1
                touched_jobs[item.job.output_path] = item.job

        translated_count += batch_translated
        completed_batches += 1
        percent = (translated_count / total) * 100
        print(
            f"Progress: {percent:.1f}% ({translated_count}/{total}), "
            f"completed batches: {completed_batches}/{batches} "
            f"(latest: {batch_index + 1}/{batches})"
        )
        for job in touched_jobs.values():
            if job.save_callback:
                job.save_callback()

    await run_parallel_batches(
        batches=all_batches,
        parallel_requests=parallel_requests,
        process_batch=process_batch,
        on_batch_completed=on_batch_completed,
    )
    return translated_count


def run_translation(config: TranslationRunConfig) -> None:
    try:
        file_kind = validate_translation_files(config.files)
        file_jobs = load_translation_jobs(config.files)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    runtime_context = build_task_runtime_context(
        provider_name=config.provider,
        target_lang=config.target_lang,
        flex_mode=config.flex_mode,
        explicit_vocab_path=config.vocab,
        explicit_rules_path=config.rules,
        inline_rules=config.rules_str,
        get_translation_provider_fn=get_translation_provider,
    )
    provider = runtime_context.provider
    client = runtime_context.client
    resource_context = runtime_context.resources

    work_items = build_translation_queue(file_jobs, retranslate_all=config.retranslate_all)
    total = len(work_items)
    if total == 0:
        print("No translatable messages found." if config.retranslate_all else "No untranslated or fuzzy messages found.")
        return

    try:
        batch_size, parallel_requests, limits_mode = resolve_runtime_limits(
            total_items=total,
            batch_size_arg=config.batch_size,
            parallel_arg=config.parallel_requests,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    translation_config = build_translation_generation_config(
        config.thinking_level,
        provider=provider,
        flex_mode=config.flex_mode,
    )

    print_startup_configuration(
        ("Input files", len(file_jobs)),
        ("File kind", file_kind.value),
        ("Provider", provider.name),
        ("Model", config.model),
        ("Flex mode", "yes" if config.flex_mode and getattr(provider, "supports_flex_mode", False) else "no"),
        ("Thinking level", config.thinking_level or "provider default"),
        ("Parallel requests", parallel_requests),
        ("Batch size", batch_size),
        ("Limits mode", limits_mode),
        ("Retranslate all", "yes" if config.retranslate_all else "no"),
        ("Vocabulary source", resource_context.vocabulary_source),
        ("Rules source", resource_context.rules_source or "none"),
    )
    all_batches = build_batches(work_items, parallel_requests, batch_size)
    print(f"Total batches: {len(all_batches)}")

    try:
        asyncio.run(
            run_translation_batches(
                provider=provider,
                client=client,
                model=config.model,
                translation_config=translation_config,
                all_batches=all_batches,
                total=total,
                parallel_requests=parallel_requests,
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                vocabulary_text=resource_context.vocabulary_text,
                project_rules=resource_context.project_rules,
            )
        )
    except RuntimeError as exc:
        sys.exit(str(exc))

    for job in file_jobs:
        if job.save_callback:
            job.save_callback()

    print("\nTranslation complete.")
    if len(file_jobs) == 1:
        print(f"Saved file: {file_jobs[0].output_path}")
    else:
        print(f"Saved files ({len(file_jobs)}):")
        for job in file_jobs:
            print(f"- {job.output_path}")
    print("All AI-generated translations are marked as fuzzy/unfinished for human review.")


def run_from_args(args: argparse.Namespace) -> None:
    run_translation(config_from_args(args))


def main(argv: list[str] | None = None) -> None:
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
