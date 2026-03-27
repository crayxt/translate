#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

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
from core.runtime import (
    MIN_ITEMS_PER_WORKER,
    add_thinking_level_argument,
    build_thinking_config,
    resolve_runtime_limits,
)

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
) -> Any:
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=TRANSLATION_RESPONSE_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
    )


def build_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    force_non_empty: bool = False,
) -> str:
    vocab_block = f"\nProject vocabulary (mandatory):\n{vocabulary}\n" if vocabulary else ""
    rules_block = (
        f"\nProject translation rules/instructions (mandatory when present):\n{translation_rules}\n"
        if translation_rules
        else ""
    )
    non_empty_block = (
        "- Every translation must be non-empty.\n"
        "- Never return empty strings. If uncertain, provide your best translation.\n"
        if force_non_empty
        else ""
    )
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    return f"""
Project context:
This is a software application UI localization project.
Source language: {source_lang}
Target language: {target_lang}
{vocab_block}
{rules_block}

Instructions:
- Translate each message independently
- Do NOT merge messages
- Return ONLY valid JSON, no Markdown fences or extra text
- Keep each translation item's "id" exactly the same as the input key
- Use "plural_texts" only for plural entries when you can provide explicit forms
- For plural entries (source contains 'Singular:'/'Plural:' or item.plural_forms is present),
  return non-empty "plural_texts" with exactly item.plural_forms forms (or at least 2 if absent)
- If the target language effectively has one plural form but multiple slots are required,
  repeat the same wording in all required plural slots
- For target languages with no plural difference, prefer the source plural form as the basis for translation
- run a silent vocabulary audit before finalizing each translation and prefer approved terminology when applicable
- If project rules are present, follow them exactly; it is mandatory, not advisory
- Return only the corrected final JSON.
- Preserve every numeric placeholder like %d or %n exactly.
{non_empty_block}

Input messages:
{messages_json}
""".strip()


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
    file: str
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


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = (
        "Pre-process and translate PO, TS, RESX, STRINGS, or TXT files using a provider adapter"
    )
    parser.add_argument("file", help="Input .po, .ts, .resx, .strings, or .txt file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER_NAME,
        help=f"Model provider (default: {DEFAULT_PROVIDER_NAME})",
    )
    parser.add_argument("--model", default=DEFAULT_PROVIDER.default_model)
    add_thinking_level_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent requests (auto if omitted)")
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt). Supports .txt and glossary .po",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional translation rules/instructions file (auto: data/<target-lang>/rules.md)",
    )
    parser.add_argument("--rules-str", default=None, help="Optional inline translation rules/instructions")
    parser.add_argument(
        "--retranslate-all",
        action="store_true",
        help="Force translation of all translatable messages, not only unfinished/fuzzy ones",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser())


def config_from_args(args: argparse.Namespace) -> TranslationRunConfig:
    return TranslationRunConfig(
        file=args.file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        provider=args.provider,
        model=args.model,
        thinking_level=args.thinking_level,
        batch_size=args.batch_size,
        parallel_requests=args.parallel_requests,
        vocab=args.vocab,
        rules=args.rules,
        rules_str=args.rules_str,
        retranslate_all=args.retranslate_all,
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


def build_resource_context(config: TranslationRunConfig) -> tuple[str | None, str | None, str | None, str]:
    vocabulary_path = resolve_resource_path(config.vocab, "vocab", "txt", config.target_lang)
    rules_path = resolve_resource_path(config.rules, "rules", "md", config.target_lang)
    vocabulary_text = read_optional_vocabulary_file(vocabulary_path, "Vocabulary")
    rules_text = read_optional_text_file(rules_path, "Rules")
    project_rules = merge_project_rules(rules_text, config.rules_str)
    rules_source = detect_rules_source(rules_path, rules_text, config.rules_str)
    vocabulary_source = f"file:{vocabulary_path}" if vocabulary_text and vocabulary_path else "none"
    return vocabulary_text, project_rules, rules_source, vocabulary_source


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
    all_batches = [work_items[i * batch_size:(i + 1) * batch_size] for i in range(batches)]
    print(f"Found {total} items to translate. Running up to {parallel_requests} batch requests in parallel.")
    return all_batches


async def run_translation_batches(
    *,
    provider,
    client: Any,
    model: str,
    translation_config: Any,
    all_batches: List[List[Any]],
    total: int,
    parallel_requests: int,
    source_lang: str,
    target_lang: str,
    vocabulary_text: str | None,
    project_rules: str | None,
    save_callback,
) -> int:
    batches = len(all_batches)

    async def process_batch(batch_index: int, batch: List[Any], sem: asyncio.Semaphore):
        async with sem:
            msg_map = {str(i): build_prompt_message_payload(entry) for i, entry in enumerate(batch)}
            prompt = build_prompt(msg_map, source_lang, target_lang, vocabulary_text, project_rules)
            response = await provider.generate_with_retry(
                client=client,
                model=model,
                prompt=prompt,
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
                retry_map = {str(idx): build_prompt_message_payload(batch[idx]) for idx in missing_indices}
                retry_prompt = build_prompt(
                    retry_map,
                    source_lang,
                    target_lang,
                    vocabulary_text,
                    project_rules,
                    force_non_empty=True,
                )
                try:
                    retry_resp = await provider.generate_with_retry(
                        client=client,
                        model=model,
                        prompt=retry_prompt,
                        batch_label=f"batch {batch_index + 1}/{batches} missing-items",
                        max_attempts=3,
                        config=translation_config,
                    )
                    translations.update(parse_response(retry_resp))
                except Exception as exc:
                    print(f"  Retry failed [batch {batch_index + 1}/{batches}]: {exc}")
            return batch_index, batch, translations

    translated_count = 0
    sem = asyncio.Semaphore(parallel_requests)
    tasks = [asyncio.create_task(process_batch(batch_index, batch, sem)) for batch_index, batch in enumerate(all_batches)]
    completed_batches = 0

    for finished in asyncio.as_completed(tasks):
        batch_index, batch, translations = await finished
        batch_translated = 0
        for i, entry in enumerate(batch):
            result = translations.get(str(i))
            if result and apply_translation_to_entry(entry, result):
                if "fuzzy" not in entry.flags:
                    entry.flags.append("fuzzy")
                batch_translated += 1

        translated_count += batch_translated
        completed_batches += 1
        percent = (translated_count / total) * 100
        print(
            f"Progress: {percent:.1f}% ({translated_count}/{total}), "
            f"completed batches: {completed_batches}/{batches} "
            f"(latest: {batch_index + 1}/{batches})"
        )
        if save_callback:
            save_callback()
    return translated_count


def run_translation(config: TranslationRunConfig) -> None:
    provider = get_translation_provider(config.provider)
    client = provider.create_client_from_env()
    vocabulary_text, project_rules, rules_source, vocabulary_source = build_resource_context(config)
    _, entries, save_callback, output_path = load_entries_for_translation(config.file)

    work_items = select_work_items(entries, retranslate_all=config.retranslate_all)
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
    )

    print("Startup configuration:")
    print(f"  Provider: {provider.name}")
    print(f"  Model: {config.model}")
    print(f"  Thinking level: {config.thinking_level or 'provider default'}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")
    print(f"  Retranslate all: {'yes' if config.retranslate_all else 'no'}")
    print(f"  Vocabulary source: {vocabulary_source}")
    print(f"  Rules source: {rules_source or 'none'}")
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
                vocabulary_text=vocabulary_text,
                project_rules=project_rules,
                save_callback=save_callback,
            )
        )
    except RuntimeError as exc:
        sys.exit(str(exc))

    if save_callback:
        save_callback()

    print("\nTranslation complete.")
    print(f"Saved file: {output_path}")
    print("All AI-generated translations are marked as fuzzy/unfinished for human review.")


def run_from_args(args: argparse.Namespace) -> None:
    run_translation(config_from_args(args))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
