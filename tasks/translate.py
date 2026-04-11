#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import polib

from core.entries import (
    TranslationResult,
    TranslationWarning,
    TRANSLATION_WARNING_CODES,
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
    load_android_xml,
    load_xliff,
    load_paired_android_xml,
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
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, TranslationProvider, get_translation_provider
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
from core.system_instructions import (
    SHARED_GLOSSARY_SENSE_RULES,
    SHARED_LOCALIZATION_INVARIANTS,
    join_instruction_sections,
    render_instruction_section,
)
from core.term_extraction import (
    ScopedVocabularyEntry,
    build_relevant_vocabulary,
    build_scoped_vocabulary_entries,
)

TRANSLATABLE_INPUT_EXTENSIONS = frozenset(
    {
        ".po",
        ".pot",
        ".xlf",
        ".xliff",
        ".ts",
        ".resx",
        ".strings",
        ".txt",
        ".xml",
    }
)
TRANSLATION_SCAN_EXCLUDED_OUTPUT_SUFFIXES = frozenset(
    {
        ".ai-translated",
        ".glossary",
        ".missing-terms",
        ".prototype-glossary",
        ".prototype-missing-terms",
    }
)
TRANSLATION_SCAN_TOOLKIT_ROOT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "core",
        "data",
        "docs",
        "logs",
        "tasks",
        "tests",
    }
)
TRANSLATION_SCAN_TOOLKIT_ROOT_EXCLUDED_FILENAMES = frozenset({"requirements.txt"})
TOOLKIT_PROJECT_ROOT = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

build_prompt_message_payload = _build_prompt_message_payload

SYSTEM_INSTRUCTION = join_instruction_sections(
    """
    You are a professional software localization translator.

    TRANSLATION REQUIREMENTS:
    - Do NOT reorder placeholders
    - Do NOT add, remove, soften, intensify, or reinterpret meaning
    - Do NOT translate protected tokens such as product names, brand names, API names, code identifiers, command flags, file extensions, paths, or variable-like strings
    - Keep original punctuation and capitalization style unless the target language requires a minimal grammatical adjustment
    - If source text is ALL CAPS, keep translation ALL CAPS
    - Translate ONLY the message text, not metadata
    """,
    SHARED_LOCALIZATION_INVARIANTS,
    SHARED_GLOSSARY_SENSE_RULES,
    render_instruction_section(
        "FORMATTING",
        "If the source text contains \\n line-wrapping markers, preserve them in the translation and try to keep lines at roughly similar lengths",
        "Avoid unnatural CamelCase in the target language unless source uses intentional branded casing",
    ),
    render_instruction_section(
        "PLURALS",
        "For plural messages, use `source_singular`, `source_plural`, `plural_forms`, and `plural_slots` to produce natural target-language `plural_texts`.",
    ),
)
from core.task_issues import build_task_issue_schema, serialize_task_issue

TRANSLATION_WARNING_CODE_GUIDANCE: Dict[str, str] = {
    "translate.ambiguous_term": "a source term has multiple plausible senses or parts of speech",
    "translate.unclear_source_meaning": "the source meaning is unclear or underspecified",
    "translate.glossary_variant_choice": "multiple approved glossary variants existed and one was chosen",
    "translate.possible_untranslated_token": "a token may have been intentionally preserved or may still need review",
    "translate.placeholder_attention": "placeholders or protected tokens required extra care",
    "translate.length_or_ui_fit_risk": "the translation may be too long or risky for UI fit",
}

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
                    "warnings": {
                        "type": "array",
                        "items": build_task_issue_schema(
                            TRANSLATION_WARNING_CODES,
                            allowed_severities=("warning", "info"),
                        ),
                    },
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def build_translation_message_payload(
    entry: Any,
    scoped_vocabulary_entries: List[ScopedVocabularyEntry],
) -> Dict[str, Any]:
    """Build one translation payload item with message-scoped glossary hints."""
    payload = build_prompt_message_payload(entry)
    source_for_vocabulary = payload.get("source")
    if source_for_vocabulary is None:
        source_parts = [
            payload.get("source_singular"),
            payload.get("source_plural"),
        ]
        source_for_vocabulary = " ".join(
            part for part in source_parts if isinstance(part, str) and part.strip()
        )
    relevant_vocabulary = build_relevant_vocabulary(source_for_vocabulary, scoped_vocabulary_entries)
    if relevant_vocabulary:
        payload["relevant_vocabulary"] = relevant_vocabulary
    return payload


def build_translation_generation_config(
    thinking_level: str | None = None,
    *,
    provider: TranslationProvider = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
    flex_mode: bool = False,
) -> Any:
    """Build the provider generation config for translation batches."""
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=TRANSLATION_RESPONSE_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
        flex_mode=flex_mode,
    )


def build_translation_request_spec(force_non_empty: bool = False) -> TaskRequestSpec:
    """Describe the structured contract for translation batches."""
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
            "Each non-plural message includes `source` and may also include `context`, `note`, and `relevant_vocabulary`.",
            "Each plural message includes `source_singular`, `source_plural`, `plural_forms`, and `plural_slots`, and may also include `context`, `note`, and `relevant_vocabulary`.",
            "When present, `relevant_vocabulary` is a message-scoped list of approved term translations and may contain multiple variants for the same source term.",
        ),
        output_lines=(
            "Return only the corrected final JSON.",
            "Keep each translation item's `id` exactly the same as the input key.",
            "Use `plural_texts` only for plural entries when you can provide explicit forms.",
            "Use `warnings` only when a message has a real ambiguity, unclear meaning, risky glossary choice, or another review-worthy concern.",
            "Each warning must be an object with `code`, `message`, and `severity`.",
            f"Allowed warning codes: {', '.join(TRANSLATION_WARNING_CODES)}.",
            *tuple(
                f"Use `{code}` when {description}."
                for code, description in TRANSLATION_WARNING_CODE_GUIDANCE.items()
            ),
            "Use severity `warning` for real ambiguity, uncertainty, or human-review risk.",
            "Use severity `info` for notable but non-risk notes, such as preserved structure or a confident glossary choice worth surfacing.",
            "Keep each warning `message` short and specific to that message.",
            "Do not add warnings for routine confident translations.",
            "For plural entries (`item.source_singular`/`item.source_plural` present), return non-empty `plural_texts` with exactly `item.plural_forms` forms (or at least 2 if absent).",
            "Treat `item.source_singular` and `item.source_plural` as separate source forms that must be translated consistently.",
            "Align `plural_texts` to the order of `item.plural_slots`.",
            "For plural entries, do not put labeled `Singular:`/`Plural:` output inside `text`; put the actual translated forms into `plural_texts` only.",
            "If the target language effectively has one plural form but multiple slots are required, repeat the same wording in all required plural slots.",
            "Run a silent vocabulary audit before finalizing each translation and prefer approved terminology when applicable.",
            "When `message.relevant_vocabulary` is present, prefer those term translations for that message.",
            "Use `message.context` and `message.note` to disambiguate meaning and select the correct approved terminology for that message.",
            "If multiple `message.relevant_vocabulary` entries share the same `source_term`, choose the variant whose `part_of_speech` and `context_note` best match `message.context` and `message.note`.",
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
    """Build the structured payload for one translation batch."""
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
    provider: TranslationProvider = DEFAULT_PROVIDER,
    force_non_empty: bool = False,
) -> Any:
    """Build provider-native request contents for a translation batch."""
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
    """Render the plain-text fallback prompt for one translation batch."""
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
    """Load a PO file through the shared format adapter."""
    return _core_load_po(file_path, pofile_loader=polib.pofile)


def load_vocabulary_pairs(
    path: str | None,
    label: str = "Vocabulary",
    *,
    target_lang: str | None = None,
):
    """Load normalized vocabulary pairs from text, PO, or TBX resources."""
    return _core_load_vocabulary_pairs(
        path,
        label,
        pofile_loader=polib.pofile,
        target_lang=target_lang,
    )


def read_optional_vocabulary_file(
    path: str | None,
    label: str = "Vocabulary",
    *,
    target_lang: str | None = None,
):
    """Read the full optional vocabulary resource as text for prompting."""
    return _core_read_optional_vocabulary_file(
        path,
        label,
        pofile_loader=polib.pofile,
        target_lang=target_lang,
    )


def build_language_code_candidates(target_lang: str) -> List[str]:
    """Return likely language-code variants for resource auto-discovery."""
    return _core_build_language_code_candidates(target_lang)


def detect_default_text_resource(
    prefix: str,
    extension: str,
    target_lang: str,
    *,
    allow_directory: bool = False,
) -> str | None:
    """Auto-detect a locale resource path under the project conventions."""
    return _core_detect_default_text_resource(
        prefix,
        extension,
        target_lang,
        allow_directory=allow_directory,
    )


def resolve_resource_path(
    explicit_path: str | None,
    prefix: str,
    extension: str,
    target_lang: str,
    *,
    allow_directory: bool = False,
) -> str | None:
    """Prefer an explicit resource path, otherwise fall back to auto-discovery."""
    return _core_resolve_resource_path(
        explicit_path,
        prefix,
        extension,
        target_lang,
        allow_directory=allow_directory,
    )


def _entry_status_from_legacy(entry: Any) -> EntryStatus:
    """Infer unified translation status from a legacy entry adapter."""
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
    """Wrap one legacy entry object in the shared unified-entry model."""
    return _core_build_unified_entry(
        entry=entry,
        file_kind=file_kind,
        status_getter=_entry_status_from_legacy,
        commit_callback=commit_callback,
    )


@dataclass(frozen=True)
class TranslationRunConfig:
    """All runtime options required to execute a translation command."""
    files: List[str]
    source_file: str | None
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
    warnings_report: bool


@dataclass
class TranslationFileJob:
    """Loaded file state plus output metadata for one translation target."""
    file_path: str
    file_kind: FileKind
    entries: List[UnifiedEntry]
    save_callback: Callable[[], None] | None
    output_path: str


@dataclass(frozen=True)
class TranslationQueueItem:
    """One entry scheduled for translation together with its owning file job."""
    job: TranslationFileJob
    entry: UnifiedEntry


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure the standalone CLI for translation runs."""
    parser.description = (
        "Pre-process and translate PO, XLIFF, TS, RESX, STRINGS, TXT, or Android XML files using a provider adapter"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input .po, .xlf/.xliff, .ts, .resx, .strings, .txt, or Android .xml file(s) or directory tree(s)",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Required for Android .xml translation runs; use the English source XML that matches the target XML.",
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
    parser.add_argument(
        "--warnings-report",
        action="store_true",
        help="Write a separate JSON sidecar with per-message translation warnings and ambiguities.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for translation runs."""
    return build_task_parser(configure_parser)


def config_from_args(args: argparse.Namespace) -> TranslationRunConfig:
    """Convert parsed CLI arguments into the normalized runtime config."""
    return TranslationRunConfig(
        files=list(args.files),
        source_file=args.source_file,
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
        warnings_report=args.warnings_report,
    )


def build_translation_warnings_output_path(output_path: str) -> str:
    """Build the sidecar JSON path for translation warning reports."""
    root, _ = os.path.splitext(output_path)
    return f"{root}.translation-warnings.json"


def _is_supported_translation_input_file(path: str) -> bool:
    """Return whether a path has a supported translation input extension."""
    return os.path.splitext(path)[1].lower() in TRANSLATABLE_INPUT_EXTENSIONS


def _normalize_scan_path(path: str) -> str:
    """Normalize a path for stable case-insensitive comparisons."""
    return os.path.normcase(os.path.abspath(path))


def _looks_like_generated_translation_artifact(path: str) -> bool:
    """Return whether a path looks like a generated translation-side artifact."""
    root, _ext = os.path.splitext(path)
    normalized_root = str(root or "").strip().lower()
    return any(
        normalized_root.endswith(suffix)
        for suffix in TRANSLATION_SCAN_EXCLUDED_OUTPUT_SUFFIXES
    )


def _should_skip_toolkit_root_scan_file(scan_root: str, candidate: str) -> bool:
    """Return whether a candidate should be pruned when scanning the toolkit repo root."""
    if _normalize_scan_path(scan_root) != TOOLKIT_PROJECT_ROOT:
        return False
    candidate_parent = _normalize_scan_path(os.path.dirname(candidate))
    if candidate_parent != TOOLKIT_PROJECT_ROOT:
        return False
    return os.path.basename(candidate).lower() in TRANSLATION_SCAN_TOOLKIT_ROOT_EXCLUDED_FILENAMES


def _collect_translation_files_from_directory(root_dir: str) -> List[str]:
    """Recursively collect supported translation inputs from a directory tree."""
    collected: List[str] = []
    normalized_root_dir = _normalize_scan_path(root_dir)
    scanning_toolkit_root = normalized_root_dir == TOOLKIT_PROJECT_ROOT
    for current_root, dirnames, filenames in os.walk(root_dir):
        dirnames.sort(key=str.lower)
        if scanning_toolkit_root and _normalize_scan_path(current_root) == TOOLKIT_PROJECT_ROOT:
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname.lower() not in TRANSLATION_SCAN_TOOLKIT_ROOT_EXCLUDED_DIRS
            ]
        for filename in sorted(filenames, key=str.lower):
            candidate = os.path.join(current_root, filename)
            if _is_supported_translation_input_file(candidate):
                if _looks_like_generated_translation_artifact(candidate):
                    continue
                if _should_skip_toolkit_root_scan_file(root_dir, candidate):
                    continue
                collected.append(candidate)
    return collected


def resolve_translation_input_paths(input_paths: List[str]) -> List[str]:
    """Expand file and directory inputs into a validated translation file list."""
    resolved: List[str] = []
    for raw_path in input_paths:
        cleaned_path = str(raw_path or "").strip()
        if not cleaned_path:
            continue
        if os.path.isdir(cleaned_path):
            directory_files = _collect_translation_files_from_directory(cleaned_path)
            if not directory_files:
                raise ValueError(f"No supported translation files found under directory: {cleaned_path}")
            resolved.extend(directory_files)
            continue
        if not os.path.exists(cleaned_path):
            raise ValueError(f"Input file does not exist: {cleaned_path}")
        if not os.path.isfile(cleaned_path):
            raise ValueError(f"Input path is not a file: {cleaned_path}")
        resolved.append(cleaned_path)

    if not resolved:
        raise ValueError("At least one input file or directory is required.")
    return resolved


def build_translation_warning_item(
    entry: Any,
    result: TranslationResult,
    scoped_vocabulary_entries: List[ScopedVocabularyEntry],
) -> Dict[str, Any]:
    """Serialize one translated entry and its warnings for the sidecar report."""
    payload = build_translation_message_payload(entry, scoped_vocabulary_entries)
    source_text = payload.get("source")
    if not isinstance(source_text, str) or not source_text.strip():
        source_text = build_entry_source_text(entry)

    translation_text = result.text
    if not _is_non_empty_text(translation_text) and result.plural_texts:
        translation_text = next(
            (text for text in result.plural_texts if _is_non_empty_text(text)),
            "",
        )

    item: Dict[str, Any] = {
        "source": source_text,
        "translation": translation_text,
        "warnings": [
            serialize_task_issue(warning)
            for warning in result.warnings
        ],
    }
    if result.plural_texts:
        item["plural_texts"] = list(result.plural_texts)
    for field_name in (
        "source_singular",
        "source_plural",
        "context",
        "note",
        "relevant_vocabulary",
        "plural_forms",
        "plural_slots",
    ):
        if field_name in payload:
            item[field_name] = payload[field_name]
    return item


def write_translation_warning_report(
    *,
    job: TranslationFileJob,
    warning_items: List[Dict[str, Any]],
    provider_name: str,
    model: str,
    source_lang: str,
    target_lang: str,
    source_file: str | None = None,
) -> str:
    """Write the warning sidecar JSON for one translated output file."""
    out_path = build_translation_warnings_output_path(job.output_path)
    payload: Dict[str, Any] = {
        "source_file": job.file_path,
        "output_file": job.output_path,
        "file_kind": job.file_kind.value,
        "provider": provider_name,
        "model": model,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning_message_count": len(warning_items),
        "messages": warning_items,
    }
    if source_file:
        payload["paired_source_file"] = source_file
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return out_path


def load_entries_for_translation(file_path: str, source_file: str | None = None):
    """Load entries, save hooks, and warnings for one translation input file."""
    try:
        file_kind = detect_file_kind(file_path)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    try:
        if file_kind == FileKind.ANDROID_XML:
            if not source_file:
                raise ValueError(
                    "--source-file is required for .xml translation runs because the translated file "
                    "does not retain the original source text."
                )
            entries, save_callback, output_path, warnings = load_paired_android_xml(source_file, file_path)
            return file_kind, entries, save_callback, output_path, warnings
        if file_kind == FileKind.XLIFF:
            return file_kind, *load_xliff(file_path), []
        if file_kind == FileKind.TS:
            return file_kind, *load_ts(file_path), []
        if file_kind == FileKind.RESX:
            return file_kind, *load_resx(file_path), []
        if file_kind == FileKind.STRINGS:
            return file_kind, *load_strings(file_path), []
        if file_kind == FileKind.TXT:
            return file_kind, *load_txt(file_path), []
        return file_kind, *load_po(file_path), []
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Failed to load translation input '{file_path}': {exc}") from exc


def _normalize_path(path: str) -> str:
    """Normalize a path for duplicate and conflict detection."""
    return os.path.normcase(os.path.abspath(path))


def validate_translation_files(file_paths: List[str], source_file: str | None = None) -> FileKind:
    """Validate translation inputs and ensure the run uses one compatible file kind."""
    file_paths = resolve_translation_input_paths(file_paths)
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

    if expected_kind == FileKind.ANDROID_XML:
        if len(file_paths) != 1:
            raise ValueError("Android .xml translation currently supports one target file at a time.")
        if not source_file:
            raise ValueError(
                "--source-file is required for .xml translation runs because the translated file "
                "does not retain the original source text."
            )
        if not os.path.exists(source_file):
            raise ValueError(f"Source file does not exist: {source_file}")
        if not os.path.isfile(source_file):
            raise ValueError(f"Source file is not a file: {source_file}")
        source_kind = detect_file_kind(source_file)
        if source_kind != expected_kind:
            raise ValueError(
                f"--source-file type mismatch: expected .{expected_kind.value}, got .{source_kind.value}"
            )
    elif source_file:
        raise ValueError("--source-file is currently supported only for Android .xml translation runs.")

    return expected_kind


def load_translation_jobs(file_paths: List[str], source_file: str | None = None) -> List[TranslationFileJob]:
    """Load all translation inputs into file jobs with output metadata."""
    jobs: List[TranslationFileJob] = []
    for file_path in file_paths:
        file_kind, entries, save_callback, output_path, warnings = load_entries_for_translation(
            file_path,
            source_file=source_file,
        )
        for warning in warnings:
            print(f"Warning: {warning}")
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
    """Flatten loaded file jobs into the actual per-entry translation queue."""
    queue: List[TranslationQueueItem] = []
    for job in jobs:
        for entry in select_work_items(job.entries, retranslate_all=retranslate_all):
            queue.append(TranslationQueueItem(job=job, entry=entry))
    return queue


def build_batches(
    work_items: List[TranslationQueueItem],
    parallel_requests: int,
    batch_size: int,
) -> List[List[TranslationQueueItem]]:
    """Split translation work into batches, including a small-file balancing mode."""
    total = len(work_items)
    all_batches: List[List[TranslationQueueItem]] = []
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
    provider: TranslationProvider,
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
    scoped_vocabulary_entries: List[ScopedVocabularyEntry],
    warning_items_by_output_path: Dict[str, List[Dict[str, Any]]] | None = None,
) -> tuple[int, set[str]]:
    """Run translation batches, apply results, and track touched outputs."""
    batches = len(all_batches)

    translated_count = 0
    completed_batches = 0
    touched_output_paths: set[str] = set()

    async def process_batch(batch_index: int, batch: List[TranslationQueueItem]) -> Dict[str, TranslationResult]:
        msg_map = {
            str(i): build_translation_message_payload(item.entry, scoped_vocabulary_entries)
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
                str(idx): build_translation_message_payload(batch[idx].entry, scoped_vocabulary_entries)
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
            if result and result.warnings and warning_items_by_output_path is not None:
                warning_items_by_output_path.setdefault(item.job.output_path, []).append(
                    build_translation_warning_item(entry, result, scoped_vocabulary_entries)
                )
            if result and apply_translation_to_entry(entry, result):
                if "fuzzy" not in entry.flags:
                    entry.flags.append("fuzzy")
                batch_translated += 1
                touched_jobs[item.job.output_path] = item.job
                touched_output_paths.add(item.job.output_path)

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
    return translated_count, touched_output_paths


def run_translation(config: TranslationRunConfig) -> None:
    """Execute a full translation run from normalized runtime config."""
    try:
        resolved_files = resolve_translation_input_paths(config.files)
        file_kind = validate_translation_files(resolved_files, source_file=config.source_file)
        file_jobs = load_translation_jobs(resolved_files, source_file=config.source_file)
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
    scoped_vocabulary_entries = build_scoped_vocabulary_entries(resource_context.vocabulary_text)

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
        ("Warnings report", "yes" if config.warnings_report else "no"),
        ("Vocabulary source", resource_context.vocabulary_source),
        ("Scoped vocabulary entries", len(scoped_vocabulary_entries)),
        ("Rules source", resource_context.rules_source or "none"),
        ("Source file", config.source_file or "embedded in input file"),
    )
    all_batches = build_batches(work_items, parallel_requests, batch_size)
    print(f"Total batches: {len(all_batches)}")
    work_output_paths = {item.job.output_path for item in work_items}
    warning_items_by_output_path: Dict[str, List[Dict[str, Any]]] | None = (
        {job.output_path: [] for job in file_jobs if job.output_path in work_output_paths}
        if config.warnings_report
        else None
    )

    try:
        _translated_count, touched_output_paths = asyncio.run(
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
                scoped_vocabulary_entries=scoped_vocabulary_entries,
                warning_items_by_output_path=warning_items_by_output_path,
            )
        )
    except RuntimeError as exc:
        sys.exit(str(exc))

    saved_jobs = [job for job in file_jobs if job.output_path in touched_output_paths]
    for job in saved_jobs:
        if job.save_callback:
            job.save_callback()

    warning_report_paths: List[str] = []
    if warning_items_by_output_path is not None:
        for job in saved_jobs:
            warning_report_paths.append(
                write_translation_warning_report(
                    job=job,
                    warning_items=warning_items_by_output_path.get(job.output_path, []),
                    provider_name=provider.name,
                    model=config.model,
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    source_file=config.source_file,
                )
            )

    print("\nTranslation complete.")
    skipped_without_work = len(file_jobs) - len(work_output_paths)
    if skipped_without_work > 0:
        label = "file" if skipped_without_work == 1 else "files"
        print(f"Skipped {skipped_without_work} fully translated {label} with no work items.")
    if not saved_jobs:
        print("No output files were written.")
    elif len(saved_jobs) == 1:
        print(f"Saved file: {saved_jobs[0].output_path}")
    else:
        print(f"Saved files ({len(saved_jobs)}):")
        for job in saved_jobs:
            print(f"- {job.output_path}")
    if warning_report_paths:
        if len(warning_report_paths) == 1:
            print(f"Saved warnings report: {warning_report_paths[0]}")
        else:
            print(f"Saved warnings reports ({len(warning_report_paths)}):")
            for path in warning_report_paths:
                print(f"- {path}")
    print("All AI-generated translations are marked as fuzzy/unfinished for human review.")


def run_from_args(args: argparse.Namespace) -> None:
    """Execute translation from parsed CLI arguments."""
    try:
        config = config_from_args(args)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    run_translation(config)


def main(argv: list[str] | None = None) -> None:
    """Run the translation CLI."""
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
