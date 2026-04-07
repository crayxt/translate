#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import TextIO

from core.formats import FileKind, detect_file_kind
from core.providers import (
    DEFAULT_PROVIDER as DEFAULT_PROVIDER_SPEC,
    DEFAULT_PROVIDER_NAME,
    SUPPORTED_TRANSLATION_PROVIDERS,
    get_translation_provider,
)
from core.resources import detect_default_text_resource
from tasks import check_translations as check_task
from tasks import extract_terms as extract_task
from tasks import extract_terms_local as extract_local_task
from tasks import revise_translations as revise_task
from tasks import translate as translate_task


DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "kk"
DEFAULT_PROVIDER = DEFAULT_PROVIDER_NAME
SUPPORTED_PROVIDER_CHOICES = tuple(sorted(SUPPORTED_TRANSLATION_PROVIDERS))
DEFAULT_MODEL = DEFAULT_PROVIDER_SPEC.default_model
THINKING_LEVEL_CHOICES = ("", "minimal", "low", "medium", "high")
GEMINI_BACKEND_CHOICES = ("studio", "vertex")
EXTRACT_MODE_CHOICES = ("missing", "all")
EXTRACT_OUTPUT_CHOICES = ("po", "json")
LOCAL_EXTRACT_MAX_LENGTH_CHOICES = ("1", "2", "3")
PROGRESS_PERCENT_RE = re.compile(r"Progress:\s*(?P<pct>\d+(?:\.\d+)?)%")
BATCH_PROGRESS_RE = re.compile(
    r"Progress:\s*completed batches\s+(?P<done>\d+)/(?P<total>\d+)"
)
TRANSLATABLE_FILETYPES = [
    ("Translatable files", "*.po *.ts *.resx *.strings *.txt *.xml"),
    ("PO files", "*.po"),
    ("Qt TS files", "*.ts"),
    ("RESX files", "*.resx"),
    ("Apple strings files", "*.strings"),
    ("Plain text files", "*.txt"),
    ("Android XML files", "*.xml"),
    ("All files", "*.*"),
]
VOCAB_FILETYPES = [
    ("Vocabulary files", "*.txt *.po"),
    ("Text files", "*.txt"),
    ("PO files", "*.po"),
    ("All files", "*.*"),
]
RULES_FILETYPES = [
    ("Markdown files", "*.md"),
    ("Text files", "*.txt"),
    ("All files", "*.*"),
]
CHECK_FILETYPES = [
    ("Checkable files", "*.po *.ts"),
    ("PO files", "*.po"),
    ("Qt TS files", "*.ts"),
    ("All files", "*.*"),
]
LOCAL_EXTRACT_FILETYPES = [
    ("Supported local extract files", "*.po *.ts *.resx *.strings *.txt *.xml *.json"),
    ("Translatable files", "*.po *.ts *.resx *.strings *.txt *.xml"),
    ("Android XML files", "*.xml"),
    ("JSON files", "*.json"),
    ("All files", "*.*"),
]
LOCAL_EXTRACT_SOURCE_FILETYPES = [
    ("Supported local extract files", "*.po *.ts *.resx *.strings *.txt *.xml"),
    ("Translatable files", "*.po *.ts *.resx *.strings *.txt *.xml"),
    ("Android XML files", "*.xml"),
    ("All files", "*.*"),
]
JSON_FILETYPES = [
    ("JSON files", "*.json"),
    ("All files", "*.*"),
]
LOG_DIR_NAME = "logs"
CLIPBOARD_WIDGET_CLASSES = frozenset({"Entry", "TEntry", "Text", "Combobox", "TCombobox"})
READONLY_STATES = frozenset({"disabled", "readonly"})


@dataclass(slots=True)
class ProcessGuiConfig:
    input_file: str = ""
    input_files: tuple[str, ...] = ()
    source_file: str = ""
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    provider: str = DEFAULT_PROVIDER
    gemini_backend: str = "studio"
    google_cloud_location: str = "global"
    model: str = DEFAULT_MODEL
    thinking_level: str = ""
    batch_size: str = ""
    parallel_requests: str = ""
    vocab_path: str = ""
    rules_path: str = ""
    rules_str: str = ""
    api_key: str = ""
    flex_mode: bool = False
    retranslate_all: bool = False
    warnings_report: bool = False


@dataclass(slots=True)
class ExtractGuiConfig:
    input_file: str = ""
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    provider: str = DEFAULT_PROVIDER
    gemini_backend: str = "studio"
    google_cloud_location: str = "global"
    model: str = DEFAULT_MODEL
    thinking_level: str = ""
    batch_size: str = ""
    parallel_requests: str = ""
    vocab_path: str = ""
    api_key: str = ""
    flex_mode: bool = False
    mode: str = "missing"
    out_format: str = "po"
    out_path: str = ""
    max_terms_per_batch: str = "80"
    max_attempts: str = "5"


@dataclass(slots=True)
class CheckGuiConfig:
    input_file: str = ""
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    provider: str = DEFAULT_PROVIDER
    gemini_backend: str = "studio"
    google_cloud_location: str = "global"
    model: str = DEFAULT_MODEL
    thinking_level: str = ""
    batch_size: str = ""
    parallel_requests: str = ""
    vocab_path: str = ""
    rules_path: str = ""
    rules_str: str = ""
    api_key: str = ""
    flex_mode: bool = False
    num_messages: str = ""
    out_path: str = ""
    include_ok: bool = False
    max_attempts: str = "5"


@dataclass(slots=True)
class LocalExtractGuiConfig:
    input_file: str = ""
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    vocab_path: str = ""
    mode: str = "missing"
    max_length: str = "1"
    out_path: str = ""
    include_rejected: bool = False
    to_po: bool = False
    also_po: bool = False
    include_borderline: bool = False


@dataclass(slots=True)
class ReviseGuiConfig:
    input_file: str = ""
    source_file: str = ""
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    provider: str = DEFAULT_PROVIDER
    gemini_backend: str = "studio"
    google_cloud_location: str = "global"
    model: str = DEFAULT_MODEL
    thinking_level: str = ""
    batch_size: str = ""
    parallel_requests: str = ""
    vocab_path: str = ""
    rules_path: str = ""
    rules_str: str = ""
    api_key: str = ""
    flex_mode: bool = False
    instruction: str = ""
    num_messages: str = ""
    out_path: str = ""
    max_attempts: str = "5"
    in_place: bool = False
    dry_run: bool = False


def _clean(value: str) -> str:
    return str(value or "").strip()


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def widget_supports_clipboard(widget_class: str) -> bool:
    return str(widget_class or "") in CLIPBOARD_WIDGET_CLASSES


def widget_is_editable(widget_class: str, state: str) -> bool:
    return widget_supports_clipboard(widget_class) and str(state or "") not in READONLY_STATES


def path_exists_as_file_or_dir(path: str) -> bool:
    cleaned_path = _clean(path)
    return bool(cleaned_path) and (os.path.isfile(cleaned_path) or os.path.isdir(cleaned_path))


def _validate_optional_positive_int(value: str, flag_name: str) -> str | None:
    cleaned = _clean(value)
    if not cleaned:
        return None

    try:
        parsed = int(cleaned)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be a whole number.") from exc

    if parsed <= 0:
        raise ValueError(f"{flag_name} must be greater than 0.")
    return str(parsed)


def _validate_choice(value: str, choices: tuple[str, ...], flag_name: str) -> None:
    cleaned = _clean(value)
    if cleaned and cleaned not in choices:
        raise ValueError(
            f"{flag_name} must be one of: {', '.join(choice for choice in choices if choice)}."
        )


def build_resource_root(base_dir: str | None = None) -> str:
    return os.path.abspath(base_dir or os.path.dirname(__file__))


def build_script_path(script_name: str, base_dir: str | None = None) -> str:
    return os.path.join(build_resource_root(base_dir), script_name)


def build_cli_script_path(base_dir: str | None = None) -> str:
    return build_script_path("translate_cli.py", base_dir=base_dir)


def _sanitize_log_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def build_log_dir(base_dir: str | None = None) -> str:
    return os.path.join(build_resource_root(base_dir), LOG_DIR_NAME)


def build_run_log_path(
    tool_key: str,
    input_file: str,
    base_dir: str | None = None,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    input_stem = os.path.splitext(os.path.basename(_clean(input_file)))[0]
    filename = f"{_sanitize_log_name(tool_key)}-{_sanitize_log_name(input_stem)}-{timestamp}.log"
    return os.path.join(build_log_dir(base_dir), filename)


def detect_default_resource_path(
    prefix: str,
    extension: str,
    target_lang: str,
    base_dir: str | None = None,
) -> str:
    return (
        detect_default_text_resource(
            prefix,
            extension,
            target_lang,
            base_dir=build_resource_root(base_dir),
            allow_directory=prefix == "vocab",
        )
        or ""
    )


def detect_default_resource_paths(
    target_lang: str,
    base_dir: str | None = None,
) -> tuple[str, str]:
    return (
        detect_default_resource_path("vocab", "txt", target_lang, base_dir=base_dir),
        detect_default_resource_path("rules", "md", target_lang, base_dir=base_dir),
    )


def build_system_prompt_preview(tool_key: str, target_lang: str) -> str:
    normalized_tool = _clean(tool_key).lower()
    resolved_target_lang = _clean(target_lang) or DEFAULT_TARGET_LANG

    if normalized_tool in {"process", "translate"}:
        return translate_task.SYSTEM_INSTRUCTION.strip()
    if normalized_tool == "extract":
        return extract_task.build_term_system_instruction(resolved_target_lang)
    if normalized_tool == "extract_local":
        return (
            "No model system prompt is used for this task.\n\n"
            "Local term discovery runs deterministic source-side extraction using:\n"
            "- core/term_extraction.py for normalization, tokenization, filtering, evidence collection, and scoring\n"
            "- core/term_handoff.py for JSON report shaping and JSON-to-PO conversion\n"
            "- data/extract/... resource files for stopwords, low-value words, allowlists, and excluded terms\n"
            "- the approved vocabulary to filter already-known terms in missing mode\n\n"
            "The task can scan one source file or a whole source directory tree.\n"
            "It can also convert a local extraction JSON report into a translation-ready PO glossary handoff."
        )
    if normalized_tool == "check":
        return check_task.build_check_system_instruction(resolved_target_lang)
    if normalized_tool == "revise":
        return revise_task.build_revision_system_instruction(resolved_target_lang)
    return "System prompt preview unavailable."


def choose_resource_field_value(
    current_value: str,
    previous_auto_value: str,
    new_auto_value: str,
    force: bool = False,
) -> tuple[str, str]:
    cleaned_current = _clean(current_value)
    cleaned_previous_auto = _clean(previous_auto_value)
    cleaned_new_auto = _clean(new_auto_value)

    if force or not cleaned_current or cleaned_current == cleaned_previous_auto:
        return cleaned_new_auto, cleaned_new_auto

    return cleaned_current, cleaned_previous_auto


def read_text_file_or_empty(path: str) -> str:
    cleaned_path = _clean(path)
    if not cleaned_path or not os.path.isfile(cleaned_path):
        return ""

    with open(cleaned_path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_progress_percent(line: str) -> float | None:
    text = str(line or "").strip()
    if not text:
        return None

    percent_match = PROGRESS_PERCENT_RE.search(text)
    if percent_match:
        return _clamp_percent(float(percent_match.group("pct")))

    batch_match = BATCH_PROGRESS_RE.search(text)
    if batch_match:
        total = int(batch_match.group("total"))
        if total <= 0:
            return None
        done = int(batch_match.group("done"))
        return _clamp_percent((done / total) * 100.0)

    if text.endswith(" complete."):
        return 100.0

    return None


def summarize_input_files(file_paths: list[str] | tuple[str, ...]) -> str:
    cleaned_paths = [_clean(path) for path in file_paths if _clean(path)]
    if not cleaned_paths:
        return ""
    if len(cleaned_paths) == 1:
        return cleaned_paths[0]
    return f"{cleaned_paths[0]} (+{len(cleaned_paths) - 1} more)"


def get_local_extract_file_dialog_config(to_po_mode: bool) -> tuple[str, list[tuple[str, str]]]:
    if to_po_mode:
        return "Select local extraction JSON file", JSON_FILETYPES
    return "Select source file for local extraction", LOCAL_EXTRACT_SOURCE_FILETYPES


def resolve_process_input_files(config: ProcessGuiConfig) -> list[str]:
    explicit_files = [_clean(path) for path in config.input_files if _clean(path)]
    if explicit_files:
        return explicit_files
    cleaned_input = _clean(config.input_file)
    return [cleaned_input] if cleaned_input else []


def _validate_base_config(
    *,
    input_file: str,
    input_files: list[str] | None = None,
    source_lang: str,
    target_lang: str,
    provider: str,
    gemini_backend: str,
    google_cloud_location: str,
    model: str,
    thinking_level: str,
    batch_size: str,
    parallel_requests: str,
    vocab_path: str,
    api_key: str,
    environ: dict[str, str] | None = None,
    rules_path: str = "",
) -> list[str]:
    env = environ if environ is not None else os.environ
    errors: list[str] = []

    cleaned_input = _clean(input_file)
    cleaned_input_files = [_clean(path) for path in (input_files or []) if _clean(path)]
    cleaned_source = _clean(source_lang)
    cleaned_target = _clean(target_lang)
    cleaned_provider = _clean(provider)
    cleaned_gemini_backend = _clean(gemini_backend).lower()
    cleaned_google_cloud_location = _clean(google_cloud_location)
    cleaned_model = _clean(model)
    cleaned_vocab = _clean(vocab_path)
    cleaned_rules = _clean(rules_path)
    cleaned_api_key = _clean(api_key)

    if cleaned_input_files:
        for file_path in cleaned_input_files:
            if not os.path.isfile(file_path):
                errors.append(f"Input file does not exist: {file_path}")
    elif not cleaned_input:
        errors.append("Input file is required.")
    elif not os.path.isfile(cleaned_input):
        errors.append(f"Input file does not exist: {cleaned_input}")

    if not cleaned_source:
        errors.append("Source language is required.")

    if not cleaned_target:
        errors.append("Target language is required.")

    if not cleaned_provider:
        errors.append("Provider is required.")
    else:
        try:
            get_translation_provider(cleaned_provider)
        except ValueError as exc:
            errors.append(str(exc))

    if cleaned_provider == "gemini":
        try:
            _validate_choice(cleaned_gemini_backend, GEMINI_BACKEND_CHOICES, "Gemini backend")
        except ValueError as exc:
            errors.append(str(exc))
        if cleaned_gemini_backend == "vertex" and (
            cleaned_google_cloud_location and cleaned_google_cloud_location.lower() != "global"
        ):
            errors.append(
                "Gemini Vertex API-key mode currently supports only the global endpoint."
            )
        if cleaned_gemini_backend != "vertex" and (
            cleaned_google_cloud_location and cleaned_google_cloud_location.lower() != "global"
        ):
            errors.append("Set Gemini backend to 'vertex' to use a custom Google Cloud location.")

    if not cleaned_model:
        errors.append("Model is required.")

    try:
        _validate_choice(thinking_level, THINKING_LEVEL_CHOICES, "Thinking level")
    except ValueError as exc:
        errors.append(str(exc))

    for label, value in (
        ("Batch size", batch_size),
        ("Parallel requests", parallel_requests),
    ):
        try:
            _validate_optional_positive_int(value, label)
        except ValueError as exc:
            errors.append(str(exc))

    if cleaned_vocab and not path_exists_as_file_or_dir(cleaned_vocab):
        errors.append(f"Vocabulary file or directory does not exist: {cleaned_vocab}")

    if cleaned_rules and not os.path.isfile(cleaned_rules):
        errors.append(f"Rules file does not exist: {cleaned_rules}")

    if cleaned_provider:
        try:
            provider_spec = get_translation_provider(cleaned_provider)
        except ValueError:
            provider_spec = None
        if provider_spec is not None:
            api_key_env = provider_spec.api_key_env
            if cleaned_provider == "gemini" and cleaned_gemini_backend == "vertex":
                if api_key_env and not cleaned_api_key and not env.get(api_key_env):
                    errors.append(
                        f"{api_key_env} is not set. Provide an API key in the GUI or the environment."
                    )
            elif api_key_env and not cleaned_api_key and not env.get(api_key_env):
                errors.append(
                    f"{api_key_env} is not set. Provide an API key in the GUI or the environment."
                )

    return errors


def validate_process_gui_config(
    config: ProcessGuiConfig,
    environ: dict[str, str] | None = None,
) -> list[str]:
    input_files = resolve_process_input_files(config)
    errors = _validate_base_config(
        input_file=input_files[0] if input_files else config.input_file,
        input_files=input_files,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
        rules_path=config.rules_path,
        api_key=config.api_key,
        environ=environ,
    )

    cleaned_source_file = _clean(config.source_file)
    if translation_requires_source_file(input_files):
        if not cleaned_source_file:
            errors.append("Source file is required for Android .xml translation runs.")
        elif not os.path.isfile(cleaned_source_file):
            errors.append(f"Source file does not exist: {cleaned_source_file}")
    elif cleaned_source_file and not os.path.isfile(cleaned_source_file):
        errors.append(f"Source file does not exist: {cleaned_source_file}")

    if not any(message.startswith("Source file ") for message in errors):
        try:
            translate_task.validate_translation_files(
                input_files,
                source_file=cleaned_source_file or None,
            )
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def validate_extract_gui_config(
    config: ExtractGuiConfig,
    environ: dict[str, str] | None = None,
) -> list[str]:
    errors = _validate_base_config(
        input_file=config.input_file,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
        api_key=config.api_key,
        environ=environ,
    )

    try:
        _validate_choice(config.mode, EXTRACT_MODE_CHOICES, "Mode")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        _validate_choice(config.out_format, EXTRACT_OUTPUT_CHOICES, "Output format")
    except ValueError as exc:
        errors.append(str(exc))

    for label, value in (
        ("Max terms per batch", config.max_terms_per_batch),
        ("Max attempts", config.max_attempts),
    ):
        try:
            _validate_optional_positive_int(value, label)
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def validate_local_extract_gui_config(config: LocalExtractGuiConfig) -> list[str]:
    errors: list[str] = []

    cleaned_input = _clean(config.input_file)
    cleaned_vocab = _clean(config.vocab_path)
    cleaned_out = _clean(config.out_path)
    cleaned_source = _clean(config.source_lang)
    cleaned_target = _clean(config.target_lang)
    cleaned_mode = _clean(config.mode)
    cleaned_max_length = _clean(config.max_length)

    if not cleaned_input:
        errors.append("Input file is required.")
    elif config.to_po:
        if not os.path.isfile(cleaned_input):
            errors.append(f"Input file does not exist: {cleaned_input}")
    elif not os.path.isfile(cleaned_input) and not os.path.isdir(cleaned_input):
        errors.append(f"Input file or directory does not exist: {cleaned_input}")

    if not config.to_po:
        if not cleaned_source:
            errors.append("Source language is required.")
        if not cleaned_target:
            errors.append("Target language is required.")
        try:
            _validate_choice(cleaned_mode, EXTRACT_MODE_CHOICES, "Mode")
        except ValueError as exc:
            errors.append(str(exc))
        try:
            max_length_value = int(cleaned_max_length)
        except ValueError:
            errors.append("Max length must be 1, 2, or 3.")
        else:
            if max_length_value not in (1, 2, 3):
                errors.append("Max length must be 1, 2, or 3.")
        if cleaned_vocab and not path_exists_as_file_or_dir(cleaned_vocab):
            errors.append(f"Vocabulary file or directory does not exist: {cleaned_vocab}")
        if cleaned_out and not cleaned_out.lower().endswith(".json"):
            errors.append("Local extraction output path should end with .json.")
    else:
        if config.also_po:
            errors.append("JSON to PO mode cannot also request one-shot PO output.")
        if cleaned_input and not cleaned_input.lower().endswith(".json"):
            errors.append("JSON to PO mode requires a .json input file.")
        if cleaned_out and not cleaned_out.lower().endswith(".po"):
            errors.append("PO output path should end with .po in JSON to PO mode.")

    return errors


def validate_check_gui_config(
    config: CheckGuiConfig,
    environ: dict[str, str] | None = None,
) -> list[str]:
    errors = _validate_base_config(
        input_file=config.input_file,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
        rules_path=config.rules_path,
        api_key=config.api_key,
        environ=environ,
    )

    for label, value in (
        ("Probe / num messages", config.num_messages),
        ("Max attempts", config.max_attempts),
    ):
        try:
            _validate_optional_positive_int(value, label)
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def _detect_revision_file_kind(input_file: str) -> FileKind | None:
    cleaned_input = _clean(input_file)
    if not cleaned_input:
        return None

    try:
        return detect_file_kind(cleaned_input)
    except ValueError:
        return None


def translation_requires_source_file(input_files: list[str] | tuple[str, ...]) -> bool:
    if len(input_files) != 1:
        return False
    return _detect_revision_file_kind(input_files[0]) == FileKind.ANDROID_XML


def revision_requires_source_file(input_file: str) -> bool:
    file_kind = _detect_revision_file_kind(input_file)
    return file_kind in (FileKind.ANDROID_XML, FileKind.STRINGS, FileKind.RESX, FileKind.TXT)


def validate_revise_gui_config(
    config: ReviseGuiConfig,
    environ: dict[str, str] | None = None,
) -> list[str]:
    errors = _validate_base_config(
        input_file=config.input_file,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
        rules_path=config.rules_path,
        api_key=config.api_key,
        environ=environ,
    )

    file_kind = _detect_revision_file_kind(config.input_file)
    if _clean(config.input_file) and file_kind is None:
        errors.append("Input file must be a supported .po, .ts, .resx, .strings, .txt, or Android .xml file.")

    cleaned_source_file = _clean(config.source_file)
    if revision_requires_source_file(config.input_file):
        if not cleaned_source_file:
            errors.append("Source file is required for Android .xml, .strings, .resx, and .txt revision runs.")
        elif not os.path.isfile(cleaned_source_file):
            errors.append(f"Source file does not exist: {cleaned_source_file}")
    elif cleaned_source_file and not os.path.isfile(cleaned_source_file):
        errors.append(f"Source file does not exist: {cleaned_source_file}")

    cleaned_instruction = _clean(config.instruction)
    if not cleaned_instruction:
        errors.append("Instruction is required.")

    for label, value in (
        ("Probe / num messages", config.num_messages),
        ("Max attempts", config.max_attempts),
    ):
        try:
            _validate_optional_positive_int(value, label)
        except ValueError as exc:
            errors.append(str(exc))

    if _clean(config.out_path) and config.in_place:
        errors.append("Output path and in-place mode cannot be used together.")

    return errors


def _append_common_cli_args(
    command: list[str],
    *,
    source_lang: str,
    target_lang: str,
    provider: str,
    gemini_backend: str,
    google_cloud_location: str,
    model: str,
    thinking_level: str,
    flex_mode: bool,
    batch_size: str,
    parallel_requests: str,
    vocab_path: str = "",
) -> None:
    provider_name = _clean(provider) or DEFAULT_PROVIDER
    command.extend(
        [
            "--source-lang",
            _clean(source_lang),
            "--target-lang",
            _clean(target_lang),
            "--provider",
            provider_name,
            "--model",
            _clean(model),
        ]
    )

    thinking_level_value = _clean(thinking_level)
    if thinking_level_value:
        command.extend(["--thinking-level", thinking_level_value])

    try:
        provider_spec = get_translation_provider(provider_name)
    except ValueError:
        provider_spec = None
    if provider_name == "gemini":
        backend_value = _clean(gemini_backend).lower()
        location_value = _clean(google_cloud_location)
        if backend_value == "vertex" or (location_value and location_value.lower() != "global"):
            command.extend(["--gemini-backend", "vertex"])
            if location_value:
                command.extend(["--google-cloud-location", location_value])
    if flex_mode and provider_spec is not None and getattr(provider_spec, "supports_flex_mode", False):
        command.append("--flex")

    batch_size_value = _validate_optional_positive_int(batch_size, "Batch size")
    if batch_size_value:
        command.extend(["--batch-size", batch_size_value])

    parallel_value = _validate_optional_positive_int(
        parallel_requests,
        "Parallel requests",
    )
    if parallel_value:
        command.extend(["--parallel-requests", parallel_value])

    vocab_value = _clean(vocab_path)
    if vocab_value:
        command.extend(["--vocab", vocab_value])


def build_process_command(
    config: ProcessGuiConfig,
    python_executable: str | None = None,
    script_path: str | None = None,
) -> list[str]:
    errors = validate_process_gui_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    resolved_script = os.path.abspath(script_path or build_cli_script_path())
    if not os.path.isfile(resolved_script):
        raise ValueError(f"translate_cli.py not found at: {resolved_script}")

    input_files = resolve_process_input_files(config)
    command = [
        python_executable or sys.executable,
        "-u",
        resolved_script,
        "translate",
        *input_files,
    ]
    _append_common_cli_args(
        command,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        flex_mode=config.flex_mode,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
    )

    rules_path = _clean(config.rules_path)
    if rules_path:
        command.extend(["--rules", rules_path])

    rules_str = _clean(config.rules_str)
    if rules_str:
        command.extend(["--rules-str", rules_str])

    source_file = _clean(config.source_file)
    if source_file:
        command.extend(["--source-file", source_file])

    if config.retranslate_all:
        command.append("--retranslate-all")
    if config.warnings_report:
        command.append("--warnings-report")

    return command


def build_extract_command(
    config: ExtractGuiConfig,
    python_executable: str | None = None,
    script_path: str | None = None,
) -> list[str]:
    errors = validate_extract_gui_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    resolved_script = os.path.abspath(script_path or build_cli_script_path())
    if not os.path.isfile(resolved_script):
        raise ValueError(f"translate_cli.py not found at: {resolved_script}")

    command = [
        python_executable or sys.executable,
        "-u",
        resolved_script,
        "extract-terms",
        _clean(config.input_file),
    ]
    _append_common_cli_args(
        command,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        flex_mode=config.flex_mode,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
    )
    command.extend(["--mode", _clean(config.mode) or "missing"])
    command.extend(["--out-format", _clean(config.out_format) or "po"])

    out_path = _clean(config.out_path)
    if out_path:
        command.extend(["--out", out_path])

    command.extend(
        [
            "--max-terms-per-batch",
            _validate_optional_positive_int(
                config.max_terms_per_batch,
                "Max terms per batch",
            )
            or "80",
            "--max-attempts",
            _validate_optional_positive_int(config.max_attempts, "Max attempts")
            or "5",
        ]
    )
    return command


def build_local_extract_command(
    config: LocalExtractGuiConfig,
    python_executable: str | None = None,
    script_path: str | None = None,
) -> list[str]:
    errors = validate_local_extract_gui_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    resolved_script = os.path.abspath(script_path or build_cli_script_path())
    if not os.path.isfile(resolved_script):
        raise ValueError(f"translate_cli.py not found at: {resolved_script}")

    command = [
        python_executable or sys.executable,
        "-u",
        resolved_script,
        "extract-terms-local",
        _clean(config.input_file),
    ]

    if config.to_po:
        command.append("--to-po")
        if config.include_borderline:
            command.append("--include-borderline")
    else:
        command.extend(
            [
                "--source-lang",
                _clean(config.source_lang),
                "--target-lang",
                _clean(config.target_lang),
                "--mode",
                _clean(config.mode) or "missing",
                "--max-length",
                _clean(config.max_length) or "1",
            ]
        )
        vocab_path = _clean(config.vocab_path)
        if vocab_path:
            command.extend(["--vocab", vocab_path])
        if config.include_rejected:
            command.append("--include-rejected")
        if config.also_po:
            command.append("--also-po")
            if config.include_borderline:
                command.append("--include-borderline")

    out_path = _clean(config.out_path)
    if out_path:
        command.extend(["--out", out_path])

    return command


def build_check_command(
    config: CheckGuiConfig,
    python_executable: str | None = None,
    script_path: str | None = None,
) -> list[str]:
    errors = validate_check_gui_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    resolved_script = os.path.abspath(script_path or build_cli_script_path())
    if not os.path.isfile(resolved_script):
        raise ValueError(f"translate_cli.py not found at: {resolved_script}")

    command = [
        python_executable or sys.executable,
        "-u",
        resolved_script,
        "check",
        _clean(config.input_file),
    ]
    _append_common_cli_args(
        command,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        flex_mode=config.flex_mode,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
    )

    rules_path = _clean(config.rules_path)
    if rules_path:
        command.extend(["--rules", rules_path])

    rules_str = _clean(config.rules_str)
    if rules_str:
        command.extend(["--rules-str", rules_str])

    num_messages = _validate_optional_positive_int(
        config.num_messages,
        "Probe / num messages",
    )
    if num_messages:
        command.extend(["--probe", num_messages])

    out_path = _clean(config.out_path)
    if out_path:
        command.extend(["--out", out_path])

    if config.include_ok:
        command.append("--include-ok")

    command.extend(
        [
            "--max-attempts",
            _validate_optional_positive_int(config.max_attempts, "Max attempts")
            or "5",
        ]
    )
    return command


def build_revise_command(
    config: ReviseGuiConfig,
    python_executable: str | None = None,
    script_path: str | None = None,
) -> list[str]:
    errors = validate_revise_gui_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    resolved_script = os.path.abspath(script_path or build_cli_script_path())
    if not os.path.isfile(resolved_script):
        raise ValueError(f"translate_cli.py not found at: {resolved_script}")

    command = [
        python_executable or sys.executable,
        "-u",
        resolved_script,
        "revise",
        _clean(config.input_file),
        "--instruction",
        _clean(config.instruction),
    ]
    _append_common_cli_args(
        command,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        model=config.model,
        thinking_level=config.thinking_level,
        flex_mode=config.flex_mode,
        batch_size=config.batch_size,
        parallel_requests=config.parallel_requests,
        vocab_path=config.vocab_path,
    )

    source_file = _clean(config.source_file)
    if source_file:
        command.extend(["--source-file", source_file])

    rules_path = _clean(config.rules_path)
    if rules_path:
        command.extend(["--rules", rules_path])

    rules_str = _clean(config.rules_str)
    if rules_str:
        command.extend(["--rules-str", rules_str])

    num_messages = _validate_optional_positive_int(
        config.num_messages,
        "Probe / num messages",
    )
    if num_messages:
        command.extend(["--probe", num_messages])

    out_path = _clean(config.out_path)
    if out_path:
        command.extend(["--out", out_path])

    if config.in_place:
        command.append("--in-place")

    if config.dry_run:
        command.append("--dry-run")

    command.extend(
        [
            "--max-attempts",
            _validate_optional_positive_int(config.max_attempts, "Max attempts")
            or "5",
        ]
    )
    return command


def build_script_env(
    api_key: str,
    provider: str = DEFAULT_PROVIDER,
    gemini_backend: str = "studio",
    google_cloud_location: str = "global",
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    cleaned_api_key = _clean(api_key)
    provider_spec = get_translation_provider(provider)
    api_key_env = provider_spec.api_key_env
    if cleaned_api_key and api_key_env:
        env[api_key_env] = cleaned_api_key
    if _clean(provider).lower() == "gemini":
        backend_value = _clean(gemini_backend).lower() or "studio"
        env["GOOGLE_GENAI_USE_VERTEXAI"] = "true" if backend_value == "vertex" else "false"
        if backend_value == "vertex":
            location_value = _clean(google_cloud_location)
            if location_value:
                env["GOOGLE_CLOUD_LOCATION"] = location_value
    return env


def build_process_env(
    config: ProcessGuiConfig,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    return build_script_env(
        config.api_key,
        provider=config.provider,
        gemini_backend=config.gemini_backend,
        google_cloud_location=config.google_cloud_location,
        base_env=base_env,
    )


class BaseToolTab(ttk.Frame):
    def __init__(
        self,
        app: "ProcessGuiApp",
        notebook: ttk.Notebook,
        *,
        tool_key: str,
        title: str,
        script_name: str,
        input_filetypes: list[tuple[str, str]],
        supports_vocab: bool = True,
        supports_rules: bool = False,
        input_label: str = "Input file",
        supports_provider_controls: bool = True,
        preview_label: str = "System prompt",
    ) -> None:
        super().__init__(notebook)
        self.app = app
        self.tool_key = tool_key
        self.title = title
        self.script_name = script_name
        self.input_filetypes = input_filetypes
        self.supports_vocab = supports_vocab
        self.supports_rules = supports_rules
        self.input_label = input_label
        self.supports_provider_controls = supports_provider_controls
        self.preview_label = preview_label
        self.resource_root = app.resource_root
        self._auto_vocab_path = ""
        self._auto_rules_path = ""
        self._auto_model = DEFAULT_MODEL
        self.api_key_var = app.api_key_var
        self.provider_var = app.provider_var
        self.gemini_backend_var = app.gemini_backend_var
        self.google_cloud_location_var = app.google_cloud_location_var
        self.thinking_level_var = app.thinking_level_var
        self.flex_mode_var = app.flex_mode_var

        self.input_file_var = tk.StringVar()
        self.source_lang_var = tk.StringVar(value=DEFAULT_SOURCE_LANG)
        self.target_lang_var = tk.StringVar(value=DEFAULT_TARGET_LANG)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.batch_size_var = tk.StringVar()
        self.parallel_requests_var = tk.StringVar()
        self.vocab_path_var = tk.StringVar()
        self.rules_path_var = tk.StringVar()
        self.api_status_var = tk.StringVar()
        self.log_text: ScrolledText | None = None
        self.run_button: ttk.Button | None = None
        self.stop_button: ttk.Button | None = None
        self.rules_preview_text: ScrolledText | None = None
        self.rules_text: ScrolledText | None = None
        self.system_prompt_text: ScrolledText | None = None
        self.flex_mode_button: ttk.Checkbutton | None = None
        self.thinking_level_combo: ttk.Combobox | None = None
        self.gemini_backend_combo: ttk.Combobox | None = None
        self.google_cloud_location_entry: ttk.Entry | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_widgets()
        self._apply_default_model(force=True)
        self._refresh_provider_specific_controls()
        self._refresh_api_status()
        self._apply_default_resources(force=True)
        self._load_rules_preview()
        self._load_system_prompt_preview()

        self.api_key_var.trace_add("write", self._on_api_key_changed)
        self.provider_var.trace_add("write", self._on_provider_changed)
        self.gemini_backend_var.trace_add("write", self._on_api_key_changed)
        self.target_lang_var.trace_add("write", self._on_target_lang_changed)
        self.rules_path_var.trace_add("write", self._on_rules_path_changed)

    def _build_widgets(self) -> None:
        paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            opaqueresize=True,
        )
        paned.grid(row=0, column=0, sticky="nsew")

        form = ttk.Frame(paned, padding=12)
        form.columnconfigure(1, weight=1)
        form.rowconfigure(100, weight=1)

        ttk.Label(form, text=self.title).grid(row=0, column=0, columnspan=3, sticky="w")

        row = 1
        row += self._build_input_row(form, row)
        self._add_entry_row(form, row=row, label="Source lang", variable=self.source_lang_var)
        row += 1
        self._add_entry_row(form, row=row, label="Target lang", variable=self.target_lang_var)
        row += 1
        if self.supports_provider_controls:
            self._add_combo_row(
                form,
                row=row,
                label="Provider",
                variable=self.provider_var,
                values=SUPPORTED_PROVIDER_CHOICES,
            )
            row += 1
            ttk.Label(form, text="Gemini backend").grid(row=row, column=0, sticky="w", pady=4)
            self.gemini_backend_combo = ttk.Combobox(
                form,
                textvariable=self.gemini_backend_var,
                values=GEMINI_BACKEND_CHOICES,
            )
            self.gemini_backend_combo.grid(row=row, column=1, sticky="ew", pady=4)
            row += 1
            ttk.Label(form, text="GCP location").grid(row=row, column=0, sticky="w", pady=4)
            self.google_cloud_location_entry = ttk.Entry(
                form,
                textvariable=self.google_cloud_location_var,
            )
            self.google_cloud_location_entry.grid(row=row, column=1, sticky="ew", pady=4)
            row += 1
            self._add_entry_row(form, row=row, label="Model", variable=self.model_var)
            row += 1
            ttk.Label(form, text="Thinking level").grid(row=row, column=0, sticky="w", pady=4)
            self.thinking_level_combo = ttk.Combobox(
                form,
                textvariable=self.thinking_level_var,
                values=THINKING_LEVEL_CHOICES,
            )
            self.thinking_level_combo.grid(row=row, column=1, sticky="ew", pady=4)
            row += 1
            self.flex_mode_button = ttk.Checkbutton(
                form,
                text="Flex mode",
                variable=self.flex_mode_var,
            )
            self.flex_mode_button.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
            row += 1
            self._add_entry_row(form, row=row, label="Batch size", variable=self.batch_size_var)
            row += 1
            self._add_entry_row(
                form,
                row=row,
                label="Parallel requests",
                variable=self.parallel_requests_var,
            )
            row += 1

        if self.supports_vocab:
            self._add_entry_row(
                form,
                row=row,
                label="Vocabulary",
                variable=self.vocab_path_var,
                browse_command=self._browse_vocab_file,
            )
            row += 1

        if self.supports_rules:
            self._add_entry_row(
                form,
                row=row,
                label="Rules file",
                variable=self.rules_path_var,
                browse_command=self._browse_rules_file,
            )
            row += 1

        if self.supports_provider_controls:
            self._add_entry_row(
                form,
                row=row,
                label="API key",
                variable=self.api_key_var,
                show="*",
            )
            row += 1

            ttk.Label(form, textvariable=self.api_status_var).grid(
                row=row,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(2, 8),
            )
            row += 1

        row = self._build_tool_specific_fields(form, row)

        ttk.Label(form, text="Instructions").grid(row=row, column=0, sticky="nw")
        instructions_notebook = ttk.Notebook(form)
        instructions_notebook.grid(
            row=row + 1,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(4, 8),
        )

        system_tab = ttk.Frame(instructions_notebook, padding=6)
        system_tab.columnconfigure(0, weight=1)
        system_tab.rowconfigure(0, weight=1)
        self.system_prompt_text = ScrolledText(
            system_tab,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.system_prompt_text.grid(row=0, column=0, sticky="nsew")
        instructions_notebook.add(system_tab, text=self.preview_label)

        if self.supports_rules:
            preview_tab = ttk.Frame(instructions_notebook, padding=6)
            preview_tab.columnconfigure(0, weight=1)
            preview_tab.rowconfigure(0, weight=1)
            self.rules_preview_text = ScrolledText(
                preview_tab,
                wrap="word",
                font=("Consolas", 10),
                state="disabled",
            )
            self.rules_preview_text.grid(row=0, column=0, sticky="nsew")
            instructions_notebook.add(preview_tab, text="Loaded rules")

            inline_tab = ttk.Frame(instructions_notebook, padding=6)
            inline_tab.columnconfigure(0, weight=1)
            inline_tab.rowconfigure(0, weight=1)
            self.rules_text = ScrolledText(
                inline_tab,
                wrap="word",
                font=("Consolas", 10),
            )
            self.rules_text.grid(row=0, column=0, sticky="nsew")
            instructions_notebook.add(inline_tab, text="Inline override")

        form.rowconfigure(row + 1, weight=1)
        row += 2

        buttons = ttk.Frame(form)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(buttons, text="Run", command=self._start_run)
        self.run_button.grid(row=0, column=1, sticky="e")

        self.stop_button = ttk.Button(
            buttons,
            text="Stop",
            command=self._stop_run,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        ttk.Button(buttons, text="Clear log", command=self.clear_log).grid(
            row=0,
            column=3,
            sticky="e",
            padx=(8, 0),
        )

        log_frame = ttk.LabelFrame(paned, text="Output", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(
            log_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        paned.add(form, minsize=430, stretch="always")
        paned.add(log_frame, minsize=300, stretch="always")

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        return start_row

    def _build_input_row(self, parent: ttk.Frame, row: int) -> int:
        self._add_entry_row(
            parent,
            row=row,
            label=self.input_label,
            variable=self.input_file_var,
            browse_command=self._browse_input_file,
        )
        return 1

    def _add_entry_row(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command: object | None = None,
        show: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, show=show or "").grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )
        if browse_command is not None:
            ttk.Button(parent, text="Browse", command=browse_command).grid(
                row=row,
                column=2,
                sticky="w",
                padx=(8, 0),
                pady=4,
            )

    def _add_combo_row(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=values,
        )
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        return combo

    def _browse_input_file(self) -> None:
        selected = filedialog.askopenfilename(
            title=f"Select input file for {self.title}",
            filetypes=self.input_filetypes,
        )
        if selected:
            self.input_file_var.set(selected)

    def _browse_vocab_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select vocabulary file",
            filetypes=VOCAB_FILETYPES,
        )
        if selected:
            self.vocab_path_var.set(selected)

    def _browse_rules_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select rules file",
            filetypes=RULES_FILETYPES,
        )
        if selected:
            self.rules_path_var.set(selected)

    def _on_api_key_changed(self, *_args: object) -> None:
        self._refresh_api_status()

    def _on_provider_changed(self, *_args: object) -> None:
        self._apply_default_model()
        self._refresh_provider_specific_controls()
        self._refresh_api_status()

    def _on_target_lang_changed(self, *_args: object) -> None:
        self._apply_default_resources()
        self._load_system_prompt_preview()

    def _on_rules_path_changed(self, *_args: object) -> None:
        self._load_rules_preview()

    def _refresh_api_status(self) -> None:
        if not self.supports_provider_controls:
            self.api_status_var.set("")
            return
        provider_name = _clean(self.provider_var.get()) or DEFAULT_PROVIDER
        try:
            provider_spec = get_translation_provider(provider_name)
        except ValueError:
            self.api_status_var.set(f"API key source: invalid provider ({provider_name})")
            return

        api_key_env = provider_spec.api_key_env
        if not api_key_env:
            self.api_status_var.set("API key source: not required")
            return

        if provider_name == "gemini" and _clean(self.gemini_backend_var.get()).lower() == "vertex":
            if _clean(self.api_key_var.get()):
                self.api_status_var.set(f"API key source: GUI field ({api_key_env})")
                return

            if os.environ.get(api_key_env):
                self.api_status_var.set(f"API key source: {api_key_env} environment variable")
                return

            self.api_status_var.set(f"API key source: missing ({api_key_env})")
            return

        if _clean(self.api_key_var.get()):
            self.api_status_var.set(f"API key source: GUI field ({api_key_env})")
            return

        if os.environ.get(api_key_env):
            self.api_status_var.set(f"API key source: {api_key_env} environment variable")
            return

        self.api_status_var.set(f"API key source: missing ({api_key_env})")

    def _refresh_provider_specific_controls(self) -> None:
        if not self.supports_provider_controls:
            return
        if (
            self.flex_mode_button is None
            and self.thinking_level_combo is None
            and self.gemini_backend_combo is None
        ):
            return

        provider_name = _clean(self.provider_var.get()) or DEFAULT_PROVIDER
        try:
            provider_spec = get_translation_provider(provider_name)
        except ValueError:
            flex_state = "disabled"
            thinking_state = "disabled"
            gemini_state = "disabled"
        else:
            flex_state = "normal" if getattr(provider_spec, "supports_flex_mode", False) else "disabled"
            thinking_state = "normal" if getattr(provider_spec, "supports_thinking", False) else "disabled"
            gemini_state = "normal" if provider_name == "gemini" else "disabled"
        if self.flex_mode_button is not None:
            self.flex_mode_button.configure(state=flex_state)
        if flex_state == "disabled":
            self.flex_mode_var.set(False)
        if self.thinking_level_combo is not None:
            self.thinking_level_combo.configure(state=thinking_state)
        if thinking_state == "disabled":
            self.thinking_level_var.set("")
        for widget in (
            self.gemini_backend_combo,
            self.google_cloud_location_entry,
        ):
            if widget is not None:
                widget.configure(state=gemini_state)

    def _apply_default_model(self, force: bool = False) -> None:
        if not self.supports_provider_controls:
            return
        provider_name = _clean(self.provider_var.get()) or DEFAULT_PROVIDER
        try:
            provider_spec = get_translation_provider(provider_name)
        except ValueError:
            new_default_model = DEFAULT_MODEL
        else:
            new_default_model = _clean(provider_spec.default_model) or DEFAULT_MODEL

        model_value, self._auto_model = choose_resource_field_value(
            current_value=self.model_var.get(),
            previous_auto_value=self._auto_model,
            new_auto_value=new_default_model,
            force=force,
        )
        if model_value != self.model_var.get():
            self.model_var.set(model_value)

    def _apply_default_resources(self, force: bool = False) -> None:
        vocab_path, rules_path = detect_default_resource_paths(
            self.target_lang_var.get(),
            base_dir=self.resource_root,
        )

        if self.supports_vocab:
            vocab_value, self._auto_vocab_path = choose_resource_field_value(
                current_value=self.vocab_path_var.get(),
                previous_auto_value=self._auto_vocab_path,
                new_auto_value=vocab_path,
                force=force,
            )
            if vocab_value != self.vocab_path_var.get():
                self.vocab_path_var.set(vocab_value)

        if self.supports_rules:
            rules_value, self._auto_rules_path = choose_resource_field_value(
                current_value=self.rules_path_var.get(),
                previous_auto_value=self._auto_rules_path,
                new_auto_value=rules_path,
                force=force,
            )
            if rules_value != self.rules_path_var.get():
                self.rules_path_var.set(rules_value)

    def _load_rules_preview(self) -> None:
        if not self.supports_rules or self.rules_preview_text is None:
            return

        content = read_text_file_or_empty(self.rules_path_var.get())
        if not content:
            content = "No rules file loaded."

        self.rules_preview_text.configure(state="normal")
        self.rules_preview_text.delete("1.0", "end")
        self.rules_preview_text.insert("1.0", content)
        self.rules_preview_text.see("1.0")
        self.rules_preview_text.configure(state="disabled")

    def _load_system_prompt_preview(self) -> None:
        if self.system_prompt_text is None:
            return

        content = build_system_prompt_preview(self.tool_key, self.target_lang_var.get())
        self.system_prompt_text.configure(state="normal")
        self.system_prompt_text.delete("1.0", "end")
        self.system_prompt_text.insert("1.0", content)
        self.system_prompt_text.see("1.0")
        self.system_prompt_text.configure(state="disabled")

    def _start_run(self) -> None:
        self.app.start_run(self)

    def _stop_run(self) -> None:
        self.app.stop_run(self)

    def set_running(self, active: bool, any_process_running: bool) -> None:
        if self.run_button is not None:
            self.run_button.configure(state="disabled" if any_process_running else "normal")
        if self.stop_button is not None:
            self.stop_button.configure(
                state="normal" if active and any_process_running else "disabled"
            )

    def append_log(self, text: str) -> None:
        if self.log_text is None:
            return

        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.update_idletasks()
        self.log_text.see("end")
        self.log_text.yview_moveto(1.0)
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        if self.log_text is None:
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def build_env(self) -> dict[str, str]:
        if not self.supports_provider_controls:
            return dict(os.environ)
        return build_script_env(
            self.api_key_var.get(),
            provider=self.provider_var.get(),
            gemini_backend=self.gemini_backend_var.get(),
            google_cloud_location=self.google_cloud_location_var.get(),
        )

    def build_command(self) -> list[str]:
        raise NotImplementedError


class ProcessToolTab(BaseToolTab):
    def __init__(self, app: "ProcessGuiApp", notebook: ttk.Notebook) -> None:
        self.source_file_var = tk.StringVar()
        self.retranslate_all_var = tk.BooleanVar(value=False)
        self.warnings_report_var = tk.BooleanVar(value=True)
        self.selected_input_files: tuple[str, ...] = ()
        self._selected_input_files_display = ""
        super().__init__(
            app,
            notebook,
            tool_key="process",
            title="Translate",
            script_name="translate_cli.py",
            input_filetypes=TRANSLATABLE_FILETYPES,
            supports_vocab=True,
            supports_rules=True,
            input_label="Input file(s)",
        )

    def _browse_input_file(self) -> None:
        selected = tuple(
            filedialog.askopenfilenames(
                title=f"Select input files for {self.title}",
                filetypes=self.input_filetypes,
            )
        )
        if not selected:
            return
        self.selected_input_files = selected
        self._selected_input_files_display = summarize_input_files(selected)
        self.input_file_var.set(self._selected_input_files_display)

    def _resolve_selected_input_files(self) -> tuple[str, ...]:
        display_value = _clean(self.input_file_var.get())
        if self.selected_input_files and display_value == self._selected_input_files_display:
            return self.selected_input_files
        return (display_value,) if display_value else ()

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        self._add_entry_row(
            parent,
            row=start_row,
            label="Source file",
            variable=self.source_file_var,
            browse_command=self._browse_source_file,
        )
        ttk.Label(
            parent,
            text="Required for Android .xml translation runs.",
        ).grid(row=start_row + 1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Checkbutton(
            parent,
            text="Retranslate all entries",
            variable=self.retranslate_all_var,
        ).grid(row=start_row + 2, column=0, columnspan=3, sticky="w", pady=(4, 8))
        ttk.Checkbutton(
            parent,
            text="Write translation warnings JSON report",
            variable=self.warnings_report_var,
        ).grid(row=start_row + 3, column=0, columnspan=3, sticky="w", pady=(0, 8))
        return start_row + 4

    def _browse_source_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select source file",
            filetypes=TRANSLATABLE_FILETYPES,
        )
        if selected:
            self.source_file_var.set(selected)

    def build_config(self) -> ProcessGuiConfig:
        input_files = self._resolve_selected_input_files()
        return ProcessGuiConfig(
            input_file=input_files[0] if len(input_files) == 1 else self.input_file_var.get(),
            input_files=input_files,
            source_file=self.source_file_var.get(),
            source_lang=self.source_lang_var.get(),
            target_lang=self.target_lang_var.get(),
            provider=self.provider_var.get(),
            gemini_backend=self.gemini_backend_var.get(),
            google_cloud_location=self.google_cloud_location_var.get(),
            model=self.model_var.get(),
            thinking_level=self.thinking_level_var.get(),
            batch_size=self.batch_size_var.get(),
            parallel_requests=self.parallel_requests_var.get(),
            vocab_path=self.vocab_path_var.get(),
            rules_path=self.rules_path_var.get(),
            rules_str=self.rules_text.get("1.0", "end-1c") if self.rules_text else "",
            api_key=self.api_key_var.get(),
            flex_mode=self.flex_mode_var.get(),
            retranslate_all=self.retranslate_all_var.get(),
            warnings_report=self.warnings_report_var.get(),
        )

    def build_command(self) -> list[str]:
        return build_process_command(self.build_config())


class ExtractToolTab(BaseToolTab):
    def __init__(self, app: "ProcessGuiApp", notebook: ttk.Notebook) -> None:
        self.mode_var = tk.StringVar(value="missing")
        self.out_format_var = tk.StringVar(value="po")
        self.out_path_var = tk.StringVar()
        self.max_terms_per_batch_var = tk.StringVar(value="80")
        self.max_attempts_var = tk.StringVar(value="5")
        super().__init__(
            app,
            notebook,
            tool_key="extract",
            title="Extract Terms",
            script_name="translate_cli.py",
            input_filetypes=TRANSLATABLE_FILETYPES,
            supports_vocab=True,
            supports_rules=False,
        )

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        self._add_combo_row(
            parent,
            row=start_row,
            label="Mode",
            variable=self.mode_var,
            values=EXTRACT_MODE_CHOICES,
        )
        self._add_combo_row(
            parent,
            row=start_row + 1,
            label="Output format",
            variable=self.out_format_var,
            values=EXTRACT_OUTPUT_CHOICES,
        )
        self._add_entry_row(
            parent,
            row=start_row + 2,
            label="Output path",
            variable=self.out_path_var,
            browse_command=self._browse_output_file,
        )
        self._add_entry_row(
            parent,
            row=start_row + 3,
            label="Max terms / batch",
            variable=self.max_terms_per_batch_var,
        )
        self._add_entry_row(
            parent,
            row=start_row + 4,
            label="Max attempts",
            variable=self.max_attempts_var,
        )
        return start_row + 5

    def _browse_output_file(self) -> None:
        out_format = _clean(self.out_format_var.get()) or "po"
        extension = ".json" if out_format == "json" else ".po"
        selected = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=extension,
            filetypes=[("Output files", f"*{extension}"), ("All files", "*.*")],
        )
        if selected:
            self.out_path_var.set(selected)

    def build_config(self) -> ExtractGuiConfig:
        return ExtractGuiConfig(
            input_file=self.input_file_var.get(),
            source_lang=self.source_lang_var.get(),
            target_lang=self.target_lang_var.get(),
            provider=self.provider_var.get(),
            gemini_backend=self.gemini_backend_var.get(),
            google_cloud_location=self.google_cloud_location_var.get(),
            model=self.model_var.get(),
            thinking_level=self.thinking_level_var.get(),
            batch_size=self.batch_size_var.get(),
            parallel_requests=self.parallel_requests_var.get(),
            vocab_path=self.vocab_path_var.get(),
            api_key=self.api_key_var.get(),
            flex_mode=self.flex_mode_var.get(),
            mode=self.mode_var.get(),
            out_format=self.out_format_var.get(),
            out_path=self.out_path_var.get(),
            max_terms_per_batch=self.max_terms_per_batch_var.get(),
            max_attempts=self.max_attempts_var.get(),
        )

    def build_command(self) -> list[str]:
        return build_extract_command(self.build_config())


class LocalExtractToolTab(BaseToolTab):
    def __init__(self, app: "ProcessGuiApp", notebook: ttk.Notebook) -> None:
        self.mode_var = tk.StringVar(value="missing")
        self.max_length_var = tk.StringVar(value="1")
        self.out_path_var = tk.StringVar()
        self.include_rejected_var = tk.BooleanVar(value=False)
        self.to_po_var = tk.BooleanVar(value=False)
        self.include_borderline_var = tk.BooleanVar(value=False)
        self.mode_combo: ttk.Combobox | None = None
        self.max_length_combo: ttk.Combobox | None = None
        self.include_rejected_button: ttk.Checkbutton | None = None
        self.include_borderline_button: ttk.Checkbutton | None = None
        self.input_file_button: ttk.Button | None = None
        self.input_folder_button: ttk.Button | None = None
        super().__init__(
            app,
            notebook,
            tool_key="extract_local",
            title="Local Extract",
            script_name="translate_cli.py",
            input_filetypes=LOCAL_EXTRACT_FILETYPES,
            supports_vocab=True,
            supports_rules=False,
            input_label="Input file / folder / JSON",
            supports_provider_controls=False,
            preview_label="Task info",
        )
        self.to_po_var.trace_add("write", self._on_to_po_changed)
        self._refresh_local_mode_controls()

    def _build_input_row(self, parent: ttk.Frame, row: int) -> int:
        ttk.Label(parent, text=self.input_label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.input_file_var).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=2, sticky="nw", padx=(8, 0), pady=4)
        self.input_file_button = ttk.Button(
            button_frame,
            text="Source file",
            command=self._browse_local_extract_file,
        )
        self.input_file_button.grid(row=0, column=0, sticky="ew")
        self.input_folder_button = ttk.Button(
            button_frame,
            text="Source folder",
            command=self._browse_local_extract_folder,
        )
        self.input_folder_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        return 1

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        ttk.Checkbutton(
            parent,
            text="Convert local JSON to PO handoff",
            variable=self.to_po_var,
        ).grid(row=start_row, column=0, columnspan=3, sticky="w", pady=(4, 4))

        self.mode_combo = self._add_combo_row(
            parent,
            row=start_row + 1,
            label="Mode",
            variable=self.mode_var,
            values=EXTRACT_MODE_CHOICES,
        )

        self.max_length_combo = self._add_combo_row(
            parent,
            row=start_row + 2,
            label="Max length",
            variable=self.max_length_var,
            values=LOCAL_EXTRACT_MAX_LENGTH_CHOICES,
        )

        self._add_entry_row(
            parent,
            row=start_row + 3,
            label="Output path",
            variable=self.out_path_var,
            browse_command=self._browse_output_file,
        )
        self.include_rejected_button = ttk.Checkbutton(
            parent,
            text="Include rejected terms in JSON output",
            variable=self.include_rejected_var,
        )
        self.include_rejected_button.grid(row=start_row + 4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.include_borderline_button = ttk.Checkbutton(
            parent,
            text="Include borderline terms in generated PO handoff",
            variable=self.include_borderline_var,
        )
        self.include_borderline_button.grid(row=start_row + 5, column=0, columnspan=3, sticky="w", pady=(0, 8))
        return start_row + 6

    def _on_to_po_changed(self, *_args: object) -> None:
        self._refresh_local_mode_controls()

    def _refresh_local_mode_controls(self) -> None:
        to_po_mode = self.to_po_var.get()
        if self.mode_combo is not None:
            self.mode_combo.configure(state="disabled" if to_po_mode else "readonly")
        if self.max_length_combo is not None:
            self.max_length_combo.configure(state="disabled" if to_po_mode else "readonly")
        if self.include_rejected_button is not None:
            self.include_rejected_button.configure(state="disabled" if to_po_mode else "normal")
        po_enabled = True
        if self.include_borderline_button is not None:
            self.include_borderline_button.configure(state="normal" if po_enabled else "disabled")
        if self.input_file_button is not None:
            self.input_file_button.configure(text="JSON file" if to_po_mode else "Source file")
        if self.input_folder_button is not None:
            self.input_folder_button.configure(state="disabled" if to_po_mode else "normal")
        if to_po_mode:
            self.include_rejected_var.set(False)

    def _browse_output_file(self) -> None:
        to_po_mode = self.to_po_var.get()
        extension = ".po" if to_po_mode else ".json"
        selected = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=extension,
            filetypes=[("Output files", f"*{extension}"), ("All files", "*.*")],
        )
        if selected:
            self.out_path_var.set(selected)

    def _browse_input_file(self) -> None:
        self._browse_local_extract_file()

    def _browse_local_extract_file(self) -> None:
        title, filetypes = get_local_extract_file_dialog_config(self.to_po_var.get())
        selected = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
        if selected:
            self.input_file_var.set(selected)

    def _browse_local_extract_folder(self) -> None:
        if self.to_po_var.get():
            return
        selected = filedialog.askdirectory(
            title="Select source folder for local extraction",
            mustexist=True,
        )
        if selected:
            self.input_file_var.set(selected)

    def build_config(self) -> LocalExtractGuiConfig:
        return LocalExtractGuiConfig(
            input_file=self.input_file_var.get(),
            source_lang=self.source_lang_var.get(),
            target_lang=self.target_lang_var.get(),
            vocab_path=self.vocab_path_var.get(),
            mode=self.mode_var.get(),
            max_length=self.max_length_var.get(),
            out_path=self.out_path_var.get(),
            include_rejected=self.include_rejected_var.get(),
            to_po=self.to_po_var.get(),
            also_po=not self.to_po_var.get(),
            include_borderline=self.include_borderline_var.get(),
        )

    def build_command(self) -> list[str]:
        return build_local_extract_command(self.build_config())

    def build_env(self) -> dict[str, str]:
        return dict(os.environ)


class CheckToolTab(BaseToolTab):
    def __init__(self, app: "ProcessGuiApp", notebook: ttk.Notebook) -> None:
        self.num_messages_var = tk.StringVar()
        self.out_path_var = tk.StringVar()
        self.include_ok_var = tk.BooleanVar(value=False)
        self.max_attempts_var = tk.StringVar(value="5")
        super().__init__(
            app,
            notebook,
            tool_key="check",
            title="Check Translations",
            script_name="translate_cli.py",
            input_filetypes=CHECK_FILETYPES,
            supports_vocab=True,
            supports_rules=True,
        )

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        self._add_entry_row(
            parent,
            row=start_row,
            label="Output path",
            variable=self.out_path_var,
            browse_command=self._browse_output_file,
        )
        self._add_entry_row(
            parent,
            row=start_row + 1,
            label="Probe / num messages",
            variable=self.num_messages_var,
        )
        self._add_entry_row(
            parent,
            row=start_row + 2,
            label="Max attempts",
            variable=self.max_attempts_var,
        )
        ttk.Checkbutton(
            parent,
            text="Include entries with no issues",
            variable=self.include_ok_var,
        ).grid(row=start_row + 3, column=0, columnspan=3, sticky="w", pady=(4, 8))
        return start_row + 4

    def _browse_output_file(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select output report path",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.out_path_var.set(selected)

    def build_config(self) -> CheckGuiConfig:
        return CheckGuiConfig(
            input_file=self.input_file_var.get(),
            source_lang=self.source_lang_var.get(),
            target_lang=self.target_lang_var.get(),
            provider=self.provider_var.get(),
            gemini_backend=self.gemini_backend_var.get(),
            google_cloud_location=self.google_cloud_location_var.get(),
            model=self.model_var.get(),
            thinking_level=self.thinking_level_var.get(),
            batch_size=self.batch_size_var.get(),
            parallel_requests=self.parallel_requests_var.get(),
            vocab_path=self.vocab_path_var.get(),
            rules_path=self.rules_path_var.get(),
            rules_str=self.rules_text.get("1.0", "end-1c") if self.rules_text else "",
            api_key=self.api_key_var.get(),
            flex_mode=self.flex_mode_var.get(),
            num_messages=self.num_messages_var.get(),
            out_path=self.out_path_var.get(),
            include_ok=self.include_ok_var.get(),
            max_attempts=self.max_attempts_var.get(),
        )

    def build_command(self) -> list[str]:
        return build_check_command(self.build_config())


class ReviseToolTab(BaseToolTab):
    def __init__(self, app: "ProcessGuiApp", notebook: ttk.Notebook) -> None:
        self.source_file_var = tk.StringVar()
        self.out_path_var = tk.StringVar()
        self.num_messages_var = tk.StringVar()
        self.max_attempts_var = tk.StringVar(value="5")
        self.in_place_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.instruction_text: ScrolledText | None = None
        super().__init__(
            app,
            notebook,
            tool_key="revise",
            title="Revise Translations",
            script_name="translate_cli.py",
            input_filetypes=TRANSLATABLE_FILETYPES,
            supports_vocab=True,
            supports_rules=True,
        )

    def _build_tool_specific_fields(self, parent: ttk.Frame, start_row: int) -> int:
        self._add_entry_row(
            parent,
            row=start_row,
            label="Source file",
            variable=self.source_file_var,
            browse_command=self._browse_source_file,
        )
        ttk.Label(
            parent,
            text="Required for Android .xml, .strings, .resx, and .txt revision runs.",
        ).grid(row=start_row + 1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._add_entry_row(
            parent,
            row=start_row + 2,
            label="Output path",
            variable=self.out_path_var,
            browse_command=self._browse_output_file,
        )
        self._add_entry_row(
            parent,
            row=start_row + 3,
            label="Probe / num messages",
            variable=self.num_messages_var,
        )
        self._add_entry_row(
            parent,
            row=start_row + 4,
            label="Max attempts",
            variable=self.max_attempts_var,
        )

        ttk.Checkbutton(
            parent,
            text="Overwrite input file",
            variable=self.in_place_var,
        ).grid(row=start_row + 5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            parent,
            text="Dry run only",
            variable=self.dry_run_var,
        ).grid(row=start_row + 6, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(parent, text="Instruction").grid(row=start_row + 7, column=0, sticky="nw")
        self.instruction_text = ScrolledText(
            parent,
            wrap="word",
            height=6,
            font=("Consolas", 10),
        )
        self.instruction_text.grid(
            row=start_row + 8,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(4, 8),
        )
        return start_row + 9

    def _browse_source_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select source file",
            filetypes=TRANSLATABLE_FILETYPES,
        )
        if selected:
            self.source_file_var.set(selected)

    def _browse_output_file(self) -> None:
        input_path = _clean(self.input_file_var.get())
        extension = os.path.splitext(input_path)[1] or ".txt"
        selected = filedialog.asksaveasfilename(
            title="Select revised output file",
            defaultextension=extension,
            filetypes=[("Matching files", f"*{extension}"), ("All files", "*.*")],
        )
        if selected:
            self.out_path_var.set(selected)

    def build_config(self) -> ReviseGuiConfig:
        return ReviseGuiConfig(
            input_file=self.input_file_var.get(),
            source_file=self.source_file_var.get(),
            source_lang=self.source_lang_var.get(),
            target_lang=self.target_lang_var.get(),
            provider=self.provider_var.get(),
            gemini_backend=self.gemini_backend_var.get(),
            google_cloud_location=self.google_cloud_location_var.get(),
            model=self.model_var.get(),
            thinking_level=self.thinking_level_var.get(),
            batch_size=self.batch_size_var.get(),
            parallel_requests=self.parallel_requests_var.get(),
            vocab_path=self.vocab_path_var.get(),
            rules_path=self.rules_path_var.get(),
            rules_str=self.rules_text.get("1.0", "end-1c") if self.rules_text else "",
            api_key=self.api_key_var.get(),
            flex_mode=self.flex_mode_var.get(),
            instruction=self.instruction_text.get("1.0", "end-1c") if self.instruction_text else "",
            num_messages=self.num_messages_var.get(),
            out_path=self.out_path_var.get(),
            max_attempts=self.max_attempts_var.get(),
            in_place=self.in_place_var.get(),
            dry_run=self.dry_run_var.get(),
        )

    def build_command(self) -> list[str]:
        return build_revise_command(self.build_config())


class ProcessGuiApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.resource_root = build_resource_root()
        self.api_key_var = tk.StringVar()
        self.provider_var = tk.StringVar(value=DEFAULT_PROVIDER)
        self.gemini_backend_var = tk.StringVar(value="studio")
        self.google_cloud_location_var = tk.StringVar(value="global")
        self.thinking_level_var = tk.StringVar()
        self.flex_mode_var = tk.BooleanVar(value=False)
        self.queue: queue.Queue[tuple[str, str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.active_tool_key: str | None = None
        self.active_log_handle: TextIO | None = None
        self.active_log_path: str | None = None
        self.progress_value_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0.0%")
        self.status_text_var = tk.StringVar(value="Ready.")
        self.tabs: dict[str, BaseToolTab] = {}
        self._clipboard_menu = tk.Menu(self, tearoff=False)
        self._clipboard_target: tk.Misc | None = None

        self.master.title("Translation Tools Tk Launcher")
        self.master.geometry("1180x820")
        self.master.minsize(980, 680)
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_widgets()
        self._bind_clipboard_shortcuts()
        self.after(100, self._poll_queue)

    def _build_widgets(self) -> None:
        ttk.Label(
            self,
            text="Run translation, revision, glossary extraction, and QA from one Tk interface.",
        ).grid(row=0, column=0, sticky="w")

        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(10, 10))

        process_tab = ProcessToolTab(self, notebook)
        revise_tab = ReviseToolTab(self, notebook)
        extract_tab = ExtractToolTab(self, notebook)
        extract_local_tab = LocalExtractToolTab(self, notebook)
        check_tab = CheckToolTab(self, notebook)

        self.tabs = {
            process_tab.tool_key: process_tab,
            revise_tab.tool_key: revise_tab,
            extract_tab.tool_key: extract_tab,
            extract_local_tab.tool_key: extract_local_tab,
            check_tab.tool_key: check_tab,
        }

        notebook.add(process_tab, text="Translate")
        notebook.add(revise_tab, text="Revise")
        notebook.add(extract_tab, text="Extract")
        notebook.add(extract_local_tab, text="Local Extract")
        notebook.add(check_tab, text="Check")

        status_frame = ttk.Frame(self)
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Progressbar(
            status_frame,
            maximum=100.0,
            variable=self.progress_value_var,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(status_frame, textvariable=self.status_text_var).grid(
            row=0,
            column=2,
            sticky="w",
        )
        ttk.Label(status_frame, textvariable=self.progress_text_var, width=8).grid(
            row=0,
            column=3,
            sticky="e",
            padx=(8, 0),
        )

        self.grid(sticky="nsew")

    def _widget_class_name(self, widget: tk.Misc | None) -> str:
        if widget is None:
            return ""
        try:
            return str(widget.winfo_class() or "")
        except tk.TclError:
            return ""

    def _widget_state(self, widget: tk.Misc | None) -> str:
        if widget is None:
            return ""
        cget = getattr(widget, "cget", None)
        if not callable(cget):
            return ""
        try:
            return str(cget("state") or "")
        except (tk.TclError, TypeError):
            return ""

    def _is_text_widget(self, widget: tk.Misc | None) -> bool:
        return self._widget_class_name(widget) == "Text"

    def _supports_clipboard_widget(self, widget: tk.Misc | None) -> bool:
        return widget_supports_clipboard(self._widget_class_name(widget))

    def _widget_is_editable(self, widget: tk.Misc | None) -> bool:
        return widget_is_editable(
            self._widget_class_name(widget),
            self._widget_state(widget),
        )

    def _selected_text(self, widget: tk.Misc | None) -> str:
        if widget is None or not self._supports_clipboard_widget(widget):
            return ""

        try:
            if self._is_text_widget(widget):
                return str(widget.get("sel.first", "sel.last"))
            selection_present = getattr(widget, "selection_present", None)
            if callable(selection_present) and not selection_present():
                return ""
            return str(widget.selection_get())
        except (tk.TclError, AttributeError):
            return ""

    def _replace_selection(self, widget: tk.Misc, text: str) -> None:
        if self._is_text_widget(widget):
            try:
                widget.delete("sel.first", "sel.last")
            except tk.TclError:
                pass
            widget.insert("insert", text)
            return

        try:
            widget.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        widget.insert("insert", text)

    def _bind_clipboard_shortcuts(self) -> None:
        self._clipboard_menu.add_command(label="Cut", command=self._menu_cut)
        self._clipboard_menu.add_command(label="Copy", command=self._menu_copy)
        self._clipboard_menu.add_command(label="Paste", command=self._menu_paste)
        self._clipboard_menu.add_separator()
        self._clipboard_menu.add_command(label="Select All", command=self._menu_select_all)

        for widget_class in sorted(CLIPBOARD_WIDGET_CLASSES):
            self.master.bind_class(widget_class, "<Control-a>", self._handle_select_all, add="+")
            self.master.bind_class(widget_class, "<Control-A>", self._handle_select_all, add="+")
            self.master.bind_class(widget_class, "<Control-c>", self._handle_copy, add="+")
            self.master.bind_class(widget_class, "<Control-C>", self._handle_copy, add="+")
            self.master.bind_class(widget_class, "<Control-v>", self._handle_paste, add="+")
            self.master.bind_class(widget_class, "<Control-V>", self._handle_paste, add="+")
            self.master.bind_class(widget_class, "<Control-x>", self._handle_cut, add="+")
            self.master.bind_class(widget_class, "<Control-X>", self._handle_cut, add="+")
            self.master.bind_class(widget_class, "<Control-Insert>", self._handle_copy, add="+")
            self.master.bind_class(widget_class, "<Shift-Insert>", self._handle_paste, add="+")
            self.master.bind_class(widget_class, "<Shift-Delete>", self._handle_cut, add="+")
            self.master.bind_class(widget_class, "<Button-3>", self._show_clipboard_menu, add="+")

    def _event_widget(self, event: tk.Event[tk.Misc] | None) -> tk.Misc | None:
        if event is not None and getattr(event, "widget", None) is not None:
            return event.widget
        return self.master.focus_get()

    def _copy_selection_to_clipboard(self, widget: tk.Misc | None) -> bool:
        selected = self._selected_text(widget)
        if not selected:
            return False
        self.master.clipboard_clear()
        self.master.clipboard_append(selected)
        return True

    def _handle_copy(self, event: tk.Event[tk.Misc] | None = None) -> str | None:
        widget = self._event_widget(event)
        if not self._supports_clipboard_widget(widget):
            return None
        if self._copy_selection_to_clipboard(widget):
            return "break"
        return "break"

    def _handle_cut(self, event: tk.Event[tk.Misc] | None = None) -> str | None:
        widget = self._event_widget(event)
        if not self._supports_clipboard_widget(widget):
            return None
        if not self._widget_is_editable(widget):
            return "break"
        if self._copy_selection_to_clipboard(widget):
            if self._is_text_widget(widget):
                widget.delete("sel.first", "sel.last")
            else:
                widget.delete("sel.first", "sel.last")
        return "break"

    def _handle_paste(self, event: tk.Event[tk.Misc] | None = None) -> str | None:
        widget = self._event_widget(event)
        if not self._supports_clipboard_widget(widget):
            return None
        if not self._widget_is_editable(widget):
            return "break"
        try:
            text = self.master.clipboard_get()
        except tk.TclError:
            return "break"
        self._replace_selection(widget, text)
        return "break"

    def _handle_select_all(self, event: tk.Event[tk.Misc] | None = None) -> str | None:
        widget = self._event_widget(event)
        if not self._supports_clipboard_widget(widget):
            return None
        try:
            widget.focus_set()
            if self._is_text_widget(widget):
                widget.tag_add("sel", "1.0", "end-1c")
                widget.mark_set("insert", "1.0")
                widget.see("insert")
            else:
                widget.selection_range(0, "end")
                widget.icursor("end")
        except (tk.TclError, AttributeError):
            return None
        return "break"

    def _show_clipboard_menu(self, event: tk.Event[tk.Misc]) -> str | None:
        widget = self._event_widget(event)
        if not self._supports_clipboard_widget(widget):
            return None

        self._clipboard_target = widget
        editable = self._widget_is_editable(widget)
        has_selection = bool(self._selected_text(widget))
        try:
            has_clipboard = bool(self.master.clipboard_get())
        except tk.TclError:
            has_clipboard = False

        self._clipboard_menu.entryconfigure(0, state="normal" if editable and has_selection else "disabled")
        self._clipboard_menu.entryconfigure(1, state="normal" if has_selection else "disabled")
        self._clipboard_menu.entryconfigure(2, state="normal" if editable and has_clipboard else "disabled")
        self._clipboard_menu.entryconfigure(4, state="normal")

        try:
            self._clipboard_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._clipboard_menu.grab_release()
        return "break"

    def _focus_clipboard_target(self) -> None:
        if self._clipboard_target is None:
            return
        try:
            self._clipboard_target.focus_set()
        except tk.TclError:
            pass

    def _menu_cut(self) -> None:
        self._focus_clipboard_target()
        self._handle_cut()

    def _menu_copy(self) -> None:
        self._focus_clipboard_target()
        self._handle_copy()

    def _menu_paste(self) -> None:
        self._focus_clipboard_target()
        self._handle_paste()

    def _menu_select_all(self) -> None:
        self._focus_clipboard_target()
        self._handle_select_all()

    def _set_status(self, text: str, percent: float | None = None) -> None:
        if percent is not None:
            clamped = _clamp_percent(percent)
            self.progress_value_var.set(clamped)
            self.progress_text_var.set(f"{clamped:.1f}%")
        self.status_text_var.set(text)

    def _set_running_state(self, active_tool_key: str | None) -> None:
        any_running = active_tool_key is not None
        for tool_key, tab in self.tabs.items():
            tab.set_running(active=tool_key == active_tool_key, any_process_running=any_running)

    def _close_run_log(self) -> None:
        if self.active_log_handle is not None:
            try:
                self.active_log_handle.close()
            except OSError:
                pass
        self.active_log_handle = None
        self.active_log_path = None

    def _start_run_log(self, tab: BaseToolTab) -> str | None:
        self._close_run_log()
        log_dir = build_log_dir(self.resource_root)
        os.makedirs(log_dir, exist_ok=True)
        log_path = build_run_log_path(
            tab.tool_key,
            tab.input_file_var.get(),
            base_dir=self.resource_root,
        )
        self.active_log_handle = open(log_path, "w", encoding="utf-8", newline="")
        self.active_log_path = log_path
        return log_path

    def _append_tool_output(self, tool_key: str, text: str) -> None:
        tab = self.tabs.get(tool_key)
        if tab is not None:
            tab.append_log(text)

        if tool_key != self.active_tool_key or self.active_log_handle is None:
            return

        self.active_log_handle.write(text)
        self.active_log_handle.flush()

    def start_run(self, tab: BaseToolTab) -> None:
        if self.process is not None and self.process.poll() is None:
            if self.active_tool_key == tab.tool_key:
                return
            messagebox.showinfo(
                "Busy",
                "Another script is already running. Stop it before starting a new one.",
            )
            return

        try:
            command = tab.build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        env = tab.build_env()
        self.active_tool_key = tab.tool_key
        log_path: str | None = None
        log_error: str | None = None
        try:
            log_path = self._start_run_log(tab)
        except OSError as exc:
            log_error = str(exc)
        self._set_running_state(tab.tool_key)
        self._set_status(f"Running {tab.title}", 0.0)
        if log_path:
            self._append_tool_output(tab.tool_key, f"Log file: {log_path}\n")
        elif log_error:
            self._append_tool_output(
                tab.tool_key,
                f"Warning: could not open log file: {log_error}\n",
            )
        self._append_tool_output(tab.tool_key, f"$ {subprocess.list2cmdline(command)}\n")
        self._append_tool_output(tab.tool_key, f"Starting {tab.script_name}...\n\n")

        worker = threading.Thread(
            target=self._run_process,
            args=(tab.tool_key, command, env),
            daemon=True,
        )
        worker.start()

    def stop_run(self, tab: BaseToolTab | None = None) -> None:
        if self.process is None or self.process.poll() is not None:
            return

        if tab is not None and self.active_tool_key != tab.tool_key:
            return

        active_tab = self.tabs.get(self.active_tool_key or "")
        if active_tab is not None:
            self._append_tool_output(active_tab.tool_key, "\nStopping process...\n")
        self._set_status("Stopping current script...")

        try:
            self.process.terminate()
        except OSError as exc:
            if active_tab is not None:
                active_tab.append_log(f"Stop failed: {exc}\n")

    def _run_process(
        self,
        tool_key: str,
        command: list[str],
        env: dict[str, str],
    ) -> None:
        working_dir = build_resource_root()

        try:
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except Exception as exc:
            self.queue.put(("error", tool_key, str(exc)))
            return

        self.process = process
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self.queue.put(("line", tool_key, line))
            return_code = process.wait()
            self.queue.put(("done", tool_key, return_code))
        finally:
            self.process = None

    def _poll_queue(self) -> None:
        try:
            while True:
                event, tool_key, payload = self.queue.get_nowait()
                tab = self.tabs.get(tool_key)
                if tab is None:
                    continue

                if event == "line":
                    line = str(payload)
                    self._append_tool_output(tool_key, line)
                    if tool_key == self.active_tool_key:
                        percent = parse_progress_percent(line)
                        if percent is not None:
                            self._set_status(f"Running {tab.title}", percent)
                    continue

                if event == "error":
                    self._append_tool_output(
                        tool_key,
                        f"\nFailed to start {tab.script_name}: {payload}\n",
                    )
                    self.active_tool_key = None
                    self._set_running_state(None)
                    self._set_status(f"Failed to start {tab.title}")
                    self._close_run_log()
                    messagebox.showerror(
                        "Launch failed",
                        f"Failed to start {tab.script_name}:\n{payload}",
                    )
                    continue

                if event == "done":
                    return_code = int(payload)
                    self._append_tool_output(
                        tool_key,
                        f"\n{tab.script_name} exited with code {return_code}.\n",
                    )
                    if self.active_log_path:
                        self._append_tool_output(
                            tool_key,
                            f"Log saved to: {self.active_log_path}\n",
                        )
                    was_active = tool_key == self.active_tool_key
                    self.active_tool_key = None
                    self._set_running_state(None)
                    self._close_run_log()
                    if return_code == 0:
                        if was_active:
                            self._set_status(f"{tab.title} completed.", 100.0)
                        messagebox.showinfo(
                            "Completed",
                            f"{tab.script_name} finished successfully.",
                        )
                    else:
                        self._set_status(
                            f"{tab.title} failed (exit {return_code}).",
                            self.progress_value_var.get(),
                        )
                        messagebox.showerror(
                            "Process failed",
                            f"{tab.script_name} exited with code {return_code}.",
                        )
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    ProcessGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
