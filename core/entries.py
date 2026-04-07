from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from core.task_issues import TaskIssue, normalize_task_issue

MAX_PROMPT_CONTEXT_CHARS = 180
MAX_PROMPT_NOTE_CHARS = 300
MAX_PROMPT_OCCURRENCES = 3

TRANSLATION_WARNING_CODES: Tuple[str, ...] = (
    "translate.ambiguous_term",
    "translate.unclear_source_meaning",
    "translate.glossary_variant_choice",
    "translate.possible_untranslated_token",
    "translate.placeholder_attention",
    "translate.length_or_ui_fit_risk",
)
DEFAULT_TRANSLATION_WARNING_CODE = "translate.unclear_source_meaning"
TranslationWarning = TaskIssue


@dataclass
class TranslationResult:
    text: str = ""
    plural_texts: List[str] = field(default_factory=list)
    warnings: List[TaskIssue] = field(default_factory=list)


def json_load_maybe(text: str) -> Any:
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


def _normalize_translation_payload(payload: Any) -> Dict[str, TranslationResult]:
    results: Dict[str, TranslationResult] = {}
    if isinstance(payload, dict):
        if "translations" in payload:
            items = payload.get("translations")
            if not isinstance(items, list):
                return results
            for item in items:
                if not isinstance(item, dict):
                    continue
                msg_id = item.get("id")
                if msg_id is None:
                    continue
                text = item.get("text")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                plural_texts_raw = item.get("plural_texts")
                plural_texts: List[str] = []
                if isinstance(plural_texts_raw, list):
                    for value in plural_texts_raw:
                        if value is None:
                            continue
                        plural_texts.append(value if isinstance(value, str) else str(value))
                warnings_raw = item.get("warnings")
                warnings: List[TaskIssue] = []
                if isinstance(warnings_raw, list):
                    for value in warnings_raw:
                        normalized_warning = normalize_translation_warning(value)
                        if normalized_warning is not None:
                            warnings.append(normalized_warning)
                results[str(msg_id)] = TranslationResult(
                    text=text,
                    plural_texts=plural_texts,
                    warnings=warnings,
                )
            return results
        for key, value in payload.items():
            if isinstance(value, str):
                results[str(key)] = TranslationResult(text=value)
            elif isinstance(value, dict):
                text = value.get("text")
                if isinstance(text, str):
                    warnings_raw = value.get("warnings")
                    warnings: List[TaskIssue] = []
                    if isinstance(warnings_raw, list):
                        for warning_value in warnings_raw:
                            normalized_warning = normalize_translation_warning(warning_value)
                            if normalized_warning is not None:
                                warnings.append(normalized_warning)
                    results[str(key)] = TranslationResult(text=text, warnings=warnings)
    return results


def normalize_translation_warning(value: Any) -> TaskIssue | None:
    return normalize_task_issue(
        value,
        allowed_codes=TRANSLATION_WARNING_CODES,
        default_code=DEFAULT_TRANSLATION_WARNING_CODE,
        allowed_severities=("warning", "info"),
        default_severity="warning",
        default_origin="model",
    )


def parse_response(response_payload: Any) -> Dict[str, TranslationResult]:
    if isinstance(response_payload, (dict, list)):
        return _normalize_translation_payload(response_payload)
    if isinstance(response_payload, str):
        return _normalize_translation_payload(json_load_maybe(response_payload))
    parsed_payload = getattr(response_payload, "parsed", None)
    if parsed_payload is not None:
        return _normalize_translation_payload(parsed_payload)
    text_payload = getattr(response_payload, "text", None) or ""
    return _normalize_translation_payload(json_load_maybe(text_payload))


def is_non_empty_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalize_model_escaped_text(source_text: str, candidate_text: str) -> str:
    if not isinstance(candidate_text, str):
        return candidate_text

    normalized = candidate_text
    replacements = (
        ("\n", "\\n"),
        ("\t", "\\t"),
        ("\r", "\\r"),
        ("\a", "\\a"),
        ("\b", "\\b"),
        ("\f", "\\f"),
        ("\v", "\\v"),
    )
    for actual_char, escaped_seq in replacements:
        source_uses_actual = actual_char in source_text and escaped_seq not in source_text
        candidate_uses_literal = escaped_seq in normalized and actual_char not in normalized
        if source_uses_actual and candidate_uses_literal:
            normalized = normalized.replace(escaped_seq, actual_char)

        source_uses_literal = escaped_seq in source_text and actual_char not in source_text
        candidate_uses_actual = actual_char in normalized and escaped_seq not in normalized
        if source_uses_literal and candidate_uses_actual:
            normalized = normalized.replace(actual_char, escaped_seq)

    source_uses_quote = '"' in source_text and '\\"' not in source_text
    candidate_uses_escaped_quote = '\\"' in normalized and '"' not in normalized
    if source_uses_quote and candidate_uses_escaped_quote:
        normalized = normalized.replace('\\"', '"')
    return normalized


def is_plural_entry(entry: Any) -> bool:
    return bool(getattr(entry, "msgid_plural", None))


def get_plural_form_count(entry: Any) -> int:
    if not is_plural_entry(entry):
        return 0
    plural_map = getattr(entry, "msgstr_plural", None)
    if isinstance(plural_map, dict) and plural_map:
        return max(2, len(plural_map))
    return 2


def translation_has_content(result: TranslationResult | None) -> bool:
    if result is None:
        return False
    if is_non_empty_text(result.text):
        return True
    return any(is_non_empty_text(text) for text in result.plural_texts)


def plural_key_sort_key(key: Any) -> Tuple[int, Any]:
    try:
        return 0, int(key)
    except (TypeError, ValueError):
        return 1, str(key)


def build_entry_source_text(entry: Any) -> str:
    if is_plural_entry(entry):
        return f"Singular: {entry.msgid}\nPlural: {entry.msgid_plural}"
    return entry.msgid


def apply_translation_to_entry(entry: Any, result: TranslationResult) -> bool:
    source_text = build_entry_source_text(entry)

    if is_plural_entry(entry):
        plural_map = getattr(entry, "msgstr_plural", None)
        if not isinstance(plural_map, dict):
            return False
        plural_keys = sorted(plural_map.keys(), key=plural_key_sort_key)
        if not plural_keys:
            plural_keys = [0]

        usable_forms = [
            normalize_model_escaped_text(source_text, text)
            for text in result.plural_texts
            if is_non_empty_text(text)
        ]
        if usable_forms:
            for idx, key in enumerate(plural_keys):
                plural_map[key] = usable_forms[idx] if idx < len(usable_forms) else usable_forms[-1]
            if hasattr(entry, "mark_translated"):
                entry.mark_translated()
            return True

        if is_non_empty_text(result.text):
            normalized_text = normalize_model_escaped_text(source_text, result.text)
            for key in plural_keys:
                plural_map[key] = normalized_text
            if hasattr(entry, "mark_translated"):
                entry.mark_translated()
            return True
        return False

    if not result.text:
        return False
    entry.msgstr = normalize_model_escaped_text(source_text, result.text)
    if hasattr(entry, "mark_translated"):
        entry.mark_translated()
    return True


def _normalize_prompt_hint(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def get_entry_prompt_context_and_note(entry: Any) -> Tuple[str | None, str | None]:
    context_parts: List[str] = []
    note_parts: List[str] = []

    def add_unique(items: List[str], value: str | None) -> None:
        if value is None:
            return
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in items:
            items.append(cleaned)

    add_unique(context_parts, getattr(entry, "msgctxt", None))
    add_unique(context_parts, getattr(entry, "prompt_context", None))
    add_unique(note_parts, getattr(entry, "prompt_note", None))
    add_unique(note_parts, getattr(entry, "tcomment", None))
    add_unique(note_parts, getattr(entry, "comment", None))

    occurrences = getattr(entry, "occurrences", None)
    if isinstance(occurrences, list) and occurrences:
        formatted: List[str] = []
        for occurrence in occurrences[:MAX_PROMPT_OCCURRENCES]:
            if not isinstance(occurrence, tuple) or not occurrence:
                continue
            file_part = str(occurrence[0]).strip() if occurrence[0] is not None else ""
            if not file_part:
                continue
            line_part = ""
            if len(occurrence) > 1 and occurrence[1] is not None and str(occurrence[1]).strip():
                line_part = str(occurrence[1]).strip()
            formatted.append(f"{file_part}:{line_part}" if line_part else file_part)
        if formatted:
            add_unique(note_parts, f"locations: {', '.join(formatted)}")

    context = _normalize_prompt_hint(" | ".join(context_parts), MAX_PROMPT_CONTEXT_CHARS)
    note = _normalize_prompt_hint(" | ".join(note_parts), MAX_PROMPT_NOTE_CHARS)
    return (context or None), (note or None)


def build_prompt_message_payload(entry: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"source": build_entry_source_text(entry)}
    context, note = get_entry_prompt_context_and_note(entry)
    if is_plural_entry(entry):
        plural_forms = get_plural_form_count(entry)
        payload["plural_forms"] = plural_forms
        plural_note = (
            f"plural forms required: {plural_forms}"
            " | for target languages with no plural difference (for example Kazakh),"
            " prefer the source plural variant as the basis for translation and repeat"
            " that wording in all required plural slots"
        )
        note = f"{note} | {plural_note}" if note else plural_note
    if context:
        payload["context"] = context
    if note:
        payload["note"] = note
    return payload


__all__ = [
    "TranslationResult",
    "apply_translation_to_entry",
    "build_entry_source_text",
    "build_prompt_message_payload",
    "get_entry_prompt_context_and_note",
    "get_plural_form_count",
    "is_non_empty_text",
    "is_plural_entry",
    "json_load_maybe",
    "normalize_model_escaped_text",
    "parse_response",
    "plural_key_sort_key",
    "translation_has_content",
]
