from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence

ISSUE_SEVERITIES: tuple[str, ...] = ("error", "warning", "info")


@dataclass(frozen=True, slots=True)
class TaskIssue:
    code: str
    message: str
    severity: str = "warning"
    origin: str = ""
    source_fragment: str = ""
    translation_fragment: str = ""
    suggested_translation: str = ""


def build_task_issue_schema(
    allowed_codes: Sequence[str],
    *,
    allowed_severities: Sequence[str],
    include_origin: bool = False,
    include_fragments: bool = False,
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "code": {"type": "string", "enum": list(allowed_codes)},
        "message": {"type": "string"},
        "severity": {"type": "string", "enum": list(allowed_severities)},
    }
    if include_origin:
        properties["origin"] = {"type": "string"}
    if include_fragments:
        properties["source_fragment"] = {"type": "string"}
        properties["translation_fragment"] = {"type": "string"}
        properties["suggested_translation"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["code", "message", "severity"],
        "additionalProperties": False,
    }


def normalize_task_issue(
    value: Any,
    *,
    allowed_codes: Sequence[str],
    default_code: str,
    allowed_severities: Sequence[str],
    default_severity: str,
    default_origin: str = "",
    legacy_code_builder: Callable[[Dict[str, Any]], str] | None = None,
) -> TaskIssue | None:
    if value is None:
        return None
    if isinstance(value, dict):
        code = value.get("code")
        if code is None and legacy_code_builder is not None:
            code = legacy_code_builder(value)
        message = value.get("message")
        if code is None or message is None:
            return None
        normalized_code = str(code).strip()
        normalized_message = str(message).strip()
        if not normalized_code or not normalized_message:
            return None
        if normalized_code not in allowed_codes:
            normalized_code = default_code

        severity = str(value.get("severity", default_severity)).strip().lower() or default_severity
        if severity not in allowed_severities:
            severity = default_severity

        origin = str(value.get("origin", default_origin)).strip()
        return TaskIssue(
            code=normalized_code,
            message=normalized_message,
            severity=severity,
            origin=origin,
            source_fragment=str(value.get("source_fragment", "")).strip(),
            translation_fragment=str(value.get("translation_fragment", "")).strip(),
            suggested_translation=str(value.get("suggested_translation", "")).strip(),
        )

    normalized_message = str(value).strip()
    if not normalized_message:
        return None
    return TaskIssue(
        code=default_code,
        message=normalized_message,
        severity=default_severity,
        origin=default_origin,
    )


def serialize_task_issue(issue: TaskIssue, *, include_origin: bool = False, include_fragments: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "code": issue.code,
        "message": issue.message,
        "severity": issue.severity,
    }
    if include_origin and issue.origin:
        payload["origin"] = issue.origin
    if include_fragments:
        if issue.source_fragment:
            payload["source_fragment"] = issue.source_fragment
        if issue.translation_fragment:
            payload["translation_fragment"] = issue.translation_fragment
        if issue.suggested_translation:
            payload["suggested_translation"] = issue.suggested_translation
    return payload


def dedupe_task_issues(issues: Iterable[TaskIssue]) -> list[TaskIssue]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    unique: list[TaskIssue] = []
    for issue in issues:
        key = (
            issue.origin.lower(),
            issue.code.lower(),
            issue.severity.lower(),
            " ".join(issue.message.split()).lower(),
            " ".join(issue.source_fragment.split()).lower(),
            " ".join(issue.translation_fragment.split()).lower(),
            " ".join(issue.suggested_translation.split()).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    unique.sort(
        key=lambda issue: (
            severity_rank.get(issue.severity, 9),
            issue.code.lower(),
            issue.message.lower(),
        )
    )
    return unique
