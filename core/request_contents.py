from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskRequestSpec:
    """Structured description of a task prompt and its payload/output contract."""
    task_intro: str
    task_lines: tuple[str, ...] = ()
    payload_lines: tuple[str, ...] = ()
    output_lines: tuple[str, ...] = ()
    text_payload_label: str = "Input payload (JSON):"


def _render_bullet_section(title: str, lines: Iterable[str]) -> list[str]:
    """Render a titled bullet block, omitting empty sections."""
    materialized = [str(line).strip() for line in lines if str(line).strip()]
    if not materialized:
        return []
    rendered = [f"{title}:"]
    rendered.extend(f"- {line}" for line in materialized)
    rendered.append("")
    return rendered


def _join_rendered_lines(lines: list[str]) -> str:
    """Join prompt lines while trimming trailing blank lines."""
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def render_structured_task_instruction(
    *,
    task_spec: TaskRequestSpec,
    function_name: str,
) -> str:
    """Render the instruction text for providers that support structured inputs."""
    lines = [task_spec.task_intro.strip(), ""]
    lines.extend(_render_bullet_section("Task requirements", task_spec.task_lines))
    input_lines = [f"Read the batch payload from the structured function response named `{function_name}`."]
    input_lines.extend(task_spec.payload_lines)
    lines.extend(_render_bullet_section("Input contract", input_lines))
    lines.extend(_render_bullet_section("Output requirements", task_spec.output_lines))
    return _join_rendered_lines(lines)


def render_text_fallback_prompt(
    *,
    task_spec: TaskRequestSpec,
    payload: dict[str, Any],
) -> str:
    """Render a plain-text prompt with the JSON payload embedded inline."""
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = [task_spec.task_intro.strip(), ""]
    lines.extend(_render_bullet_section("Task requirements", task_spec.task_lines))
    input_lines = ["Read the batch payload from the JSON object below."]
    input_lines.extend(task_spec.payload_lines)
    lines.extend(_render_bullet_section("Input contract", input_lines))
    lines.extend(_render_bullet_section("Output requirements", task_spec.output_lines))
    lines.append(task_spec.text_payload_label)
    lines.append(payload_json)
    return _join_rendered_lines(lines)


def build_task_request_contents(
    *,
    provider: object,
    task_spec: TaskRequestSpec,
    function_name: str,
    payload: dict[str, Any],
) -> Any:
    """Build request contents using provider-native structured input when available."""
    builder = getattr(provider, "build_request_contents", None)
    if callable(builder):
        return builder(
            task_instruction=render_structured_task_instruction(
                task_spec=task_spec,
                function_name=function_name,
            ),
            function_name=function_name,
            payload=payload,
            fallback_prompt=render_text_fallback_prompt(task_spec=task_spec, payload=payload),
        )
    return render_text_fallback_prompt(task_spec=task_spec, payload=payload)


__all__ = [
    "TaskRequestSpec",
    "build_task_request_contents",
    "render_structured_task_instruction",
    "render_text_fallback_prompt",
]
