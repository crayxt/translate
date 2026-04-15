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


def render_task_instruction(
    *,
    task_spec: TaskRequestSpec,
    payload_source_line: str,
) -> str:
    """Render task instructions for one provider-specific payload transport."""
    lines = [task_spec.task_intro.strip(), ""]
    lines.extend(_render_bullet_section("Task requirements", task_spec.task_lines))
    input_lines = [payload_source_line]
    input_lines.extend(task_spec.payload_lines)
    lines.extend(_render_bullet_section("Input contract", input_lines))
    lines.extend(_render_bullet_section("Output requirements", task_spec.output_lines))
    return _join_rendered_lines(lines)


def render_structured_task_instruction(
    *,
    task_spec: TaskRequestSpec,
    function_name: str,
) -> str:
    """Render the instruction text for providers that support structured inputs."""
    return render_task_instruction(
        task_spec=task_spec,
        payload_source_line=(
            f"Read the batch payload from the structured function response named `{function_name}`."
        ),
    )


def render_text_fallback_prompt(
    *,
    task_spec: TaskRequestSpec,
    payload: dict[str, Any],
) -> str:
    """Render a plain-text prompt with the JSON payload embedded inline."""
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = [
        render_task_instruction(
            task_spec=task_spec,
            payload_source_line="Read the batch payload from the JSON object below.",
        ),
        "",
    ]
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
        supports_structured_input = bool(getattr(provider, "supports_structured_input", False))
        return builder(
            task_instruction=render_task_instruction(
                task_spec=task_spec,
                payload_source_line=(
                    f"Read the batch payload from the structured function response named `{function_name}`."
                    if supports_structured_input
                    else "Read the batch payload from the user input JSON."
                ),
            ),
            function_name=function_name,
            payload=payload,
            fallback_prompt=render_text_fallback_prompt(task_spec=task_spec, payload=payload),
        )
    return render_text_fallback_prompt(task_spec=task_spec, payload=payload)


__all__ = [
    "TaskRequestSpec",
    "build_task_request_contents",
    "render_task_instruction",
    "render_structured_task_instruction",
    "render_text_fallback_prompt",
]
