from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

from core.resources import (
    detect_rules_source,
    load_glossary_pairs,
    merge_project_rules,
    read_optional_text_file,
    read_optional_glossary_file,
    resolve_resource_path,
)


@dataclass(slots=True)
class TaskResourceContext:
    """Resolved glossary and rules resources for a task run."""
    glossary_path: str | None = None
    glossary_text: str | None = None
    glossary_source: str = "none"
    glossary_pairs: List[Tuple[str, str]] = field(default_factory=list)
    rules_path: str | None = None
    rules_text: str | None = None
    project_rules: str | None = None
    rules_source: str | None = None


def load_task_resource_context(
    *,
    target_lang: str,
    explicit_glossary_path: str | None = None,
    explicit_rules_path: str | None = None,
    inline_rules: str | None = None,
    include_glossary: bool = True,
    include_rules: bool = True,
    load_glossary_pairs_flag: bool = False,
) -> TaskResourceContext:
    """Resolve glossary and rules resources, including auto-detected defaults."""
    context = TaskResourceContext()

    if include_glossary:
        context.glossary_path = resolve_resource_path(
            explicit_path=explicit_glossary_path,
            prefix="glossary",
            extension="po",
            target_lang=target_lang,
            allow_directory=True,
        )
        context.glossary_text = read_optional_glossary_file(
            context.glossary_path,
            "Glossary",
            target_lang=target_lang,
        )
        if context.glossary_text and context.glossary_path:
            source_prefix = "dir" if os.path.isdir(context.glossary_path) else "file"
            context.glossary_source = f"{source_prefix}:{context.glossary_path}"
        if load_glossary_pairs_flag and context.glossary_path:
            context.glossary_pairs = load_glossary_pairs(
                context.glossary_path,
                "Glossary",
                target_lang=target_lang,
            )

    if include_rules:
        context.rules_path = resolve_resource_path(
            explicit_path=explicit_rules_path,
            prefix="rules",
            extension="md",
            target_lang=target_lang,
        )
        context.rules_text = read_optional_text_file(context.rules_path, "Rules")
        context.project_rules = merge_project_rules(context.rules_text, inline_rules)
        context.rules_source = detect_rules_source(
            context.rules_path,
            context.rules_text,
            inline_rules,
        )

    return context


__all__ = [
    "TaskResourceContext",
    "load_task_resource_context",
]
