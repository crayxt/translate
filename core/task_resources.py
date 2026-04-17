from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

from core.resources import (
    detect_rules_source,
    load_vocabulary_pairs,
    merge_project_rules,
    read_optional_text_file,
    read_optional_vocabulary_file,
    resolve_resource_path,
)


@dataclass(slots=True)
class TaskResourceContext:
    """Resolved glossary and rules resources for a task run."""
    vocabulary_path: str | None = None
    vocabulary_text: str | None = None
    vocabulary_source: str = "none"
    vocabulary_pairs: List[Tuple[str, str]] = field(default_factory=list)
    rules_path: str | None = None
    rules_text: str | None = None
    project_rules: str | None = None
    rules_source: str | None = None


def load_task_resource_context(
    *,
    target_lang: str,
    explicit_vocab_path: str | None = None,
    explicit_rules_path: str | None = None,
    inline_rules: str | None = None,
    include_vocab: bool = True,
    include_rules: bool = True,
    load_vocab_pairs_flag: bool = False,
) -> TaskResourceContext:
    """Resolve glossary and rules resources, including auto-detected defaults."""
    context = TaskResourceContext()

    if include_vocab:
        context.vocabulary_path = resolve_resource_path(
            explicit_path=explicit_vocab_path,
            prefix="vocab",
            extension="txt",
            target_lang=target_lang,
            allow_directory=True,
        )
        context.vocabulary_text = read_optional_vocabulary_file(
            context.vocabulary_path,
            "Glossary",
            target_lang=target_lang,
        )
        if context.vocabulary_text and context.vocabulary_path:
            source_prefix = "dir" if os.path.isdir(context.vocabulary_path) else "file"
            context.vocabulary_source = f"{source_prefix}:{context.vocabulary_path}"
        if load_vocab_pairs_flag and context.vocabulary_path:
            context.vocabulary_pairs = load_vocabulary_pairs(
                context.vocabulary_path,
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
