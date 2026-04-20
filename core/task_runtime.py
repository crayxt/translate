from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.providers import TranslationProvider, get_translation_provider, validate_provider_seed
from core.task_resources import TaskResourceContext, load_task_resource_context


@dataclass(slots=True)
class TaskRuntimeContext:
    """Provider client plus resolved task resources for one command run."""
    provider: TranslationProvider
    client: Any
    resources: TaskResourceContext


def build_task_runtime_context(
    *,
    provider_name: str | None,
    target_lang: str,
    flex_mode: bool = False,
    seed: int | None = None,
    explicit_vocab_path: str | None = None,
    explicit_rules_path: str | None = None,
    inline_rules: str | None = None,
    include_vocab: bool = True,
    include_rules: bool = True,
    load_vocab_pairs_flag: bool = False,
    get_translation_provider_fn: Callable[[str | None], TranslationProvider] = get_translation_provider,
    load_task_resource_context_fn: Callable[..., TaskResourceContext] = load_task_resource_context,
) -> TaskRuntimeContext:
    """Create the provider client and load any task-level resource files."""
    provider = get_translation_provider_fn(provider_name)
    validate_provider_seed(provider, seed)
    client = provider.create_client_from_env(flex_mode=flex_mode)
    resources = load_task_resource_context_fn(
        target_lang=target_lang,
        explicit_vocab_path=explicit_vocab_path,
        explicit_rules_path=explicit_rules_path,
        inline_rules=inline_rules,
        include_vocab=include_vocab,
        include_rules=include_rules,
        load_vocab_pairs_flag=load_vocab_pairs_flag,
    )
    return TaskRuntimeContext(
        provider=provider,
        client=client,
        resources=resources,
    )


def print_startup_configuration(*entries: tuple[str, Any]) -> None:
    """Print a stable label/value summary for task startup diagnostics."""
    print("Startup configuration:")
    for label, value in entries:
        print(f"  {label}: {value}")


__all__ = [
    "TaskRuntimeContext",
    "build_task_runtime_context",
    "print_startup_configuration",
]
