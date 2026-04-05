from __future__ import annotations

import argparse
import os
from typing import Callable

from core.providers import get_translation_provider
from core.runtime import add_thinking_level_argument


GEMINI_BACKEND_CHOICES = ("studio", "vertex")


def add_language_arguments(
    parser: argparse.ArgumentParser,
    *,
    source_default: str = "en",
    target_default: str = "kk",
) -> None:
    parser.add_argument("--source-lang", default=source_default, help=f"Default: {source_default}")
    parser.add_argument("--target-lang", default=target_default, help=f"Default: {target_default}")


def add_provider_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_provider_name: str,
    default_model: str,
    include_thinking: bool = True,
    include_flex: bool = True,
) -> None:
    parser.add_argument(
        "--provider",
        default=default_provider_name,
        help=f"Model provider (default: {default_provider_name})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name (default: provider default; {default_model} for {default_provider_name})",
    )
    if include_thinking:
        add_thinking_level_argument(parser)
    if include_flex:
        parser.add_argument(
            "--flex",
            dest="flex_mode",
            action="store_true",
            help="Use provider flex mode when supported",
        )
    parser.add_argument(
        "--gemini-backend",
        choices=GEMINI_BACKEND_CHOICES,
        default=None,
        help="Gemini backend override: studio or vertex",
    )
    parser.add_argument(
        "--google-cloud-location",
        default=None,
        help="Google Cloud location for Gemini Vertex AI mode (default: global)",
    )


def resolve_provider_model(
    provider_name: str | None,
    model_name: str | None,
    *,
    get_translation_provider_fn: Callable[[str | None], object] = get_translation_provider,
) -> str:
    cleaned_model = str(model_name or "").strip()
    if cleaned_model:
        return cleaned_model
    provider = get_translation_provider_fn(provider_name)
    return str(getattr(provider, "default_model", "")).strip()


def add_runtime_limit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent requests (auto if omitted)")


def add_vocabulary_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vocab",
        default=None,
        help=(
            "Optional vocabulary file or directory "
            "(auto: data/locales/<target-lang>/vocab.txt or data/locales/<target-lang>/vocab). "
            "Supports .txt and glossary .po"
        ),
    )


def add_rules_arguments(
    parser: argparse.ArgumentParser,
    *,
    rules_help: str,
    rules_str_help: str,
) -> None:
    parser.add_argument(
        "--rules",
        default=None,
        help=rules_help,
    )
    parser.add_argument("--rules-str", default=None, help=rules_str_help)


def add_probe_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--probe",
        "--num-messages",
        dest="num_messages",
        type=int,
        default=None,
        help=help_text,
    )


def add_max_attempts_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = 5,
    help_text: str = "Retry attempts per batch",
) -> None:
    parser.add_argument("--max-attempts", type=int, default=default, help=help_text)


def build_task_parser(
    configure_parser_fn: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    return configure_parser_fn(argparse.ArgumentParser())


def run_task_main(
    *,
    configure_parser_fn: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
    run_from_args_fn: Callable[[argparse.Namespace], None],
    argv: list[str] | None = None,
) -> None:
    parser = build_task_parser(configure_parser_fn)
    args = parser.parse_args(argv)
    apply_provider_environment_from_args(args)
    run_from_args_fn(args)


def apply_provider_environment_from_args(
    args: argparse.Namespace,
    environ: dict[str, str] | None = None,
) -> None:
    env = environ if environ is not None else os.environ
    provider_name = str(getattr(args, "provider", "") or "").strip().lower()
    if provider_name != "gemini":
        return

    backend = str(getattr(args, "gemini_backend", "") or "").strip().lower()
    if backend:
        env["GOOGLE_GENAI_USE_VERTEXAI"] = "true" if backend == "vertex" else "false"

    if backend == "vertex":
        location = str(getattr(args, "google_cloud_location", "") or "").strip()
        if location:
            env["GOOGLE_CLOUD_LOCATION"] = location


__all__ = [
    "GEMINI_BACKEND_CHOICES",
    "add_language_arguments",
    "add_max_attempts_argument",
    "add_probe_argument",
    "add_provider_arguments",
    "apply_provider_environment_from_args",
    "add_rules_arguments",
    "add_runtime_limit_arguments",
    "add_vocabulary_argument",
    "build_task_parser",
    "resolve_provider_model",
    "run_task_main",
]
