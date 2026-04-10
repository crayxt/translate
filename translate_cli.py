#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from core.task_cli import apply_provider_environment_from_args
from tasks import check_translations, extract_terms, extract_terms_local, revise_translations, translate


def build_parser() -> argparse.ArgumentParser:
    """Build the unified top-level parser and register task subcommands."""
    parser = argparse.ArgumentParser(
        description="Unified CLI for translation, term extraction, QA, and revision."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate_parser = translate.configure_parser(
        subparsers.add_parser("translate", help="Translate localization files")
    )
    translate_parser.set_defaults(handler=run_translate)

    extract_parser = extract_terms.configure_parser(
        subparsers.add_parser("extract-terms", help="Extract glossary terms")
    )
    extract_parser.set_defaults(handler=run_extract_terms)

    extract_local_parser = extract_terms_local.configure_parser(
        subparsers.add_parser("extract-terms-local", help="Run local term discovery and PO handoff conversion")
    )
    extract_local_parser.set_defaults(handler=run_extract_terms_local)

    check_parser = check_translations.configure_parser(
        subparsers.add_parser("check", help="Check translated files")
    )
    check_parser.set_defaults(handler=run_check)

    revise_parser = revise_translations.configure_parser(
        subparsers.add_parser("revise", help="Revise translated files")
    )
    revise_parser.set_defaults(handler=run_revise)

    return parser


def run_translate(args: argparse.Namespace) -> None:
    """Dispatch the `translate` subcommand."""
    translate.run_from_args(args)


def run_extract_terms(args: argparse.Namespace) -> None:
    """Dispatch the `extract-terms` subcommand."""
    extract_terms.run_from_args(args)


def run_extract_terms_local(args: argparse.Namespace) -> None:
    """Dispatch the `extract-terms-local` subcommand."""
    extract_terms_local.run_from_args(args)


def run_check(args: argparse.Namespace) -> None:
    """Dispatch the `check` subcommand."""
    check_translations.run_from_args(args)


def run_revise(args: argparse.Namespace) -> None:
    """Dispatch the `revise` subcommand."""
    revise_translations.run_from_args(args)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, apply shared env overrides, and run the handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_provider_environment_from_args(args)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
