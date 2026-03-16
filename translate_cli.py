#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for translation, term extraction, QA, and revision."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("translate", "extract-terms", "check", "revise"):
        subparsers.add_parser(name)

    return parser


def run_translate(argv: list[str]) -> None:
    from tasks import translate

    translate.main(argv)


def run_extract_terms(argv: list[str]) -> None:
    from tasks import extract_terms

    extract_terms.main(argv)


def run_check(argv: list[str]) -> None:
    from tasks import check_translations

    check_translations.main(argv)


def run_revise(argv: list[str]) -> None:
    from tasks import revise_translations

    revise_translations.main(argv)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command == "translate":
        run_translate(remaining)
        return
    if args.command == "extract-terms":
        run_extract_terms(remaining)
        return
    if args.command == "check":
        run_check(remaining)
        return
    if args.command == "revise":
        run_revise(remaining)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
