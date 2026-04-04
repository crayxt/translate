#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List

from core.formats import FileKind, detect_file_kind, load_android_xml, load_po, load_resx, load_strings, load_ts, load_txt
from core.resources import load_vocabulary_pairs, resolve_resource_path
from core.task_cli import add_language_arguments, add_vocabulary_argument, build_task_parser, run_task_main
from core.term_extraction import collect_source_messages, extract_terms_locally
from core.term_handoff import build_json_payload, build_output_path, convert_json_to_po


def load_entries_for_file(file_path: str, file_kind: FileKind) -> List[Any]:
    """Load localization entries for any supported local-discovery source file kind."""
    if file_kind == FileKind.ANDROID_XML:
        entries, _, _ = load_android_xml(file_path)
        return entries
    if file_kind == FileKind.TS:
        entries, _, _ = load_ts(file_path)
        return entries
    if file_kind == FileKind.RESX:
        entries, _, _ = load_resx(file_path)
        return entries
    if file_kind == FileKind.STRINGS:
        entries, _, _ = load_strings(file_path)
        return entries
    if file_kind == FileKind.TXT:
        entries, _, _ = load_txt(file_path)
        return entries
    entries, _, _ = load_po(file_path)
    return entries


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure the CLI for local term discovery and JSON-to-PO conversion."""
    parser.description = (
        "Run local glossary-term discovery or convert local extraction JSON into a PO handoff file."
    )
    parser.add_argument("file", help="Input source .po/.ts/.resx/.strings/.txt/Android .xml file or local extraction JSON file")
    add_language_arguments(parser)
    add_vocabulary_argument(parser)
    parser.add_argument(
        "--to-po",
        action="store_true",
        help="Treat the input as local extraction JSON and convert it to a PO glossary file.",
    )
    parser.add_argument(
        "--also-po",
        action="store_true",
        help="After writing the local extraction JSON report, also write the matching PO handoff.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "missing"],
        default="missing",
        help="all: keep all local candidates; missing: drop terms already in vocabulary",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "Maximum candidate length: 1 keeps single words only, "
            "2 enables bigrams, 3 enables bi- and tri-grams."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <input>.prototype-*.json or matching .po handoff path)",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected candidates in the JSON output for debugging.",
    )
    parser.add_argument(
        "--include-borderline",
        action="store_true",
        help="When writing a PO handoff, also convert borderline candidates into PO entries.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for local term discovery."""
    return build_task_parser(configure_parser)


def run_from_args(args: argparse.Namespace) -> None:
    """Execute local term discovery or JSON-to-PO conversion from parsed CLI args."""
    if args.to_po and args.also_po:
        sys.exit("ERROR: --to-po and --also-po cannot be used together.")

    if not args.to_po and args.out and not str(args.out).lower().endswith(".json"):
        sys.exit("ERROR: Local extraction output path should end with .json.")

    if args.to_po and args.out and not str(args.out).lower().endswith(".po"):
        sys.exit("ERROR: JSON to PO output path should end with .po.")

    if args.to_po:
        try:
            out_path = convert_json_to_po(
                args.file,
                out_path=args.out,
                include_borderline=args.include_borderline,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"ERROR: {exc}")

        print("Local term PO conversion complete.")
        print(f"Saved file: {out_path}")
        return

    try:
        file_kind = detect_file_kind(args.file)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    vocabulary_path = resolve_resource_path(
        explicit_path=args.vocab,
        prefix="vocab",
        extension="txt",
        target_lang=args.target_lang,
    )
    vocabulary_pairs = load_vocabulary_pairs(vocabulary_path, "Vocabulary")

    entries = load_entries_for_file(args.file, file_kind)
    messages = collect_source_messages(entries)
    if not messages:
        print("No source messages found for local terminology extraction.")
        return

    out_path = args.out or build_output_path(args.file, args.mode)
    result = extract_terms_locally(
        messages,
        mode=args.mode,
        vocabulary_pairs=vocabulary_pairs,
        max_length=args.max_length,
    )
    payload = build_json_payload(
        file_path=args.file,
        out_path=out_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        mode=args.mode,
        max_length=args.max_length,
        vocabulary_path=vocabulary_path,
        total_source_messages=len(messages),
        result=result,
        include_rejected=args.include_rejected,
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("Local term extraction complete.")
    print(f"Saved file: {out_path}")
    if args.also_po:
        try:
            po_out_path = convert_json_to_po(
                out_path,
                include_borderline=args.include_borderline,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"ERROR: JSON report was written, but PO conversion failed: {exc}")
        print(f"Saved PO handoff: {po_out_path}")
    print(f"Accepted terms: {len(result.accepted_terms)}")
    print(f"Borderline terms: {len(result.borderline_terms)}")
    print(f"Rejected terms: {len(result.rejected_terms)}")


def main(argv: list[str] | None = None) -> None:
    """Run the local term discovery CLI."""
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
