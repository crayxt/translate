#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.glossary_catalog import (
    build_glossary_pot,
    load_glossary_source_terms,
    sync_locale_glossary_po,
    write_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate glossary POT and sync locale PO catalogs from a canonical JSONL source.",
    )
    parser.add_argument(
        "--source",
        default="data/glossary/glossary.jsonl",
        help="Canonical JSONL glossary source file.",
    )
    parser.add_argument(
        "--pot-out",
        default="data/glossary/glossary.pot",
        help="Output POT path.",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Optional locale code for syncing one locale PO catalog.",
    )
    parser.add_argument(
        "--po-out",
        default=None,
        help="Output PO path for --locale. Defaults to data/locales/<locale>/glossary.po.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    terms = load_glossary_source_terms(args.source)
    write_catalog(build_glossary_pot(terms), args.pot_out)

    if args.locale:
        po_out = args.po_out or str(Path("data") / "locales" / args.locale / "glossary.po")
        existing_catalog = polib.pofile(po_out, wrapwidth=78) if Path(po_out).exists() else None
        write_catalog(
            sync_locale_glossary_po(terms, locale=args.locale, existing_catalog=existing_catalog),
            po_out,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
