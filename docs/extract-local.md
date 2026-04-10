# Local Extract Task

## Purpose

`extract-terms-local` runs deterministic local term discovery without a model API call.

Use it when you want a cheap, repeatable source-side glossary pass before any model validation or translation.

## Supported Inputs

Source discovery input:

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML
- a directory tree containing supported source files

JSON conversion input:

- local extraction JSON produced by this task

## Inputs And Resources

Core inputs:

- source file, source directory tree, or local extraction JSON
- source language
- target language

Optional shared resources:

- vocabulary from `data/locales/<target-lang>/vocab.txt`, a glossary `.tbx`, or a glossary directory bundle

Extraction behavior is driven by data files under:

- `data/extract/common/...`
- `data/extract/en/...`

## Main CLI Shape

Source discovery:

```powershell
python translate_cli.py extract-terms-local source.po
python translate_cli.py extract-terms-local C:\path\to\source-tree
```

JSON to PO handoff:

```powershell
python translate_cli.py extract-terms-local source.prototype-missing-terms.json --to-po
```

Useful options:

- `--mode`
- `--max-length`
- `--vocab`
- `--include-rejected`
- `--also-po`
- `--include-borderline`
- `--out`
- `--to-po`

## Entry Paths

Unified CLI path:

- `translate_cli.main()`
- `translate_cli.build_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `translate_cli.run_extract_terms_local()`
- `tasks.extract_terms_local.run_from_args()`

GUI path:

- `process_gui.build_local_extract_command()`
- subprocess call to `translate_cli.py extract-terms-local ...`
- same unified CLI path as above

Standalone module path:

- `tasks.extract_terms_local.main()`
- `core.task_cli.run_task_main()`
- `tasks.extract_terms_local.configure_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `tasks.extract_terms_local.run_from_args()`

## Execution Flow

Normal extraction mode:

```text
translate_cli / process_gui
        |
        v
tasks.extract_terms_local.run_from_args
        |
        +--> load_messages_for_input
        |     +--> discover_supported_source_files
        |     +--> detect_file_kind
        |     +--> load_entries_for_file
        |     \--> core.term_extraction.collect_source_messages
        +--> resolve_resource_path
        +--> load_vocabulary_pairs
        +--> core.term_extraction.extract_terms_locally
        |     +--> build_source_messages_from_payloads
        |     +--> collect_candidate_evidence
        |     +--> build_vocabulary_exclusion_keys
        |     +--> build_strong_atomic_terms
        |     \--> decide_candidate
        +--> core.term_handoff.build_json_payload
        +--> write JSON
        \--> optional core.term_handoff.convert_json_to_po
```

JSON-to-PO mode:

```text
tasks.extract_terms_local.run_from_args
        |
        \--> core.term_handoff.convert_json_to_po
```

## Detailed Call Order

1. `run_from_args()` first validates flag combinations and output extensions.
2. If `--to-po` is set, the task skips discovery entirely and calls `core.term_handoff.convert_json_to_po()`.
3. Otherwise, `load_messages_for_input()` decides whether the input is a single file or a directory tree.
4. Directory inputs use `discover_supported_source_files()`, which recursively walks the tree and calls `detect_file_kind()` on each candidate.
5. Each discovered file is normalized with `normalize_source_file_label()`, loaded through `load_entries_for_file()`, and projected into `SourceMessage` objects via `core.term_extraction.collect_source_messages()`.
6. `resolve_resource_path()` auto-detects the glossary resource when `--vocab` is omitted, and `load_vocabulary_pairs()` reads the normalized vocabulary pairs.
7. `extract_terms_locally()` runs the deterministic extraction pipeline.
8. Inside `extract_terms_locally()`, the main stages are `build_source_messages_from_payloads()`, `collect_candidate_evidence()`, `build_vocabulary_exclusion_keys()`, `build_strong_atomic_terms()`, and `decide_candidate()`.
9. `collect_candidate_evidence()` itself drills into helpers such as `collect_raw_candidate_evidence()`, `extract_message_candidate_counts()`, `tokenize_source_text()`, `parse_location_note()`, and `build_location_scope()`.
10. Once accepted, borderline, and rejected candidates are classified, `core.term_handoff.build_json_payload()` shapes the JSON report.
11. The task writes the JSON report directly to disk.
12. If `--also-po` is enabled, it immediately calls `core.term_handoff.convert_json_to_po()` on the newly written JSON report to produce a PO handoff beside it.

## Modes

- `all`
  - keep all local candidates
- `missing`
  - drop candidates already covered by the approved vocabulary

## Output

Normal extraction writes JSON:

- `<input>.prototype-glossary.json`
- or `<input>.prototype-missing-terms.json`

Optional one-shot PO handoff:

- `--also-po` writes the matching PO handoff beside the JSON report

JSON to PO conversion:

- `--to-po` converts an existing local extraction JSON into a translation-ready PO glossary file

## What The JSON Contains

The JSON report includes:

- accepted terms
- borderline terms
- rejected terms, when requested
- translation candidates
- extraction metadata such as mode, max length, vocabulary source, and counts

## Important Behavior

- repeated source messages are deduplicated across files
- extraction uses shared logic from `core/term_extraction.py`
- JSON shaping and PO conversion use `core/term_handoff.py`
- `max-length 1` means unigrams only
- `max-length 2` enables bigrams
- `max-length 3` enables bi- and tri-grams

## Relationship To Model Extraction

This task is intentionally separate from `extract-terms`.

Use `extract-terms-local` for:

- deterministic local candidate discovery
- vocabulary-aware filtering in `missing` mode
- JSON and PO handoff preparation

Use `extract-terms` for:

- model-driven term discovery
- model-suggested target-language glossary entries
