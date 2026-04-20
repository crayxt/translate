# Extract Terms Task

## Purpose

`extract-terms` uses a model to propose glossary candidates from source localization files.

Use it when you want model-driven term discovery and suggested translations, not deterministic local discovery.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML

## Inputs And Resources

Core inputs:

- source file
- source language
- target language
- provider and model

Optional shared resources:

- glossary from `data/locales/<target-lang>/glossary.po`, a glossary PO, or a glossary TBX

In `missing` mode, the glossary is used to suppress already known terms and keep the output focused on missing glossary entries.

## Main CLI Shape

```powershell
python translate_cli.py extract-terms source.po --target-lang kk
python translate_cli.py extract-terms source.xml --target-lang kk
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--glossary`
- `--mode`
- `--out-format`
- `--out`
- `--max-terms-per-batch`

## Entry Paths

Unified CLI path:

- `translate_cli.main()`
- `translate_cli.build_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `translate_cli.run_extract_terms()`
- `tasks.extract_terms.run_from_args()`

GUI path:

- `process_gui.build_extract_command()`
- subprocess call to `translate_cli.py extract-terms ...`
- same unified CLI path as above

Standalone module path:

- `tasks.extract_terms.main()`
- `core.task_cli.run_task_main()`
- `tasks.extract_terms.configure_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `tasks.extract_terms.run_from_args()`

## Execution Flow

```text
translate_cli / process_gui
        |
        v
tasks.extract_terms.run_from_args
        |
        +--> resolve_provider_model
        +--> build_task_runtime_context
        |     \--> load_task_resource_context
        +--> detect_file_kind
        +--> load_entries_for_file
        |     \--> load_po / load_ts / load_resx / load_strings / load_txt / load_android_xml
        +--> collect_source_messages
        |     \--> core.term_extraction.collect_source_messages
        +--> normalize_limits
        +--> build_fixed_batches
        +--> build_term_output_path
        +--> build_term_generation_config
        |     \--> build_term_system_instruction
        |
        \--> asyncio.run(run_extraction)
                |
                +--> build_term_request_contents
                |     +--> build_indexed_batch_map
                |     +--> build_term_request_payload
                |     +--> build_term_request_spec
                |     \--> build_task_request_contents
                +--> run_model_batches
                +--> parse_term_response
                |     \--> _json_load_maybe
                \--> collect raw candidates
        |
        +--> merge_term_candidates
        +--> sort candidates
        \--> save_terms_as_po or write JSON
```

## Detailed Call Order

1. `run_from_args()` resolves the effective model name and validates `--max-terms-per-batch` and `--max-attempts`.
2. `build_task_runtime_context()` creates the provider client and loads glossary resources. This task sets `include_rules=False`.
3. `detect_file_kind()` validates the input format, and `load_entries_for_file()` dispatches to `load_po()`, `load_ts()`, `load_resx()`, `load_strings()`, `load_txt()`, or `load_android_xml()`.
4. `collect_source_messages()` projects those entries through `core.term_extraction.collect_source_messages()` so extraction batches use contextual source messages instead of raw strings only.
5. `normalize_limits()` chooses task-default batching when the user omits both runtime knobs.
6. `build_fixed_batches()` groups source messages into extraction batches.
7. `build_term_output_path()` chooses the default output filename based on `--mode` and `--out-format`.
8. `build_term_generation_config()` creates the provider config and injects target-script guidance through `build_term_system_instruction()`.
9. `run_extraction()` prepares each batch with `build_term_request_contents()`.
10. `build_term_request_contents()` calls `build_indexed_batch_map()`, `build_term_request_payload()`, `build_term_request_spec()`, and `build_task_request_contents()`.
11. `run_model_batches()` executes the provider calls and parses each response with `parse_term_response()`.
12. `parse_term_response()` accepts structured payloads directly or falls back through `_json_load_maybe()` for plain-text JSON responses.
13. After all batches complete, `merge_term_candidates()` deduplicates source terms case-insensitively and preserves the best-known fields.
14. The final output path is written either through `save_terms_as_po()` or by dumping the JSON payload with metadata.

## Modes

- `all`
  - broader glossary building from source content
- `missing`
  - focus on terms not already present in the supplied glossary

## Output

Supported output formats:

- `po`
- `json`

Default paths:

- `all` + `po` -> `<input>.glossary.po`
- `missing` + `po` -> `<input>.missing-terms.po`
- `missing` + `json` -> `<input>.missing-terms.json`

## Important Behavior

- the task now uses the shared source-message projection path from `core/term_extraction.py`
- contextual source messages are sent instead of only bare strings
- the model is instructed to prefer atomic reusable terms over phrase-shaped message fragments
- term candidates are merged case-insensitively before final output

If `missing` mode is used with `--out-format po`, known glossary entries are merged into the generated PO before new missing terms are appended.

## Relationship To Local Discovery

This task is the model-driven extractor.

If you want:

- deterministic local filtering
- accepted / borderline / rejected classification
- no API usage

use `extract-terms-local` instead.
