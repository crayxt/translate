# Translate Task

## Purpose

`translate` fills unfinished localization entries, or retranslates existing ones when explicitly requested.

Use it when you want normal translation output, not QA findings, revision-only changes, or glossary extraction.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML
- a directory tree containing supported files

Multi-file translation is supported when all resolved inputs use the same format.

Android XML is a special case:

- it currently supports one translated target file at a time
- it requires `--source-file` because the translated XML does not retain the original source text

## Inputs And Resources

Core inputs:

- source file or files
- source language
- target language
- provider and model

Optional shared resources:

- glossary from `data/locales/<target-lang>/glossary.po`, a glossary PO, or a glossary TBX
- rules from `data/locales/<target-lang>/rules.md`

At runtime, the task builds message-scoped glossary hints:

- the full approved glossary is parsed into rich entries
- only the relevant subset for each message is attached as `relevant_vocabulary`

For plural messages, the task sends structured plural data:

- `source_singular`
- `source_plural`
- `plural_forms`
- `plural_slots`

## Main CLI Shape

```powershell
python translate_cli.py translate file.po
python translate_cli.py translate first.po second.po third.po
python translate_cli.py translate translated.xml --source-file source.xml
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--glossary`
- `--rules`
- `--rules-str`
- `--retranslate-all`
- `--warnings-report`

## Entry Paths

Unified CLI path:

- `translate_cli.main()`
- `translate_cli.build_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `translate_cli.run_translate()`
- `tasks.translate.run_from_args()`

GUI path:

- `process_gui.build_process_command()`
- subprocess call to `translate_cli.py translate ...`
- same unified CLI path as above

Standalone module path:

- `tasks.translate.main()`
- `core.task_cli.run_task_main()`
- `tasks.translate.configure_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `tasks.translate.run_from_args()`

## Execution Flow

```text
translate_cli / process_gui
        |
        v
tasks.translate.run_from_args
        |
        v
config_from_args
        |
        v
run_translation
        |
        +--> resolve_translation_input_paths
        +--> validate_translation_files
        +--> load_translation_jobs
                |
                +--> load_entries_for_translation
                        |
                        +--> detect_file_kind
                        +--> load_po / load_ts / load_resx / load_strings / load_txt
                        \--> load_paired_android_xml
        |
        +--> build_task_runtime_context
                |
                +--> get_translation_provider
                +--> provider.create_client_from_env
                \--> load_task_resource_context
                        |
                        +--> resolve_resource_path
                        +--> read_optional_vocabulary_file
                        +--> read_optional_text_file
                        +--> merge_project_rules
                        \--> detect_rules_source
        |
        +--> build_translation_queue
        +--> resolve_runtime_limits
        +--> build_translation_generation_config
        +--> build_scoped_vocabulary_entries
        +--> build_batches
        |
        \--> asyncio.run(run_translation_batches)
                |
                +--> build_translation_message_payload
                |     \--> build_relevant_vocabulary
                +--> build_translation_request_contents
                |     +--> build_translation_request_spec
                |     +--> build_translation_request_payload
                |     \--> build_task_request_contents
                +--> provider.generate_with_retry
                +--> parse_response
                +--> optional retry for missing items
                +--> apply_translation_to_entry
                +--> build_translation_warning_item
                \--> save_callback per touched job
        |
        +--> final save_callback pass
        \--> optional write_translation_warning_report
```

## Detailed Call Order

1. `run_from_args()` converts parsed args into `TranslationRunConfig` with `config_from_args()`.
2. `run_translation()` expands file and directory inputs with `resolve_translation_input_paths()`.
3. `validate_translation_files()` checks duplicate inputs, output-path collisions, mixed formats, and Android `--source-file` rules.
4. `load_translation_jobs()` loads each file through `load_entries_for_translation()`.
5. `load_entries_for_translation()` dispatches to the format adapter:
   `load_po()`, `load_ts()`, `load_resx()`, `load_strings()`, `load_txt()`, or `load_paired_android_xml()`.
6. `build_task_runtime_context()` creates the provider client and loads shared resources.
7. `load_task_resource_context()` resolves glossary and rules files through `resolve_resource_path()`, `read_optional_vocabulary_file()`, `read_optional_text_file()`, `merge_project_rules()`, and `detect_rules_source()`.
8. `build_translation_queue()` flattens loaded files into entry-level work items using `select_work_items()`.
9. `resolve_runtime_limits()` chooses batch size and concurrency, then `build_translation_generation_config()` builds the provider config.
10. `build_scoped_vocabulary_entries()` precomputes glossary matchers for message-scoped hints.
11. `build_batches()` either splits evenly in small-file mode or uses `build_fixed_batches()`.
12. `run_translation_batches()` drives the async work:
    `build_translation_message_payload()` shapes each message, `build_relevant_vocabulary()` attaches glossary hints, `build_translation_request_contents()` builds provider input, `provider.generate_with_retry()` performs the API call, and `parse_response()` normalizes model output.
13. If the model omits items, `run_translation_batches()` rebuilds a smaller request with `force_non_empty=True` and retries just the missing subset.
14. `on_batch_completed()` applies translations with `apply_translation_to_entry()`, records structured warning items with `build_translation_warning_item()`, and writes touched outputs through each job's `save_callback`.
15. After the async phase, `run_translation()` writes any remaining touched outputs again, optionally emits `write_translation_warning_report()`, and prints the final saved output paths.

## Output

Default translated output:

- `<input>.ai-translated.<ext>`

Optional warning sidecar:

- `<input>.ai-translated.translation-warnings.json`

The warning sidecar contains only messages where the model reported a review-worthy concern.

## Structured Warning Reporting

When `--warnings-report` is enabled, the task can return per-message structured issues.

Current warning namespace:

- `translate.ambiguous_term`
- `translate.unclear_source_meaning`
- `translate.glossary_variant_choice`
- `translate.possible_untranslated_token`
- `translate.placeholder_attention`
- `translate.length_or_ui_fit_risk`

Each warning item uses:

- `code`
- `message`
- `severity`

Translation warnings use:

- `warning` for real ambiguity or risk
- `info` for notable but non-risk notes

## Important Behavior

- by default, only unfinished or fuzzy entries are translated
- `--retranslate-all` forces all translatable entries back through the model
- recursive directory translation skips generated toolkit artifacts such as `*.ai-translated.*`, `*.glossary.po`, `*.missing-terms.po`, and `*.prototype-*.po`
- when the scan root is this toolkit repository itself, recursive translation also skips toolkit-owned directories such as `data/`, `logs/`, `docs/`, `tests/`, `tasks/`, and `core/`
- placeholders and protected tokens are preserved
- message `context` and `note` are sent to the model when available
- `relevant_vocabulary` can include multiple variants for the same source term, and the model is instructed to choose by context and part of speech

## When Not To Use This Task

Use another task when the goal is different:

- `check` for QA findings
- `revise` for targeted edits to an already translated file
- `extract-terms` for model glossary discovery
- `extract-terms-local` for deterministic local glossary discovery without an API call
