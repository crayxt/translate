# Check Task

## Purpose

`check` reviews translated files for QA issues.

Use it when you want structured findings about translation quality, placeholders, tags, terminology, plural handling, or script usage.

## Supported Inputs

- `.po`
- `.ts`

This task currently does not review `.resx`, `.strings`, `.txt`, or Android XML.

## Inputs And Resources

Core inputs:

- translated file
- source language
- target language
- provider and model

Optional shared resources:

- glossary from `data/locales/<target-lang>/glossary.po`, a glossary PO, or a glossary TBX
- rules from `data/locales/<target-lang>/rules.md`

The checker combines:

- model review
- deterministic local checks for placeholders, tags, accelerators, plural slots, and approved glossary usage

## Main CLI Shape

```powershell
python translate_cli.py check translated.po
python translate_cli.py check translated.ts
```

Useful options:

- `--provider`
- `--model`
- `--thinking-level`
- `--flex`
- `--batch-size`
- `--parallel-requests`
- `--probe`
- `--glossary`
- `--rules`
- `--rules-str`
- `--include-ok`
- `--out`

## Entry Paths

Unified CLI path:

- `translate_cli.main()`
- `translate_cli.build_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `translate_cli.run_check()`
- `tasks.check_translations.run_from_args()`

GUI path:

- `process_gui.build_check_command()`
- subprocess call to `translate_cli.py check ...`
- same unified CLI path as above

Standalone module path:

- `tasks.check_translations.main()`
- `core.task_cli.run_task_main()`
- `tasks.check_translations.configure_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `tasks.check_translations.run_from_args()`

## Execution Flow

```text
translate_cli / process_gui
        |
        v
tasks.check_translations.run_from_args
        |
        +--> resolve_provider_model
        +--> detect_file_kind
        +--> load_entries_for_check
        |     \--> load_po / load_ts
        +--> select_review_entries
        |     \--> has_reviewable_translation
        +--> limit_review_entries
        +--> build_task_runtime_context
        |     +--> get_translation_provider
        |     +--> provider.create_client_from_env
        |     \--> load_task_resource_context
        +--> normalize_limits
        +--> build_fixed_batches
        +--> build_check_generation_config
        |     \--> build_check_system_instruction
        |
        \--> asyncio.run(run_checks)
                |
                +--> build_check_request_contents
                |     +--> build_indexed_batch_map
                |     +--> build_check_message_payload
                |     +--> build_check_request_payload
                |     \--> build_task_request_contents
                +--> run_model_batches
                +--> parse_check_response
                |     +--> json_load_maybe
                |     +--> normalize_task_issue
                |     +--> _build_legacy_check_issue_code
                |     \--> dedupe_issues
                \--> serialize_task_issue
        |
        \--> write final JSON report
```

## Detailed Call Order

1. `run_from_args()` resolves the effective model name with `resolve_provider_model()`.
2. `detect_file_kind()` validates that the input is `.po` or `.ts`.
3. `build_task_runtime_context()` creates the provider client and loads optional vocabulary and rules resources.
4. `load_entries_for_check()` loads translated entries through `load_po()` or `load_ts()`.
5. `select_review_entries()` filters down to translated entries by calling `has_reviewable_translation()`.
6. `limit_review_entries()` applies `--probe` through the shared `limit_items()` helper.
7. `normalize_limits()` applies task defaults by delegating to `core.review_flow.normalize_limits()`.
8. `build_fixed_batches()` groups reviewable entries into QA batches.
9. `build_check_generation_config()` builds the provider config, and `build_check_system_instruction()` appends any target-script guidance from `build_target_script_guidance()`.
10. `run_checks()` prepares each batch with `build_check_request_contents()`.
11. `build_check_request_contents()` calls `build_indexed_batch_map()`, `build_check_message_payload()`, `build_check_request_payload()`, and `build_task_request_contents()`.
12. `run_model_batches()` executes the provider calls and feeds each raw response into `parse_check_response()`.
13. `parse_check_response()` normalizes JSON via `json_load_maybe()`, maps legacy issue categories with `_build_legacy_check_issue_code()`, validates each issue through `normalize_task_issue()`, and deduplicates them with `dedupe_issues()`.
14. `on_batch_completed()` converts normalized issues into report rows, serializes them with `serialize_task_issue()`, and accumulates progress counters.
15. When all batches complete, `run_from_args()` writes the final `<input>.translation-check.json` report and prints the issue totals.

## Output

Default report path:

- `<input>.translation-check.json`

The report is JSON only.

## Structured Issue Model

Check findings use structured issues with:

- `code`
- `message`
- `severity`

Current issue namespace:

- `check.meaning`
- `check.grammar`
- `check.tone`
- `check.terminology`
- `check.placeholder`
- `check.tag`
- `check.accelerator`
- `check.plural`
- `check.fluency`
- `check.script`
- `check.other`

Typical severities:

- `error`
- `warning`

## Important Behavior

- only reviewable translated entries are sent for checking
- `--probe` limits how many entries are checked, mainly for testing
- `--include-ok` keeps entries with no findings in the output JSON
- glossary and rules are treated as real QA constraints, not just suggestions

## When Not To Use This Task

- use `translate` to generate translations
- use `revise` to apply a targeted natural-language instruction
- use `extract-terms` or `extract-terms-local` for glossary work
