# Revise Task

## Purpose

`revise` applies a targeted natural-language instruction to an already translated file.

Use it when you want selective corrections or terminology changes, not a full retranslation and not a QA-only report.

## Supported Inputs

- `.po`
- `.ts`
- `.resx`
- `.strings`
- `.txt`
- Android `<resources>` XML

Some formats require a paired source file because the translated file does not retain the original source text.

## Source-File Requirements

`--source-file` is:

- optional for `.po` and `.ts`
- required for Android `.xml`
- required for `.resx`
- required for `.strings`
- required for `.txt`

The source file must match the translated file type.

## Inputs And Resources

Core inputs:

- translated file
- natural-language revision instruction
- optional paired source file
- source language
- target language
- provider and model

Optional shared resources:

- glossary from `data/locales/<target-lang>/glossary.po`, a glossary PO, or a glossary TBX
- rules from `data/locales/<target-lang>/rules.md`

## Main CLI Shape

```powershell
python translate_cli.py revise translated.po --instruction "Use a shorter term for Preferences"
python translate_cli.py revise translated.ts --instruction "Replace archive with package where the source says package"
python translate_cli.py revise translated.ai-translated.xml --source-file source.xml --instruction "Use natural confirmation questions"
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
- `--source-file`
- `--out`
- `--in-place`
- `--dry-run`

## Entry Paths

Unified CLI path:

- `translate_cli.main()`
- `translate_cli.build_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `translate_cli.run_revise()`
- `tasks.revise_translations.run_from_args()`

GUI path:

- `process_gui.build_revise_command()`
- subprocess call to `translate_cli.py revise ...`
- same unified CLI path as above

Standalone module path:

- `tasks.revise_translations.main()`
- `core.task_cli.run_task_main()`
- `tasks.revise_translations.configure_parser()`
- `core.task_cli.apply_provider_environment_from_args()`
- `tasks.revise_translations.run_from_args()`

## Execution Flow

```text
translate_cli / process_gui
        |
        v
tasks.revise_translations.run_from_args
        |
        +--> configure_stdio
        +--> resolve_provider_model
        +--> load_review_bundle
        |     +--> build_single_file_bundle
        |     +--> build_paired_bundle
        |     +--> load_paired_txt_bundle
        |     \--> load_paired_android_xml
        +--> limit_review_items
        +--> normalize_limits
        +--> build_task_runtime_context
        +--> build_revision_generation_config
        +--> build_final_output_path
        +--> build_fixed_batches
        |
        \--> asyncio.run(run_revision)
                |
                +--> build_revision_request_contents
                |     +--> build_indexed_batch_map
                |     +--> build_review_message_payload
                |     +--> build_revision_request_payload
                |     \--> build_task_request_contents
                +--> run_model_batches
                +--> parse_revision_response
                |     +--> json_load_maybe
                |     +--> _normalize_revision_payload
                |     \--> normalize_task_issue
                +--> retry missing ids with build_retry_contents
                \--> apply_revision_to_item
                        \--> build_candidate_plural_forms
        |
        +--> print_change_samples / print_issue_samples
        +--> save_callback
        \--> move_output_file
```

## Detailed Call Order

1. `run_from_args()` normalizes console output with `configure_stdio()` and resolves the effective model name.
2. `load_review_bundle()` picks the right bundle builder for the file format.
3. Embedded-source formats use `build_single_file_bundle()`, paired formats use `build_paired_bundle()` or `load_paired_txt_bundle()`, and Android XML uses `load_paired_android_xml()`.
4. Bundle construction creates `ReviewItem` objects through `build_review_item()`, which calls `get_entry_prompt_context_and_note()`, `get_plural_texts()`, `get_plural_form_count()`, and `build_entry_source_text()`.
5. `limit_review_items()` applies `--probe`, and `normalize_limits()` resolves batch size and concurrency.
6. `build_task_runtime_context()` creates the provider client and loads optional vocabulary and rules resources.
7. `build_revision_generation_config()` creates the provider config, using `build_revision_system_instruction()` and `build_target_script_guidance()` when needed.
8. `build_final_output_path()` decides whether the run writes a new file, overwrites in place, or stays dry-run only.
9. `build_fixed_batches()` groups review items for API calls.
10. `run_revision()` prepares each batch with `build_revision_request_contents()`, which calls `build_indexed_batch_map()`, `build_review_message_payload()`, `build_revision_request_payload()`, and `build_task_request_contents()`.
11. `run_model_batches()` performs the API calls, retries missing item ids through `build_retry_contents()`, and parses responses with `parse_revision_response()`.
12. `parse_revision_response()` funnels through `_normalize_revision_payload()` and `normalize_task_issue()` so revision actions and issues are typed consistently.
13. `on_batch_completed()` applies accepted updates with `apply_revision_to_item()`. For plurals, that path also uses `build_candidate_plural_forms()`.
14. After all batches, `run_from_args()` prints summary samples with `print_change_samples()` and `print_issue_samples()`.
15. If `--dry-run` is not set, the task writes the generated output via the bundle `save_callback()` and then finalizes the destination path with `move_output_file()`.

## Output

Default revised output:

- `<input>.revised.<ext>`

Alternative behaviors:

- `--in-place` overwrites the translated input
- `--dry-run` reports revision results without writing output

## Structured Issues

Revision responses can include structured review issues with:

- `code`
- `message`
- `severity`

Current issue namespace:

- `revise.instruction_ambiguous`
- `revise.source_unclear`
- `revise.glossary_variant_choice`
- `revise.placeholder_attention`
- `revise.length_or_ui_fit_risk`
- `revise.other`

Typical severities:

- `warning`
- `info`

## Important Behavior

- unchanged translations are kept unchanged
- only items that clearly need modification are updated
- empty revised translations are not allowed
- the task respects shared localization invariants and glossary-sense rules
- changed AI-reviewed entries are marked review-required where the format supports it

## When Not To Use This Task

- use `translate` for initial translation
- use `check` for QA reporting
- use `extract-terms` or `extract-terms-local` for glossary discovery
