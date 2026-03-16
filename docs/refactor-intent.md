# Translation Toolkit Refactor Intent

## Goal

Refactor this repository from a set of partially overlapping scripts into one coherent translation toolkit with:

- a shared backend for file loading, resource lookup, runtime controls, and Gemini task execution
- a unified CLI surface with subcommands
- a separate GUI frontend that targets the unified CLI/backend

This is an architectural refactor, not a product rewrite. Existing behavior should be preserved during the migration.

## Current Scripts

- `process.py`: translation pipeline for `.po`, `.ts`, `.resx`, `.strings`, `.txt`
- `revise_translations.py`: instruction-driven revision of existing translations
- `check_translations.py`: QA/check pass over existing translations
- `extract_terms.py`: glossary extraction
- `process_gui.py`: Tkinter frontend that shells out to the scripts above

## Direction

We do not want one giant Python file.

We do want:

- one shared backend
- one CLI entry point with subcommands
- one separate GUI frontend

## Target Shape

Suggested end state:

- `core/`
- `tasks/`
- `translate_cli.py`
- compatibility wrappers:
  - `process.py`
  - `revise_translations.py`
  - `check_translations.py`
  - `extract_terms.py`
- `process_gui.py` as frontend-only code

## Guiding Rules

- Prefer staged extraction over large rewrites.
- Keep current command-line behavior stable until wrappers are in place.
- Move shared mechanics first, then unify task runners, then unify CLI surface.
- Keep task-specific prompts and schemas separate.
- Keep GUI-specific code out of backend modules.

## Planned Stages

### Stage 1: Extract Shared Infrastructure

Extract low-risk duplicated helpers first:

- resource lookup/loading helpers
- runtime limit wrappers
- JSON payload parsing helpers
- script/alphabet guidance helpers
- common reviewable-entry filtering helpers

### Stage 2: Formalize File and Entry APIs

Treat the reusable parts of `process.py` as the file-format backend:

- file kind detection
- unified entry model
- format loaders and save callbacks
- output-path helpers

### Stage 3: Shared Review Engine

Create shared abstractions for tasks that operate on existing translations:

- review item
- review bundle
- paired-source loading
- batch runner

Then migrate:

- `check_translations.py`
- `revise_translations.py`

onto that engine.

### Stage 4: Isolate Task Contracts

Each task should own only:

- prompt builder
- response schema
- response parser
- task-specific result handling

Tasks:

- translate
- revise
- check
- extract terms

### Stage 5: Unified CLI

Introduce one CLI with subcommands:

- `translate`
- `revise`
- `check`
- `extract-terms`

Keep current scripts as compatibility wrappers during migration.

### Stage 6: GUI Simplification

Keep the GUI separate, but make it call the unified CLI. The GUI should retain only:

- form state
- file pickers
- subprocess orchestration
- log/progress display

Validation and command assembly should move into shared backend code where possible.

## First Refactor Slice

The first implementation slice should be intentionally small:

1. add a shared review utility module
2. move duplicated helpers from `check_translations.py` and `revise_translations.py`
3. keep both existing scripts functional with unchanged CLIs
4. add or update tests around the extracted helpers

## Non-Goals For Early Stages

- no prompt unification across all tasks
- no giant generic response schema
- no GUI rewrite
- no big-bang move to one script
