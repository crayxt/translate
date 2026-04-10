# Extraction Refactor

## Goal

Unify three things that had started to diverge:

- local term discovery
- translation-time glossary matching
- extraction resource handling

The refactor moves reusable extraction logic into `core/term_extraction.py`, moves JSON/PO handoff logic into `core/term_handoff.py`, and promotes the local workflow into `tasks/extract_terms_local.py`.

The broader extraction goal remains the same:

- make glossary output more context-aware
- make extracted terms more atomic and reusable
- reduce noisy phrase-shaped candidates
- reduce dependence on prompt wording alone

This is still intended as an incremental change, not a destabilizing rewrite of the model-based extractor.

## What Lives Where

### Shared core

`core/term_extraction.py` now owns:

- extraction resource loading from `data/extract/...`
- UI text normalization
- accelerator stripping
- tokenization
- stopword / low-value / excluded-term filtering
- candidate counting for unigrams / bigrams / trigrams
- vocabulary-aware canonicalization
- evidence collection
- scoring and classification into:
  - `accepted`
  - `borderline`
  - `rejected`
- scoped glossary matching for translation-time vocabulary hints
- message-scoped glossary matching that can be reused by the translator

Main public pieces include:

- `SourceMessage`
- `CandidateEvidence`
- `ExtractionResult`
- `extract_terms_locally(...)`
- `build_scoped_vocabulary_entries(...)`
- `build_relevant_vocabulary(...)`
- `collect_source_messages(...)`

`core/term_handoff.py` now owns:

- building the local extraction JSON payload
- converting local extraction JSON into PO
- translation candidate export helpers
- PO note and occurrence shaping

This keeps extraction mechanics separate from export and review handoff logic.

### Local task

`tasks/extract_terms_local.py` now owns:

- loading entries from supported file formats
- local extraction workflow
- JSON save flow
- one-shot JSON + PO handoff generation
- JSON-to-PO conversion CLI mode
- CLI argument handling
- the dedicated GUI-backed local discovery workflow

In the GUI-backed flow, normal local extraction writes both the JSON report and the derived PO handoff in one run.

## Data Layout

The refactor also separates locale resources from extraction resources:

```text
data/
  locales/
    kk/
      vocab.txt
      rules.md
  extract/
    common/
      abbreviations.txt
      excluded_terms.txt
    en/
      stopwords.txt
      low_value_words.txt
      fixed_multiword_allowlist.txt
```

This split matters because:

- `data/locales/...` is target-language translation policy and approved terminology
- `data/extract/...` is source-language term-mining behavior

## Translation-Time Glossary Flow

The translation task no longer relies only on a large top-level vocabulary blob.

Current flow:

1. Load the approved vocabulary from `data/locales/<target-lang>/vocab.txt`, a glossary PO, or a glossary TBX.
2. Parse it into rich entries:
   - `source_term`
   - `target_term`
   - `part_of_speech`
   - `context_note`
3. Build scoped glossary matchers once per run.
4. For each source message, find only the relevant glossary entries.
5. Inject those into that message as `relevant_vocabulary`.

Example message payload:

```json
{
  "source": "Start playback",
  "context": "Toolbar action",
  "relevant_vocabulary": [
    {
      "source_term": "start",
      "target_term": "бастау",
      "part_of_speech": "verb",
      "context_note": "Start playback"
    },
    {
      "source_term": "playback",
      "target_term": "ойнату",
      "part_of_speech": "noun",
      "context_note": "Media playback"
    }
  ]
}
```

The full vocabulary text is still kept in the request for compatibility, but the design direction is now message-scoped suggestions, not global prompt dumping.

The translation task can also return a per-message `warnings` array when a message is ambiguous or otherwise review-worthy. When enabled, those warnings are written to a separate `*.translation-warnings.json` sidecar so risky messages can be inspected directly.

## Local Discovery Flow

The local discovery process now looks like this:

1. Load source entries from PO / TS / RESX / STRINGS / TXT through `tasks/extract_terms_local.py`.
2. Convert them into normalized `SourceMessage` objects.
3. Run shared local extraction through `core.term_extraction.extract_terms_locally(...)`.
4. Emit:
   - accepted terms
   - borderline terms
   - rejected terms
   - translation candidates
5. Optionally write the matching PO handoff in the same run, or convert the resulting JSON into a translation-ready PO glossary file through `core.term_handoff.py`.

## Problems This Addresses

- bare-string extraction lost useful context such as `msgctxt`, comments, and notes
- the model tended to return message-shaped phrases instead of glossary terms
- `missing` mode relied too heavily on prompt wording instead of deterministic local logic
- the old pipeline had little evidence aggregation across messages
- there was no local confidence model for deciding when to keep a compound term and when to split it

## Why This Refactor Matters

- One normalization and matching path now serves both local discovery and translation-time glossary injection.
- Extraction resources are data-driven instead of hardcoded Python sets.
- The local discovery workflow is now a proper task instead of a standalone prototype script.
- The production translator can reuse shared extraction primitives without importing export or CLI logic.
- Translation-time glossary injection and translation warning reporting now both work at the message level instead of as coarse batch-level blobs.

## Current Direction

The intended long-term direction is still a conservative hybrid flow:

1. collect contextual source messages
2. generate local candidate terms from those messages
3. aggregate evidence per candidate
4. apply conservative local filtering
5. use the model for validation, canonicalization, and translation rather than raw free-form discovery

That split is safer than asking the model to do raw discovery and target-language suggestion in one step.

## Remaining Roadmap

- Reuse `core.term_extraction` inside `tasks/extract_terms.py` before model-based term extraction.
- Tune phrase-denylist and allowlist resources under `data/extract/`.
- Reduce dependence on the full top-level `vocabulary` field once message-scoped hints prove reliable enough.
- Continue improving local evidence signals:
  - message frequency
  - exact-label frequency
  - context diversity
  - repeated occurrence across files or locations
  - optional source-file occurrence summaries
- Continue tightening deterministic local filtering for:
  - existing glossary terms
  - stop-word noise
  - placeholders, tags, and structural tokens
  - weak one-off multi-word phrases
  - trivial non-terms that should never reach the final glossary

Important: local filtering should keep preferring rejection of uncertain phrases over inventing overly aggressive splits.

## Acceptance Criteria

The extraction work should be considered successful when:

- loose UI phrases are substantially reduced
- fixed technical compounds are preserved reliably
- context-sensitive terms are handled better than before
- `missing` mode drops already-known glossary terms deterministically
- output carries enough evidence to make human review faster

## Non-Goals

- no big destabilizing rewrite of `tasks/extract_terms.py` in one step
- no attempt to solve all terminology quality issues with prompt wording alone
- no aggressive local auto-splitting that could damage valid compounds
