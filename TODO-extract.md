# Terminology Extraction TODO

## Goal

Improve terminology extraction so the glossary output is:

- more context-aware
- more atomic and reusable
- less noisy
- less dependent on prompt wording alone

This should remain an incremental change. We do not want to destabilize the current
`extract-terms` command while we are still learning what filtering works well.

## Current Problems

- Bare-string extraction loses useful context such as `msgctxt`, comments, and notes.
- The model still tends to return message-shaped phrases instead of glossary terms.
- `missing` mode is mostly enforced by prompt wording, not by deterministic local logic.
- The current pipeline has little evidence aggregation across messages.
- There is no local confidence model for deciding when to keep a compound term and when to split it.

## Current Direction

We want the extractor to move toward a hybrid approach:

1. collect contextual source messages
2. generate local candidate terms from those messages
3. aggregate evidence per candidate
4. apply conservative local filtering
5. use the model for validation/canonicalization/translation rather than raw free-form discovery

## In Progress

- `tasks/extract_terms.py` now sends contextual message objects instead of only bare source strings
- `prototype_term_extractor.py` exists as a standalone local prototype
- the prototype keeps the main extractor untouched while we evaluate heuristics

## Intent For The Prototype

The prototype should answer these questions:

- Can local candidate mining reduce phrase noise like `audio channel`?
- Can we preserve fixed compounds like `access token` and `dark mode`?
- Can `missing` mode be enforced locally with glossary-aware filtering?
- Can we attach evidence such as examples, contexts, and notes to each term candidate?

## Planned Steps

### Step 1: Validate Local Candidate Mining

Use `prototype_term_extractor.py` to test:

- one-word candidates
- repeated multi-word candidates
- fixed compound allowlist behavior
- glossary filtering in `missing` mode
- evidence capture for contexts and notes

### Step 2: Improve Evidence Model

Add stronger evidence signals:

- message frequency
- exact-label frequency
- context diversity
- repeated occurrence across files or locations
- optional source-file occurrence summaries

### Step 3: Tighten Local Filtering

Implement conservative deterministic filtering for:

- existing glossary terms
- stop-word noise
- placeholders / tags / structural tokens
- weak one-off multi-word phrases
- trivial non-terms that should never reach the final glossary

Important: local filtering should prefer rejecting uncertain phrases over inventing splits.

### Step 4: Compare Prototype Output Against Real Files

Run the prototype on representative files and review:

- false positives
- false negatives
- over-splitting
- under-splitting
- useful compounds missing from the allowlist

### Step 5: Integrate Carefully Into Main Extractor

After the prototype is good enough:

- move reusable local filtering helpers into shared code
- keep the current `extract-terms` interface stable
- integrate the local candidate/evidence pass before model output is finalized
- preserve structured request/response behavior

### Step 6: Move Toward Two-Stage Extraction

Longer term, the main flow should become:

1. source-side candidate discovery and filtering
2. model validation and canonicalization
3. target-language suggestion generation

This is safer than asking the model to do raw discovery and translation in one step.

## Acceptance Criteria

We should consider the extractor improved when:

- loose UI phrases are substantially reduced
- fixed technical compounds are preserved reliably
- context-sensitive terms are handled better than before
- `missing` mode drops already-known glossary terms deterministically
- output carries enough evidence to make human review faster

## Non-Goals

- no big rewrite of `tasks/extract_terms.py` yet
- no immediate replacement of the current extraction command
- no attempt to solve all terminology quality issues with prompt wording alone
- no aggressive local auto-splitting that could damage valid compounds
