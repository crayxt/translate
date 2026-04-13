#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from core.review_common import (
    build_target_script_guidance as build_shared_target_script_guidance,
    json_load_maybe,
    plural_key_sort_key,
)
from core.formats import EntryStatus
from core.entries import normalize_model_escaped_text
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, TranslationProvider
from core.request_contents import TaskRequestSpec, build_task_request_contents
from core.review_bundle import (
    ReviewItem,
    build_revision_output_path,
    load_review_bundle,
)
from core.task_cli import (
    add_language_arguments,
    add_max_attempts_argument,
    add_probe_argument,
    add_provider_arguments,
    add_rules_arguments,
    add_runtime_limit_arguments,
    add_vocabulary_argument,
    build_task_parser,
    resolve_provider_model,
    run_task_main,
)
from core.review_flow import (
    ReviewBatchRunnerSpec,
    ReviewRetryRunnerSpec,
    build_retry_review_batch_messages,
    build_review_batch_messages,
    find_missing_mapping_indices,
    limit_items,
    merge_mapping_review_results,
    prepare_review_run,
    print_review_startup,
    run_review_batches,
)
from core.runtime import DEFAULT_BATCH_SIZE, DEFAULT_PARALLEL_REQUESTS
from core.system_instructions import (
    SHARED_GLOSSARY_SENSE_RULES,
    SHARED_LOCALIZATION_INVARIANTS,
    join_instruction_sections,
)
from core.task_issues import (
    TaskIssue,
    build_task_issue_schema,
    normalize_task_issue,
)


DEFAULT_REVISION_BATCH_SIZE = DEFAULT_BATCH_SIZE
DEFAULT_REVISION_PARALLEL = DEFAULT_PARALLEL_REQUESTS

REVISION_SYSTEM_INSTRUCTION = join_instruction_sections(
    """
    You are revising existing software localization translations.

    REVISION REQUIREMENTS:
    - Review each item against the source text, current translation, and user instruction
    - Keep the current translation unchanged when it already satisfies the instruction
    - Change only entries where the instruction clearly applies and the current translation needs an update
    - If the instruction is ambiguous or not clearly applicable to a specific item, keep that item unchanged
    - Never return an empty updated translation
    - Do not rewrite unrelated wording just because a different phrasing is possible
    - Do not translate or rewrite context/note metadata
    """,
    SHARED_LOCALIZATION_INVARIANTS,
    SHARED_GLOSSARY_SENSE_RULES,
)

REVISION_ISSUE_CODES: tuple[str, ...] = (
    "revise.instruction_ambiguous",
    "revise.source_unclear",
    "revise.glossary_variant_choice",
    "revise.placeholder_attention",
    "revise.length_or_ui_fit_risk",
    "revise.other",
)

REVISION_ISSUE_CODE_GUIDANCE: Dict[str, str] = {
    "revise.instruction_ambiguous": "the instruction is ambiguous for this item",
    "revise.source_unclear": "the source meaning is unclear or underspecified",
    "revise.glossary_variant_choice": "multiple approved glossary variants existed and one was chosen",
    "revise.placeholder_attention": "placeholders or protected tokens required extra care",
    "revise.length_or_ui_fit_risk": "the revised text may be risky for UI fit or length",
    "revise.other": "there is another review-worthy concern that does not fit a more specific code",
}


REVISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "action": {"type": "string"},
                    "text": {"type": "string"},
                    "plural_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": build_task_issue_schema(
                            REVISION_ISSUE_CODES,
                            allowed_severities=("warning", "info"),
                        ),
                    },
                },
                "required": ["id", "action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["revisions"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class RevisionResult:
    """Normalized model output for one reviewed translation item."""
    action: str
    text: str = ""
    plural_texts: List[str] = field(default_factory=list)
    reason: str = ""
    issues: List[TaskIssue] = field(default_factory=list)


def build_revision_generation_config(
    thinking_level: str | None = None,
    *,
    provider: TranslationProvider = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
    flex_mode: bool = False,
) -> Any:
    """Build the provider generation config for revision batches."""
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=REVISION_RESPONSE_SCHEMA,
        system_instruction=REVISION_SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
        flex_mode=flex_mode,
    )


def configure_stdio() -> None:
    """Force UTF-8 console output when the runtime supports reconfiguration."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _clean(text: str | None) -> str:
    """Trim a possibly missing string value."""
    return str(text or "").strip()


def _normalize_revision_payload(payload: Any) -> Dict[str, RevisionResult]:
    """Normalize raw revision JSON into typed per-item results."""
    results: Dict[str, RevisionResult] = {}
    if not isinstance(payload, dict):
        return results

    items = payload.get("revisions")
    if not isinstance(items, list):
        return results

    for item in items:
        if not isinstance(item, dict):
            continue
        msg_id = item.get("id")
        action = item.get("action")
        if msg_id is None or action is None:
            continue

        text = item.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        reason = item.get("reason")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            reason = str(reason)

        plural_texts: List[str] = []
        plural_texts_raw = item.get("plural_texts")
        if isinstance(plural_texts_raw, list):
            for value in plural_texts_raw:
                if value is None:
                    continue
                plural_texts.append(value if isinstance(value, str) else str(value))

        issues: List[TaskIssue] = []
        issues_raw = item.get("issues")
        if isinstance(issues_raw, list):
            for value in issues_raw:
                issue = normalize_task_issue(
                    value,
                    allowed_codes=REVISION_ISSUE_CODES,
                    default_code="revise.other",
                    allowed_severities=("warning", "info"),
                    default_severity="warning",
                    default_origin="model",
                )
                if issue is not None:
                    issues.append(issue)

        results[str(msg_id)] = RevisionResult(
            action=str(action).strip().lower(),
            text=text,
            plural_texts=plural_texts,
            reason=reason,
            issues=issues,
        )

    return results


def parse_revision_response(response_payload: Any) -> Dict[str, RevisionResult]:
    """Parse provider output into revision results keyed by item id."""
    if isinstance(response_payload, dict):
        return _normalize_revision_payload(response_payload)

    if isinstance(response_payload, str):
        return _normalize_revision_payload(json_load_maybe(response_payload))

    parsed_payload = getattr(response_payload, "parsed", None)
    if parsed_payload is not None:
        return _normalize_revision_payload(parsed_payload)

    text_payload = getattr(response_payload, "text", None) or ""
    return _normalize_revision_payload(json_load_maybe(text_payload))


def build_target_script_guidance(target_lang: str) -> str | None:
    """Return target-script guidance for revised translations."""
    return build_shared_target_script_guidance(
        target_lang,
        update_wording=lambda: "updated target text",
    )


def build_revision_system_instruction(target_lang: str) -> str:
    """Build the final revision system instruction for the selected language."""
    parts = [REVISION_SYSTEM_INSTRUCTION.strip()]
    script_guidance = build_target_script_guidance(target_lang)
    if script_guidance:
        parts.append(f"- {script_guidance}")
    return "\n\n".join(parts)


def build_revision_request_spec() -> TaskRequestSpec:
    """Describe the structured contract for revision batches."""
    return TaskRequestSpec(
        task_intro="Revise each software localization translation item.",
        task_lines=(
            "Review each item against the source text, current translation, and user instruction.",
        ),
        payload_lines=(
            "The payload contains the user instruction, source language, target language, optional approved vocabulary/glossary, optional project translation rules/instructions, and an `items` map.",
        ),
        output_lines=(
            "Return only valid JSON, with no markdown or commentary.",
            "Return one result for every input item id.",
            "Keep each result `id` exactly the same as the input key.",
            "Use action `keep` when no change is needed.",
            "Use action `update` only when the current translation should change to satisfy the instruction.",
            "If action is `update`, provide the full corrected target text.",
            "For plural items, if action is `update`, provide exactly `item.plural_forms` plural_texts.",
            "If the target language effectively uses one plural wording, repeat it in all required plural slots.",
            "Keep `reason` short and concrete.",
            "Use `issues` only when an item has a real ambiguity or another review-worthy concern.",
            "Each issue must be an object with `code`, `message`, and `severity`.",
            f"Allowed issue codes: {', '.join(REVISION_ISSUE_CODES)}.",
            *tuple(
                f"Use `{code}` when {description}."
                for code, description in REVISION_ISSUE_CODE_GUIDANCE.items()
            ),
            "Use severity `warning` for real revision risk and `info` for notable but non-blocking decisions.",
        ),
    )


def build_revision_request_payload(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> dict[str, Any]:
    """Build the structured payload for a revision batch."""
    return {
        "project_type": "software_ui_revision",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "instruction": instruction,
        "vocabulary": vocabulary,
        "translation_rules": translation_rules,
        "items": messages,
    }


def build_revision_request_contents(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    instruction: str,
    vocabulary: str | None,
    translation_rules: str | None,
    *,
    provider: TranslationProvider = DEFAULT_PROVIDER,
) -> Any:
    """Build provider-native request contents for a revision batch."""
    return build_task_request_contents(
        provider=provider,
        task_spec=build_revision_request_spec(),
        function_name="translation_revision_batch",
        payload=build_revision_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            instruction=instruction,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
        ),
    )


def build_review_message_payload(item: ReviewItem) -> Dict[str, Any]:
    """Build one revision payload item from a prepared review record."""
    payload: Dict[str, Any] = {
        "source": item.source_text,
        "current_translation": item.current_text,
    }
    if item.current_plural_texts:
        payload["current_plural_texts"] = item.current_plural_texts
    if item.plural_form_count:
        payload["plural_forms"] = item.plural_form_count
    if item.context:
        payload["context"] = item.context
    if item.note:
        payload["note"] = item.note
    return payload


def build_candidate_plural_forms(item: ReviewItem, result: RevisionResult) -> List[str]:
    """Normalize and pad model-supplied plural revisions to the required slot count."""
    usable_forms = [
            normalize_model_escaped_text(item.source_text, text)
        for text in result.plural_texts
        if _clean(text)
    ]
    if usable_forms:
        if len(usable_forms) >= item.plural_form_count:
            return usable_forms[: item.plural_form_count]
        return usable_forms + [usable_forms[-1]] * (item.plural_form_count - len(usable_forms))

    if _clean(result.text):
        repeated = normalize_model_escaped_text(item.source_text, result.text)
        return [repeated] * item.plural_form_count

    return []


def apply_revision_to_item(item: ReviewItem, result: RevisionResult) -> bool:
    """Apply one normalized revision result back onto its entry."""
    if result.action != "update":
        return False

    if item.plural_form_count:
        candidate_forms = build_candidate_plural_forms(item, result)
        if not candidate_forms:
            return False
        if candidate_forms == item.current_plural_texts:
            return False

        plural_keys = sorted(item.entry.msgstr_plural.keys(), key=plural_key_sort_key)
        if not plural_keys:
            plural_keys = list(range(item.plural_form_count))
        for index, key in enumerate(plural_keys):
            item.entry.msgstr_plural[key] = candidate_forms[index]
        item.entry.msgstr = candidate_forms[0]
    else:
        candidate = normalize_model_escaped_text(item.source_text, result.text)
        if not _clean(candidate):
            return False
        if candidate == item.current_text:
            return False
        item.entry.msgstr = candidate

    if "fuzzy" not in item.entry.flags:
        item.entry.flags.append("fuzzy")
    item.entry.status = EntryStatus.FUZZY
    return True


def move_output_file(generated_output_path: str, final_output_path: str) -> None:
    """Move a generated revision output into its final destination when needed."""
    if generated_output_path == final_output_path:
        return
    os.replace(generated_output_path, final_output_path)


def build_final_output_path(
    translated_file: str,
    explicit_out: str | None = None,
    in_place: bool = False,
) -> str:
    """Resolve the final output location for a revision run."""
    if explicit_out:
        return explicit_out
    if in_place:
        return translated_file
    return build_revision_output_path(translated_file)


def print_change_samples(changes: List[Tuple[ReviewItem, RevisionResult]], limit: int = 10) -> None:
    """Print a short sample of updated entries for operator visibility."""
    if not changes:
        return

    print("Sample updated entries:")
    for item, result in changes[:limit]:
        label = item.pair_key or item.context or item.source_text
        before = item.current_text.replace("\n", "\\n")
        after = (
            result.plural_texts[0]
            if result.plural_texts
            else result.text
        ).replace("\n", "\\n")
        print(f"  - {label}: {before} -> {after}")
        if result.issues:
            for issue in result.issues[:3]:
                print(f"      [{issue.code}] {issue.message}")


def print_issue_samples(issues: List[Tuple[ReviewItem, RevisionResult]], limit: int = 10) -> None:
    """Print a short sample of revision issues returned by the model."""
    if not issues:
        return

    print("Sample revision issues:")
    for item, result in issues[:limit]:
        label = item.pair_key or item.context or item.source_text
        print(f"  - {label}")
        for issue in result.issues[:3]:
            print(f"      [{issue.code}] {issue.message}")


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure the standalone CLI for translation revision."""
    parser.description = (
        "Review existing translations against a natural-language instruction and "
        "update only the entries that need a change."
    )
    parser.add_argument(
        "file",
        help="Current translated .po, .xlf/.xliff, .ts, .resx, .strings, .txt, or Android .xml file",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural-language revision instruction, for example: change the translation of 'Save' to 'Store'",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Required for Android .xml, .resx, .strings, and .txt revision runs; optional otherwise.",
    )
    add_language_arguments(parser)
    add_provider_arguments(
        parser,
        default_provider_name=DEFAULT_PROVIDER_NAME,
        default_model=DEFAULT_PROVIDER.default_model,
    )
    add_runtime_limit_arguments(parser)
    add_probe_argument(
        parser,
        help_text="Review only the first N reviewable entries.",
    )
    add_max_attempts_argument(parser)
    add_vocabulary_argument(parser)
    add_rules_arguments(
        parser,
        rules_help="Optional translation rules file (auto: data/locales/<target-lang>/rules.md)",
        rules_str_help="Optional inline translation rules",
    )
    parser.add_argument("--out", default=None, help="Output path (default: <input>.revised.<ext>)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the translated input file")
    parser.add_argument("--dry-run", action="store_true", help="Review and report changes without writing output")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for translation revision."""
    return build_task_parser(configure_parser)


def run_from_args(args: argparse.Namespace) -> None:
    """Execute translation revision from parsed CLI arguments."""
    configure_stdio()
    model_name = resolve_provider_model(args.provider, args.model)

    if args.out and args.in_place:
        sys.exit("ERROR: --out and --in-place cannot be used together")
    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

    try:
        review_bundle = load_review_bundle(args.file, source_file=args.source_file)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    try:
        review_items = limit_items(review_bundle.items, args.num_messages)
        if not review_items:
            print("No translated entries found to review.")
            return
        review_run = prepare_review_run(
            items=review_items,
            provider_name=args.provider,
            target_lang=args.target_lang,
            flex_mode=args.flex_mode,
            explicit_vocab_path=args.vocab,
            explicit_rules_path=args.rules,
            inline_rules=args.rules_str,
            batch_size_arg=args.batch_size,
            parallel_arg=args.parallel_requests,
            default_batch_size=DEFAULT_REVISION_BATCH_SIZE,
            default_parallel=DEFAULT_REVISION_PARALLEL,
            label="revision",
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    provider = review_run.runtime_context.provider
    client = review_run.runtime_context.client
    resource_context = review_run.runtime_context.resources
    revision_config = build_revision_generation_config(
        args.thinking_level,
        provider=provider,
        system_instruction=build_revision_system_instruction(args.target_lang),
        flex_mode=args.flex_mode,
    )
    final_output_path = build_final_output_path(
        translated_file=args.file,
        explicit_out=args.out,
        in_place=args.in_place,
    )

    print_review_startup(
        provider=provider,
        model_name=model_name,
        flex_mode=args.flex_mode,
        thinking_level=args.thinking_level,
        parallel_requests=review_run.parallel_requests,
        batch_size=review_run.batch_size,
        limits_mode=review_run.limits_mode,
        resource_context=resource_context,
        item_label="Review items",
        item_count=len(review_run.items),
        extra_entries=(
            ("Source file", args.source_file or "embedded in translated file"),
            ("Output path", final_output_path),
            ("Dry run", "yes" if args.dry_run else "no"),
            ("Total batches", review_run.batch_count),
        ),
    )
    for warning in review_bundle.warnings:
        print(f"Warning: {warning}")

    total_batches = review_run.batch_count

    async def run_revision() -> Tuple[int, List[Tuple[ReviewItem, RevisionResult]], List[Tuple[ReviewItem, RevisionResult]]]:
        changed_total = 0
        changed_items: List[Tuple[ReviewItem, RevisionResult]] = []
        issue_items: List[Tuple[ReviewItem, RevisionResult]] = []
        completed_batches = 0

        def build_contents(_batch_index: int, batch: List[ReviewItem]) -> Any:
            return build_revision_request_contents(
                messages=build_review_batch_messages(batch, build_review_message_payload),
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                instruction=args.instruction,
                vocabulary=resource_context.vocabulary_text,
                translation_rules=resource_context.project_rules,
                provider=provider,
            )

        def build_retry_contents(
            _batch_index: int,
            batch: List[ReviewItem],
            missing_indices: List[int],
        ) -> Any:
            return build_revision_request_contents(
                messages=build_retry_review_batch_messages(
                    batch,
                    missing_indices,
                    build_review_message_payload,
                ),
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                instruction=args.instruction,
                vocabulary=resource_context.vocabulary_text,
                translation_rules=resource_context.project_rules,
                provider=provider,
            )

        def on_batch_completed(
            batch_index: int,
            batch: List[ReviewItem],
            revisions: Dict[str, RevisionResult],
        ) -> None:
            nonlocal changed_total, completed_batches
            batch_changed = 0

            for index, item in enumerate(batch):
                result = revisions.get(str(index))
                if result is None:
                    continue
                if result.issues:
                    issue_items.append((item, result))
                if apply_revision_to_item(item, result):
                    batch_changed += 1
                    changed_items.append((item, result))

            changed_total += batch_changed
            completed_batches += 1
            percent = (completed_batches / total_batches) * 100.0
            print(
                f"Progress: {percent:.1f}% ({completed_batches}/{total_batches} batches), "
                f"changed so far: {changed_total}"
            )

        await run_review_batches(
            batches=review_run.batch_plan.batches,
            parallel_requests=review_run.parallel_requests,
            provider=provider,
            client=client,
            model=model_name,
            config=revision_config,
            max_attempts=args.max_attempts,
            runner_spec=ReviewBatchRunnerSpec(
                build_contents=build_contents,
                parse_response=parse_revision_response,
                on_batch_completed=on_batch_completed,
                build_batch_label=lambda batch_index: f"revision batch {batch_index + 1}/{total_batches}",
                retry_spec=ReviewRetryRunnerSpec(
                    find_missing_indices=find_missing_mapping_indices,
                    build_retry_contents=build_retry_contents,
                    build_retry_label=lambda batch_index: (
                        f"revision batch {batch_index + 1}/{total_batches} missing-items"
                    ),
                    merge_retry_result=merge_mapping_review_results,
                    retry_max_attempts=lambda _batch_index: max(2, min(args.max_attempts, 3)),
                    on_missing_indices=lambda batch_index, _batch, missing_indices: print(
                        f"  Warning [batch {batch_index + 1}/{total_batches}]: "
                        f"{len(missing_indices)} items missing from response. Retrying them..."
                    ),
                    on_retry_error=lambda batch_index, _batch, _missing_indices, exc: print(
                        f"  Retry failed [batch {batch_index + 1}/{total_batches}]: {exc}"
                    ),
                ),
            ),
        )

        return changed_total, changed_items, issue_items

    try:
        changed_count, changed_items, issue_items = asyncio.run(run_revision())
    except RuntimeError as exc:
        sys.exit(str(exc))

    print("")
    print(f"Reviewed entries: {len(review_run.items)}")
    print(f"Changed entries: {changed_count}")
    print(f"Entries with revision issues: {len(issue_items)}")

    if not changed_count:
        if issue_items:
            print("No changes were needed, but some entries reported revision issues.")
            print_issue_samples(issue_items)
        else:
            print("No changes were needed.")
        return

    print_change_samples(changed_items)
    if issue_items:
        print_issue_samples(issue_items)

    if args.dry_run:
        print("Dry run complete. No file was written.")
        return

    review_bundle.save_callback()
    move_output_file(review_bundle.generated_output_path, final_output_path)

    print("")
    print("Revision complete.")
    print(f"Saved file: {final_output_path}")
    print("Changed AI-reviewed entries are marked as fuzzy/unfinished for human review.")


def main(argv: list[str] | None = None) -> None:
    """Run the translation revision CLI."""
    run_task_main(
        configure_parser_fn=configure_parser,
        run_from_args_fn=run_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
