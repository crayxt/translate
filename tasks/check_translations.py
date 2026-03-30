#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from core.review_common import (
    build_target_script_guidance as build_shared_target_script_guidance,
    json_load_maybe,
    plural_key_sort_key,
)
from core.formats import (
    FileKind,
    build_entry_source_text,
    detect_file_kind,
    get_entry_prompt_context_and_note,
    load_po,
)
from core.providers import DEFAULT_PROVIDER, DEFAULT_PROVIDER_NAME, get_translation_provider
from core.request_contents import TaskRequestSpec, build_task_request_contents, render_text_fallback_prompt
from core.resources import read_optional_text_file, read_optional_vocabulary_file, resolve_resource_path
from core.runtime import add_thinking_level_argument
from core.review_flow import (
    has_reviewable_translation as has_shared_reviewable_translation,
    limit_items,
    normalize_limits as normalize_review_limits,
)
from core.task_batches import build_fixed_batches, run_parallel_batches
from core.task_resources import load_task_resource_context


DEFAULT_CHECK_BATCH_SIZE = 150
DEFAULT_CHECK_PARALLEL = 6

CHECK_SYSTEM_INSTRUCTION = """
You are a software localization QA reviewer.

STRICT MUST-CHECK:
- Placeholders must be preserved exactly (%s, %d, %(name)s, %1, %n, {var}, {{var}})
- HTML/XML tags must be preserved exactly and remain well-formed
- Keyboard accelerators/hotkeys must be preserved and usable (`_`, `&`)
- Approved vocabulary is mandatory when supplied
- Inflection and derivation are acceptable when they preserve the approved lexical choice
- Flag missing meaning, added meaning, mistranslation, wrong tone, or broken grammar only when they are real QA issues
- Ignore purely stylistic alternatives unless they violate terminology, rules, or UI constraints
- Prefer fewer but higher-confidence findings over speculative nitpicks
- When project rules are provided, apply them as mandatory QA criteria
"""


CHECK_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "severity": {"type": "string"},
                                "message": {"type": "string"},
                                "source_fragment": {"type": "string"},
                                "translation_fragment": {"type": "string"},
                                "suggested_translation": {"type": "string"},
                            },
                            "required": ["category", "severity", "message"],
                        },
                    },
                },
                "required": ["id", "issues"],
            },
        },
    },
    "required": ["results"],
}


@dataclass
class CheckIssue:
    origin: str
    category: str
    severity: str
    message: str
    source_fragment: str = ""
    translation_fragment: str = ""
    suggested_translation: str = ""


def build_check_generation_config(
    thinking_level: str | None = None,
    *,
    provider: Any = DEFAULT_PROVIDER,
    system_instruction: str | None = None,
) -> Any:
    return provider.build_generation_config(
        thinking_level=thinking_level,
        json_schema=CHECK_RESPONSE_SCHEMA,
        system_instruction=CHECK_SYSTEM_INSTRUCTION if system_instruction is None else system_instruction,
    )


async def generate_with_retry(
    *,
    provider: Any,
    client: Any,
    model: str,
    contents: Any,
    batch_label: str,
    max_attempts: int = 5,
    config: Any = None,
) -> Any:
    return await provider.generate_with_retry(
        client=client,
        model=model,
        contents=contents,
        batch_label=batch_label,
        max_attempts=max_attempts,
        config=config,
    )


def build_check_output_path(file_path: str) -> str:
    root, _ = os.path.splitext(file_path)
    return f"{root}.translation-check.json"


def normalize_limits(
    total_items: int,
    batch_size_arg: int | None,
    parallel_arg: int | None,
) -> Tuple[int, int, str]:
    return normalize_review_limits(
        total_items=total_items,
        batch_size_arg=batch_size_arg,
        parallel_arg=parallel_arg,
        default_batch_size=DEFAULT_CHECK_BATCH_SIZE,
        default_parallel=DEFAULT_CHECK_PARALLEL,
        label="check",
    )


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _json_load_maybe(text: str) -> Any:
    return json_load_maybe(text)


def _plural_key_sort_key(value: Any) -> Tuple[int, Any]:
    return plural_key_sort_key(value)


def get_translation_plural_forms(entry: Any) -> List[str]:
    plural_map = getattr(entry, "msgstr_plural", None)
    if not isinstance(plural_map, dict) or not plural_map:
        return []
    return [
        str(plural_map[key] or "")
        for key in sorted(plural_map.keys(), key=plural_key_sort_key)
    ]


def get_entry_translation_text(entry: Any) -> str:
    plural_forms = get_translation_plural_forms(entry)
    if plural_forms:
        return plural_forms[0]
    return str(getattr(entry, "msgstr", "") or "")


def get_joined_translation_text(entry: Any) -> str:
    plural_forms = get_translation_plural_forms(entry)
    if plural_forms:
        return "\n".join(plural_forms)
    return get_entry_translation_text(entry)


def has_reviewable_translation(entry: Any) -> bool:
    return has_shared_reviewable_translation(
        entry,
        plural_texts=get_translation_plural_forms(entry),
    )


def select_review_entries(entries: List[Any]) -> List[Any]:
    return [entry for entry in entries if has_reviewable_translation(entry)]


def limit_review_entries(entries: List[Any], num_messages: int | None) -> List[Any]:
    return limit_items(entries, num_messages)



def build_target_script_guidance(target_lang: str) -> str | None:
    guidance = build_shared_target_script_guidance(
        target_lang,
        update_wording=lambda: "suggested target text",
    )
    if guidance and "Kazakh Cyrillic alphabet" in guidance:
        return (
            "For Kazakh, write suggested target text in the real Kazakh Cyrillic alphabet. "
            "Do not use Latin transliteration or lookalike letters such as o/\u00f6/u/\u00fc instead of "
            "Kazakh Cyrillic letters like \u04e9, \u04af, \u04b1, \u049b, \u04a3, \u0493, \u04d9, \u0456, \u04bb."
        )
    return (
        "When you provide target-language fragments or suggested fixes, use the real writing system "
        "and alphabet/script of the target language, not transliteration or lookalike Latin characters."
    )


def build_check_system_instruction(target_lang: str) -> str:
    parts = [CHECK_SYSTEM_INSTRUCTION.strip()]
    script_guidance = build_target_script_guidance(target_lang)
    if script_guidance:
        parts.append(
            "Suggested fixes must use the actual target-language alphabet/script.\n"
            f"- {script_guidance}"
        )
    return "\n\n".join(parts)


def build_check_request_spec() -> TaskRequestSpec:
    return TaskRequestSpec(
        task_intro="Review each software localization translation item.",
        task_lines=(
            "Review each item independently against the source and current translation.",
        ),
        payload_lines=(
            "The payload contains source language, target language, optional approved vocabulary/glossary, optional project translation rules/instructions, and a `messages` map.",
            "Each message item may include `source`, `translation`, optional `translation_plural_forms`, `context`, and `note`.",
        ),
        output_lines=(
            "Return only valid JSON, with no markdown or commentary.",
            "Keep each result `id` exactly the same as the input key.",
            "Return an empty `issues` list when the translation is acceptable.",
            "Use severity `error` for broken placeholders, tags, accelerators, terminology violations, or incorrect meaning.",
            "Use severity `warning` for likely but less certain QA issues.",
            "`suggested_translation` is optional and should be included only when you can provide a clear fix.",
            "If you provide `suggested_translation`, it must use the real target-language alphabet/script.",
            "Do not duplicate the same issue in multiple phrasings.",
            "Do not flag a terminology issue solely because the translation uses an inflected or derived form instead of the exact glossary dictionary form.",
        ),
    )


def build_check_request_payload(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> dict[str, Any]:
    return {
        "project_type": "software_ui_localization_qa",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "vocabulary": vocabulary,
        "translation_rules": translation_rules,
        "messages": messages,
    }


def build_check_request_contents(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
    *,
    provider: Any = DEFAULT_PROVIDER,
) -> Any:
    return build_task_request_contents(
        provider=provider,
        task_spec=build_check_request_spec(),
        function_name="translation_check_batch",
        payload=build_check_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
        ),
    )


def build_check_prompt(
    messages: Dict[str, Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    vocabulary: str | None,
    translation_rules: str | None,
) -> str:
    return render_text_fallback_prompt(
        task_spec=build_check_request_spec(),
        payload=build_check_request_payload(
            messages=messages,
            source_lang=source_lang,
            target_lang=target_lang,
            vocabulary=vocabulary,
            translation_rules=translation_rules,
        ),
    )


def build_check_message_payload(entry: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": build_entry_source_text(entry),
        "translation": get_entry_translation_text(entry),
    }
    plural_forms = get_translation_plural_forms(entry)
    if plural_forms:
        payload["translation_plural_forms"] = plural_forms
    context, note = get_entry_prompt_context_and_note(entry)
    if context:
        payload["context"] = context
    if note:
        payload["note"] = note
    return payload


def parse_check_response(response_payload: Any) -> Dict[str, List[CheckIssue]]:
    if isinstance(response_payload, dict):
        payload = response_payload
    elif isinstance(response_payload, str):
        payload = _json_load_maybe(response_payload)
    else:
        payload = getattr(response_payload, "parsed", None)
        if payload is None:
            payload = _json_load_maybe(getattr(response_payload, "text", None) or "")

    if not isinstance(payload, dict):
        return {}

    results = payload.get("results")
    if not isinstance(results, list):
        return {}

    parsed: Dict[str, List[CheckIssue]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        raw_issues = item.get("issues")
        if not isinstance(raw_issues, list):
            raw_issues = []

        issues: List[CheckIssue] = []
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue
            category = str(raw_issue.get("category", "")).strip() or "other"
            severity = str(raw_issue.get("severity", "")).strip().lower() or "warning"
            if severity not in {"error", "warning"}:
                severity = "warning"
            message = str(raw_issue.get("message", "")).strip()
            if not message:
                continue
            issues.append(
                CheckIssue(
                    origin="model",
                    category=category,
                    severity=severity,
                    message=message,
                    source_fragment=str(raw_issue.get("source_fragment", "")).strip(),
                    translation_fragment=str(raw_issue.get("translation_fragment", "")).strip(),
                    suggested_translation=str(raw_issue.get("suggested_translation", "")).strip(),
                )
            )
        parsed[item_id] = dedupe_issues(issues)

    return parsed


def dedupe_issues(issues: List[CheckIssue]) -> List[CheckIssue]:
    seen: set[Tuple[str, str, str, str, str]] = set()
    unique: List[CheckIssue] = []
    for issue in issues:
        key = (
            issue.origin,
            issue.category.lower(),
            issue.severity.lower(),
            _normalize_space(issue.message).lower(),
            _normalize_space(issue.suggested_translation).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)

    severity_rank = {"error": 0, "warning": 1}
    unique.sort(
        key=lambda issue: (
            severity_rank.get(issue.severity, 9),
            issue.category.lower(),
            issue.message.lower(),
        )
    )
    return unique


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Check translated PO files using the configured provider"
    parser.add_argument("file", help="Input translated .po file")
    parser.add_argument("--source-lang", default="en", help="Default: en")
    parser.add_argument("--target-lang", default="kk", help="Default: kk")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER_NAME,
        help=f"Model provider (default: {DEFAULT_PROVIDER_NAME})",
    )
    parser.add_argument("--model", default=DEFAULT_PROVIDER.default_model)
    add_thinking_level_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if omitted)")
    parser.add_argument("--parallel-requests", type=int, default=None, help="Concurrent requests (auto if omitted)")
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary file (auto: data/<target-lang>/vocab.txt). Supports .txt and glossary .po",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Optional translation rules/instructions file (auto: data/<target-lang>/rules.md)",
    )
    parser.add_argument("--rules-str", default=None, help="Optional inline translation rules/instructions")
    parser.add_argument(
        "--probe",
        "--num-messages",
        dest="num_messages",
        type=int,
        default=None,
        help="Limit how many translated messages are sent for checking (testing only)",
    )
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.translation-check.json)")
    parser.add_argument("--include-ok", action="store_true", help="Include entries with no issues in the output JSON")
    parser.add_argument("--max-attempts", type=int, default=5, help="Retry attempts per batch")
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser())


def run_from_args(args: argparse.Namespace) -> None:

    if args.max_attempts <= 0:
        sys.exit("ERROR: --max-attempts must be greater than 0")

    try:
        file_kind = detect_file_kind(args.file)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    if file_kind != FileKind.PO:
        sys.exit("ERROR: the check command currently supports only .po files")

    provider = get_translation_provider(args.provider)
    client = provider.create_client_from_env()

    resource_context = load_task_resource_context(
        target_lang=args.target_lang,
        explicit_vocab_path=args.vocab,
        explicit_rules_path=args.rules,
        inline_rules=args.rules_str,
        resolve_resource_path_fn=resolve_resource_path,
        read_optional_vocabulary_file_fn=read_optional_vocabulary_file,
        read_optional_text_file_fn=read_optional_text_file,
    )

    entries, _, _ = load_po(args.file)
    try:
        review_entries = limit_review_entries(
            select_review_entries(entries),
            args.num_messages,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    total = len(review_entries)
    if total == 0:
        print("No translated PO entries found to review.")
        return

    try:
        batch_size, parallel_requests, limits_mode = normalize_limits(
            total_items=total,
            batch_size_arg=args.batch_size,
            parallel_arg=args.parallel_requests,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    all_batches = build_fixed_batches(review_entries, batch_size)
    out_path = args.out or build_check_output_path(args.file)
    config = build_check_generation_config(
        args.thinking_level,
        provider=provider,
        system_instruction=build_check_system_instruction(args.target_lang),
    )

    print("Startup configuration:")
    print(f"  Provider: {provider.name}")
    print(f"  Model: {args.model}")
    print(f"  Thinking level: {args.thinking_level or 'provider default'}")
    print(f"  Parallel requests: {parallel_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limits mode: {limits_mode}")
    print(f"  Vocabulary source: {resource_context.vocabulary_source}")
    print(f"  Rules source: {resource_context.rules_source or 'none'}")
    print(f"  Probe limit: {args.num_messages if args.num_messages is not None else 'none'}")
    print(f"  Total translated entries: {total}")
    print(f"  Total batches: {len(all_batches)}")

    async def run_checks() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        completed = 0
        base_index = 0
        batch_start_indices: Dict[int, int] = {}
        for idx, batch in enumerate(all_batches):
            batch_start_indices[idx] = base_index
            base_index += len(batch)

        async def process_batch(batch_index: int, batch: List[Any]) -> Dict[str, List[CheckIssue]]:
            msg_map: Dict[str, Dict[str, Any]] = {}

            for i, entry in enumerate(batch):
                item_id = str(i)
                msg_map[item_id] = build_check_message_payload(entry)

            contents = build_check_request_contents(
                messages=msg_map,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                vocabulary=resource_context.vocabulary_text,
                translation_rules=resource_context.project_rules,
                provider=provider,
            )

            response = await generate_with_retry(
                provider=provider,
                client=client,
                model=args.model,
                contents=contents,
                batch_label=f"check batch {batch_index + 1}/{len(all_batches)}",
                max_attempts=args.max_attempts,
                config=config,
            )
            return parse_check_response(response)

        def on_batch_completed(
            batch_index: int,
            batch: List[Any],
            model_issues_by_id: Dict[str, List[CheckIssue]],
        ) -> None:
            nonlocal completed
            start_index = batch_start_indices[batch_index]

            for i, entry in enumerate(batch):
                item_id = str(i)
                issues = dedupe_issues(model_issues_by_id.get(item_id, []))
                if not issues and not args.include_ok:
                    continue

                result_payload = {
                    "entry_index": start_index + i,
                    "msgctxt": str(getattr(entry, "msgctxt", "") or ""),
                    "source": build_entry_source_text(entry),
                    "translation": get_entry_translation_text(entry),
                    "translation_plural_forms": get_translation_plural_forms(entry),
                    "flags": list(getattr(entry, "flags", []) or []),
                    "verdict": "issues" if issues else "ok",
                    "issues": [asdict(issue) for issue in issues],
                }
                results.append(result_payload)

            completed += 1
            issue_count = sum(len(item["issues"]) for item in results)
            print(
                f"Progress: completed batches {completed}/{len(all_batches)} "
                f"(latest: {batch_index + 1}/{len(all_batches)}), "
                f"reported issues: {issue_count}"
            )

        await run_parallel_batches(
            batches=all_batches,
            parallel_requests=parallel_requests,
            process_batch=process_batch,
            on_batch_completed=on_batch_completed,
        )

        results.sort(key=lambda item: int(item["entry_index"]))
        return results

    try:
        results = asyncio.run(run_checks())
    except RuntimeError as e:
        sys.exit(str(e))

    payload = {
        "source_file": args.file,
        "output_file": out_path,
        "provider": provider.name,
        "model": args.model,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "vocabulary_source": resource_context.vocabulary_source,
        "rules_source": resource_context.rules_source or "none",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_limit": args.num_messages,
        "total_entries_checked": total,
        "entries_with_issues": sum(1 for item in results if item["issues"]),
        "clean_entries": total - sum(1 for item in results if item["issues"]),
        "issue_count": sum(len(item["issues"]) for item in results),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nTranslation QA complete.")
    print(f"Saved report: {out_path}")
    print(f"Entries with issues: {payload['entries_with_issues']} / {total}")
    print(f"Total issues: {payload['issue_count']}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
