from __future__ import annotations

import json
from typing import Any, Callable, Tuple


def json_load_maybe(text: str) -> Any:
    payload = (text or "").strip()
    if not payload:
        return None

    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def plural_key_sort_key(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def build_target_script_guidance(
    target_lang: str,
    *,
    update_wording: Callable[[], str] | None = None,
) -> str | None:
    normalized = str(target_lang or "").strip().lower().replace("-", "_")
    if not normalized:
        return None

    action_phrase = update_wording() if update_wording is not None else "target text"
    if normalized == "kk" or normalized.startswith("kk_"):
        return (
            f"For Kazakh, write {action_phrase} in the real Kazakh Cyrillic alphabet. "
            "Do not use Latin transliteration or Latin lookalikes."
        )

    return (
        f"When you provide {action_phrase}, use the real writing system of the target "
        "language, not transliteration."
    )
