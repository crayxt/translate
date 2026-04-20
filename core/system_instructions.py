from __future__ import annotations


def render_instruction_section(title: str, *lines: str) -> str:
    """Render a titled prompt section with one bullet per line."""
    cleaned_title = str(title or "").strip()
    cleaned_lines = [str(line or "").strip() for line in lines if str(line or "").strip()]
    if not cleaned_lines:
        return cleaned_title
    body = "\n".join(f"- {line}" for line in cleaned_lines)
    return f"{cleaned_title}:\n{body}" if cleaned_title else body


SHARED_LOCALIZATION_INVARIANTS = render_instruction_section(
    "MANDATORY LOCALIZATION INVARIANTS",
    "Placeholders must be preserved exactly (%s, %d, %(name)s, %1, %n, {var}, {{var}})",
    "HTML/XML tags must be preserved exactly and remain well-formed",
    "Keyboard accelerators/hotkeys must be preserved exactly and remain usable (`_`, `&`)",
    "Leading and trailing spaces must be preserved exactly",
    "Escapes, entities, and meaningful punctuation must be preserved when they carry formatting or structure",
    "Literal escape sequences such as `\\n` and `\\t` must remain literal backslash sequences when the source uses them",
    "Never output NUL bytes or hidden control characters in translated text",
    "Approved vocabulary and project rules are mandatory when supplied",
    "Use message context and notes to disambiguate meaning, UI role, and terminology whenever they are present",
)


SHARED_GLOSSARY_SENSE_RULES = render_instruction_section(
    "AMBIGUOUS TERMS AND GLOSSARY MATCHING",
    "Determine the intended sense of the source text from the full message, UI role, context, note, surrounding words, punctuation, and placeholders",
    "Treat approved glossary entries as applicable only when their meaning matches the current message",
    "Do not rely on source-token overlap alone when selecting, revising, or evaluating terminology",
    "If multiple glossary entries match the same source term, use the one whose meaning, part of speech, and context best fit the message",
    "If no glossary entry fits the intended sense, do not force an inappropriate glossary term",
)


def join_instruction_sections(*sections: str) -> str:
    """Join prompt sections with stable blank-line spacing."""
    return "\n\n".join(section.strip() for section in sections if str(section or "").strip())


__all__ = [
    "render_instruction_section",
    "SHARED_LOCALIZATION_INVARIANTS",
    "SHARED_GLOSSARY_SENSE_RULES",
    "join_instruction_sections",
]
