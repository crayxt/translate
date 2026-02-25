# Kazakh Translation Rules

Apply these project rules for all UI localization.
If any rule conflicts with core STRICT RULES in `process.py`, STRICT RULES win.

## 1) Safety and Preservation
- Keep placeholders exactly as-is: `%s`, `%d`, `%(name)s`, `{name}`, `{{name}}`.
- Preserve HTML/XML tags, escapes, entities, and line breaks exactly.
- Preserve leading/trailing spaces exactly.
- Preserve keyboard accelerators and hotkeys (`_`, `&`) and keep them usable/unique in menus.
- For accelerators in patterns like `_Apply`/`&Apply`, do not assign the accelerator to Kazakh-specific letters: `ә, і, ң, ғ, ү, ұ, қ, ө, һ`.

## 2) Terminology Consistency
- Follow the project vocabulary file exactly for mapped terms.
- Use one canonical term per concept across the whole product.
- Do not alternate synonyms for the same core UI action.
- If no good Kazakh equivalent exists, use a standard technical borrowing consistently.

## 3) Kazakh Grammar and Word Order
- Use natural Kazakh SOV order in full sentences (main verb near sentence end).
- Use modifier-before-noun order in noun phrases.
- Keep participial/relative modifiers before the noun they modify.
- Use postpositions after noun phrases (not preposition-style order).
- Use natural Kazakh possessive/genitive construction (possessor before possessed).
- Avoid English/Russian word-order calques.

## 4) Verb Form and Tone
- Default to neutral action style for labels/menu items: verbal noun/infinitive.
- Prefer `Бетті жабу`, `Файлды ашу`, `Параметрлерді сақтау` over polite imperative forms.
- Prefer `Бетті жабу` instead of `Бетті жабыңыз` in most UI command labels.
- Use polite imperative only when source clearly addresses the user directly (instruction/warning).
- Keep tone consistent within one screen/module; do not mix styles arbitrarily.

## 5) Number, Plural, and Morphology
- After numerals, keep noun form natural for Kazakh UI usage (usually singular base form).
- For plural-capable messages, use natural Kazakh forms and avoid forced literal plural calques.
- Keep case endings correct for object, direction, location, and source roles.

## 6) UI Style and Brevity
- Use concise UI language; avoid unnecessary explanatory additions.
- Keep sentence case unless source is ALL CAPS.
- Keep punctuation and emphasis close to source (`:`, `...`, `?`, `!`) unless grammar requires small adjustments.
- Use guillemet quotes `«...»` for quoted text when quotes are needed.
- For nested quotes, use `„...“` inside `«...»`; two levels are enough.
- Keep button/menu labels short action phrases; avoid long full sentences in compact UI controls.

## 7) Non-Translatable and Protected Content
- Do not translate product names, brand names, API names, code identifiers, command flags, or file extensions.
- Do not alter technical tokens, paths, or variable-like strings.

## 8) Meaning Fidelity
- Do not add/remove meaning.
- Do not soften/intensify modality (`must`, `cannot`, `optional`) unless source does.
- If source is ambiguous, choose the most neutral reusable UI translation.

## 9) Prohibited Patterns
- Do not add words not present in source for explanation.
- Do not transliterate blindly when a standard localized term already exists.
