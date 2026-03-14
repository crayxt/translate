# Kazakh UI Translation Rules

Apply these rules for Kazakh software UI localization.
If any rule conflicts with the system MUST rules, the system MUST rules win.

## Script
- Use real Kazakh Cyrillic.
- Do not use Latin transliteration or mixed-script lookalikes.
- Do not transliterate blindly when a standard Kazakh localized term already exists.
- If no good Kazakh equivalent exists, use a standard technical borrowing consistently.

## Grammar and Word Order
- Prefer natural Kazakh word order and avoid English/Russian calques.
- Use natural Kazakh SOV structure in full sentences where appropriate.
- Keep modifiers before the noun where natural.
- Keep participial or relative modifiers before the noun they modify.
- Use postpositions after noun phrases, not preposition-style calques.
- Use natural Kazakh possessive and genitive constructions.

## Questions
- Rephrase prompts into proper natural Kazakh question forms before finalizing.
- For yes/no questions, use interrogative particles `ма/ме/ба/бе/па/пе` naturally as separate words after the predicate or focus word.
- Do not force interrogative particles in wh-questions that already contain words such as `кім`, `не`, `қайда`, `қашан`, `неге`, `қалай`.

## UI Wording and Tone
- Default to neutral action style for labels and menu items: verbal noun or infinitive style.
- Prefer forms like `Бетті жабу`, `Файлды ашу`, `Параметрлерді сақтау` over polite imperative forms in typical UI commands.
- Use polite imperative only when the source clearly addresses the user directly, such as in instructions or warnings.
- Keep tone consistent within the same screen or module.
- Keep Kazakh UI text concise and natural.
- Keep button and menu labels short action phrases.

## Number and Morphology
- After numerals, prefer natural Kazakh noun forms, usually the singular base form where appropriate in UI usage.
- For plural-capable messages, use natural Kazakh wording and avoid forced literal plural calques.
- Keep case endings correct for object, direction, location, source, and other grammatical roles.

## Typography
- When quotes are needed, prefer `«...»`.
- For nested quotes, use `„...“` inside `«...»`.

## Accelerators
- For accelerators in patterns like `_Apply` or `&Apply`, assign the accelerator to a natural Cyrillic letter.
- Avoid using Kazakh-specific letters `ә, і, ң, ғ, ү, ұ, қ, ө, һ` for accelerator assignment where a more practical Cyrillic alternative is available.

## Meaning
- Choose the most neutral reusable Kazakh UI translation when the source is ambiguous.
