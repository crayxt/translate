# Kazakh UI Translation Rules

Use these rules for Kazakh software UI localization.
Apply them after the system MUST rules and the approved vocabulary.
If a rule conflicts with the system MUST rules or approved glossary terms, the system MUST rules and glossary win.

## Priority
- Prefer standard Kazakh UI wording over literal English or Russian calques.
- Prefer one stable translation for the same UI concept within the same product.
- If no standard Kazakh equivalent exists, use a widely understood technical borrowing consistently.

## Script and Alphabet
- Write in real Kazakh Cyrillic only.
- Do not mix Latin lookalikes with Cyrillic.
- Do not transliterate a term if a standard Kazakh UI equivalent already exists.
- Keep brand names, product names, API names, paths, flags, and code-like tokens unchanged unless the source itself localizes them.

## Commands, Labels, and Tone
- For menu items, buttons, and action labels, prefer neutral action wording: verbal noun or infinitive style.
- Prefer forms like `Бетті жабу`, `Файлды ашу`, `Параметрлерді сақтау` for standard UI commands.
- Use polite imperative only when the source clearly addresses the user directly, such as in instructions, warnings, or guided prompts.
- Keep labels short, natural, and reusable across the UI.
- If the source is a noun label, keep it a noun label rather than turning it into a command.

## Questions and Messages
- Rewrite prompts as natural Kazakh questions rather than word-for-word calques.
- For yes/no questions, use `ма/ме/ба/бе/па/пе` naturally as separate particles after the predicate or focus word.
- Do not add interrogative particles to wh-questions that already contain `кім`, `не`, `қайда`, `қашан`, `неге`, or `қалай`.
- For warnings, confirmations, and status messages, prefer clear neutral phrasing over conversational tone.

## Grammar and Morphology
- Prefer natural Kazakh word order.
- Keep modifiers and participial phrases in natural pre-nominal position when needed.
- Use postpositions and case endings naturally; avoid Russian-style preposition calques.
- After numerals, prefer natural Kazakh noun usage, usually the singular base form where appropriate in UI text.
- For plural-capable messages, choose natural Kazakh wording rather than literal plural calques.

## Terminology
- Follow the approved glossary exactly; inflect glossary terms naturally when grammar requires it.
- Do not replace approved terms with near-synonyms just for style.
- When the source is ambiguous, choose the most neutral and reusable UI meaning.

## Typography and Accelerators
- When quotes are needed, prefer `«...»`; for nested quotes use `„...“` inside `«...»`.
- For accelerators like `_Apply` or `&Apply`, assign the mnemonic to a natural Cyrillic letter that remains practical on common keyboard layouts.
- Avoid using `ә, і, ң, ғ, ү, ұ, қ, ө, һ` for accelerators when a simpler Cyrillic alternative is available.
