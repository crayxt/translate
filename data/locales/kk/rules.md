# Kazakh UI Translation Rules

Use these rules for Kazakh software UI localization.
These rules are language and style policy only.
System MUST rules cover structural constraints such as placeholders, tags, protected tokens, and formatting markers.
Approved glossary terms are mandatory; these rules only guide Kazakh wording around them.

## Core Policy
- Prefer standard Kazakh UI wording over literal English or Russian calques.
- Prefer one stable translation for the same UI concept within the same product.
- Write in real Kazakh Cyrillic. Do not use Latin transliteration or mixed-script lookalikes.
- If no good Kazakh equivalent exists, use a widely understood technical borrowing consistently.
- When the source is ambiguous, choose the most neutral and reusable UI meaning.

## Labels and Commands
- For menu items, buttons, and action labels, prefer neutral action wording: verbal noun or infinitive style.
- Prefer forms like `Бетті жабу`, `Файлды ашу`, `Параметрлерді сақтау` for standard UI commands.
- If the source is a noun label, keep it a noun label rather than turning it into a command.
- Keep labels short, natural, and reusable across the UI.
- Use polite imperative only when the source clearly addresses the user directly.

## Questions and Messages
- Rewrite prompts as natural Kazakh questions rather than word-for-word calques.
- For yes/no questions, use `ма/ме/ба/бе/па/пе` naturally as separate particles after the predicate or focus word.
- Do not add interrogative particles to wh-questions that already contain `кім`, `не`, `қайда`, `қашан`, `неге`, or `қалай`.
- Prefer clear neutral phrasing for warnings, confirmations, and status messages.
- For confirmation-style prompts about user actions, settings, permissions, replacement, deletion, hiding, restarting, exiting, and similar decisions, prefer a natural active question built around the action the user may take.
- In such prompts, prefer patterns like `<object> <verb>-у керек пе?` or another equally natural active confirmation form when that fits the UI context.
- Avoid passive or bureaucratic calques such as `өшірілсін бе?`, `алмастырылсын ба?`, `жасырылсын ба?`, `қайта іске қосылсын ба?`, `қосылсын ба?` when the UI is really asking the user whether to perform an action.
- Prefer `«%1$s» альбомын өшіру керек пе?`, `«%1$s» ойнату тізімін алмастыру керек пе?`, `Қолданбаны қайта іске қосу керек пе?` over passive variants like `«%1$s» альбомы өшірілсін бе?`.
- When an item name appears in a confirmation question, inflect that item naturally for the intended action, especially the accusative where appropriate.
- For English prompts like `Are you sure you want to ...?`, prefer a concise natural Kazakh confirmation question centered on the action itself rather than a literal translation of `Are you sure`.

## Grammar and Morphology
- Prefer natural Kazakh word order and avoid Russian-style preposition calques.
- Use natural case endings and postpositions.
- After numerals, prefer natural Kazakh noun usage, usually the singular base form where appropriate in UI text.
- For plural-capable messages, use natural Kazakh wording rather than literal plural calques.
- If a glossary term needs inflection, inflect it naturally without replacing it with a synonym.

## Typography and Accelerators
- When quotes are needed, prefer `«...»`; for nested quotes use `„...“` inside `«...»`.
- For accelerators like `_Apply` or `&Apply`, assign the mnemonic to a natural Cyrillic letter that remains practical on common keyboard layouts.
- Avoid using `ә, і, ң, ғ, ү, ұ, қ, ө, һ` for accelerators when a simpler Cyrillic alternative is available.
