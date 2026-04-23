# Kazakh UI Translation Rules

Use these rules for Kazakh software UI localization.
These rules govern Kazakh wording, grammar, register, and typography.
System MUST rules control structural constraints such as placeholders, tags, protected tokens, escapes, and formatting markers.
Approved glossary terms are mandatory. Apply these rules around approved terms, not instead of them.

## Priority Order
- If an approved glossary term matches the intended sense, use it.
- If no approved term exists, prefer established natural Kazakh UI wording over literal English or Russian calques.
- If no good natural Kazakh equivalent exists, use a widely understood technical borrowing consistently.
- Do not invent ad hoc transliterations, mixed-script spellings, or unstable synonyms for the same UI concept.

## Core Policy
- Prefer one stable translation for the same UI concept within the same product.
- Write in real Kazakh Cyrillic. Do not use Latin transliteration or mixed-script lookalikes.
- When the source is ambiguous, choose the most neutral, reusable, and UI-appropriate meaning.
- Prefer wording that sounds like natural software UI language, not like bureaucratic, academic, or machine-translated prose.

## Labels and Commands
- Distinguish UI roles before choosing the form.
- For commands and actions such as menu items, buttons, and toolbar actions, prefer neutral action wording, usually verbal noun or infinitive style.
- Prefer forms like `Бетті жабу`, `Файлды ашу`, `Параметрлерді сақтау` for standard UI commands.
- For section names, categories, navigation labels, settings names, and other noun labels, keep them as noun labels rather than turning them into commands.
- Keep labels short, natural, and reusable across the UI.
- Use polite imperative only when the source clearly addresses the user directly and the product voice genuinely requires it.

## Questions and Messages
- Rewrite prompts as natural Kazakh questions rather than word-for-word calques.
- Prefer clear, neutral phrasing for warnings, confirmations, and status messages.
- For yes/no questions, use `ма/ме/ба/бе/па/пе` naturally as separate particles after the predicate or focus word.
- Do not add interrogative particles to wh-questions that already contain `кім`, `не`, `қайда`, `қашан`, `неге`, or `қалай`.
- For confirmation prompts about user actions, settings, permissions, replacement, deletion, hiding, restarting, exiting, and similar decisions, prefer a natural active question built around the action the user may take.
- Prefer active confirmation patterns such as `<object> <verb>-у керек пе?` when they fit the UI context naturally, but do not force this exact form if a shorter active wording is more idiomatic.
- Avoid passive or bureaucratic calques such as `өшірілсін бе?`, `алмастырылсын ба?`, `жасырылсын ба?`, `қайта іске қосылсын ба?`, `қосылсын ба?` when the UI is really asking the user whether to perform an action.
- Prefer `«%1$s» альбомын өшіру керек пе?`, `«%1$s» ойнату тізімін алмастыру керек пе?`, `Қолданбаны қайта іске қосу керек пе?` over passive variants like `«%1$s» альбомы өшірілсін бе?`.
- When an item name appears in a confirmation question, inflect that item naturally for the intended action, especially the accusative where appropriate.
- For English prompts like `Are you sure you want to ...?`, prefer a concise natural Kazakh confirmation question centered on the action itself rather than a literal translation of `Are you sure`.

## Register and Person
- Prefer neutral UI register over conversational filler.
- Avoid explicit second-person pronouns unless the source or product voice clearly addresses the user directly.
- Do not add extra politeness, emphasis, or reassurance that is not present in the source.

## Grammar and Morphology
- Prefer natural Kazakh word order and avoid Russian-style preposition calques.
- Build neutral UI clauses in natural Kazakh flow: object/arguments + complements + action adverbs + main verb (predicate-final tendency).
- Keep action adverbs (for example `тікелей`, `автоматты түрде`, `жылдам`) attached to the predicate phrase near the main verb, not inside noun/complement groups, unless the source clearly modifies a noun.
- Before finalizing, do a natural-flow check: if adverb placement sounds like a literal calque, move the adverb to the verb phrase while preserving meaning.
- Use natural case endings and postpositions.
- After numerals, prefer natural Kazakh noun usage, usually the singular base form where appropriate in UI text.
- For plural-capable messages, use natural Kazakh wording rather than literal plural calques.
- If an approved glossary term needs inflection, inflect it naturally instead of replacing it with a synonym.

## Typography and Accelerators
- When quotes are needed, prefer `«...»`; for nested quotes use `„...“` inside `«...»`.
- If the product or platform clearly expects another quote style, follow the product convention consistently.
- For accelerators like `_Apply` or `&Apply`, assign the mnemonic to a natural Cyrillic letter that remains practical on common keyboard layouts.
- Avoid using `ә, і, ң, ғ, ү, ұ, қ, ө, һ` for accelerators when a simpler Cyrillic alternative is available.
