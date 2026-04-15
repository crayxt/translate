from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Tuple

from core.entries import get_entry_prompt_context_and_note
from core.resources import parse_vocabulary_fields


REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DATA_ROOT = REPO_ROOT / "data" / "extract"

# Unicode-aware tokenization so extraction works on non-English sources too.
# Keep underscores inside one token so mnemonic-marked or identifier-like strings
# do not get split into false sub-terms such as `fra_me` -> `me`.
# Preserve a leading `%` or `-` so placeholder-like tokens (`%PRODUCTNAME`) and
# option-like tokens (`-help`, `--verbose`) can be rejected before extraction.
TOKEN_RE = re.compile(r"[%\-]?[^\W]+(?:[-'][^\W]+)*", re.UNICODE)
ACCELERATOR_AMPERSAND_PREFIX_RE = re.compile(r"(^|[\s([{<])&(?=\w)", re.UNICODE)
ACCELERATOR_AMPERSAND_INLINE_RE = re.compile(r"(?<=\w)&(?=[a-z])", re.UNICODE)
ACCELERATOR_UNDERSCORE_PREFIX_RE = re.compile(r"(^|[\s([{<])_(?=\w)", re.UNICODE)
ACCELERATOR_UNDERSCORE_INLINE_RE = re.compile(r"(?<=\w)_(?=\w)", re.UNICODE)
XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
XML_ATTR_TAG_RE = re.compile(
    r"<[A-Za-z][^>]*\b[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)[^>]*>",
    re.UNICODE,
)
XML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>", re.UNICODE)
ATTRIBUTE_ASSIGNMENT_RE = re.compile(
    r"\b[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.UNICODE,
)
URL_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9+.-]*:(?:\/\/)?|www\.)[^\s<>()\"']+",
    re.IGNORECASE | re.UNICODE,
)
URL_NOISE_SENTINEL = " -urlnoise "
URL_NOISE_SEQUENCE_RE = re.compile(r"(?:\s*-urlnoise\s*)+", re.IGNORECASE)


def normalize_space(text: str | None) -> str:
    """Collapse whitespace for stable comparisons and output."""
    return " ".join(str(text or "").split())


def load_extract_word_set(*relative_parts: str) -> set[str]:
    """Load a normalized extractor word list from a repo-relative text file."""
    path = EXTRACT_DATA_ROOT.joinpath(*relative_parts)
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.add(normalize_space(stripped).lower())
    return values


STOP_WORDS = load_extract_word_set("en", "stopwords.txt")
LOW_VALUE_SINGLE_WORDS = STOP_WORDS | load_extract_word_set("en", "low_value_words.txt")
FIXED_MULTIWORD_ALLOWLIST = load_extract_word_set("en", "fixed_multiword_allowlist.txt")
EXCLUDED_FUNCTION_NAMES = load_extract_word_set("common", "function_names.txt")
EXCLUDED_SOURCE_TERMS = (
    load_extract_word_set("common", "abbreviations.txt")
    | load_extract_word_set("common", "excluded_terms.txt")
    | EXCLUDED_FUNCTION_NAMES
)
EXCLUDED_MULTIWORD_SOURCE_PATTERNS = tuple(
    re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    for term in sorted(
        (term for term in EXCLUDED_SOURCE_TERMS if " " in term),
        key=len,
        reverse=True,
    )
)

DiscoveryMode = Literal["all", "missing"]
Decision = Literal["accepted", "borderline", "rejected"]
LOCATION_NOTE_PREFIX_RE = re.compile(r"(?:^|\s)locations:\s*(.+)$", re.IGNORECASE)
LOCATION_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+:\d+")
STRUCTURAL_CONTEXT_RE = re.compile(r"^line:\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceMessage:
    """A normalized source message used by local term extraction."""

    source: str
    context: str = ""
    note: str = ""
    source_file: str = ""


@dataclass
class CandidateEvidence:
    """Aggregated evidence and decision data for one extracted term candidate."""

    source_term: str
    occurrence_count: int = 0
    message_count: int = 0
    exact_source_match_count: int = 0
    context_diversity: int = 0
    file_count: int = 0
    location_file_count: int = 0
    location_scope_count: int = 0
    examples: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    location_files: List[str] = field(default_factory=list)
    location_scopes: List[str] = field(default_factory=list)
    surface_forms: List[str] = field(default_factory=list)
    known_translation: str = ""
    accepted: bool = False
    decision: Decision = "rejected"
    score: int = 0
    reasons: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Final local extraction result split by accepted/borderline/rejected classes."""

    accepted_terms: List[CandidateEvidence]
    borderline_terms: List[CandidateEvidence]
    rejected_terms: List[CandidateEvidence]


@dataclass
class _CandidateAccumulator:
    """Mutable accumulator used while merging evidence from multiple source messages."""

    source_term: str
    occurrence_count: int = 0
    exact_source_match_count: int = 0
    examples: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    context_keys: set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    note_keys: set[str] = field(default_factory=set)
    files: List[str] = field(default_factory=list)
    file_keys: set[str] = field(default_factory=set)
    location_files: List[str] = field(default_factory=list)
    location_file_keys: set[str] = field(default_factory=set)
    location_scopes: List[str] = field(default_factory=list)
    location_scope_keys: set[str] = field(default_factory=set)
    surface_forms: List[str] = field(default_factory=list)
    message_keys: set[str] = field(default_factory=set)
    known_translation: str = ""


@dataclass(frozen=True)
class ScopedVocabularyEntry:
    """One approved glossary entry prepared for fast source-text matching."""

    source_term: str
    target_term: str
    part_of_speech: str = ""
    context_note: str = ""
    matcher: re.Pattern[str] = field(repr=False, compare=False, default=None)  # type: ignore[assignment]


def validate_max_length(max_length: int) -> int:
    """Validate the maximum n-gram length supported by extraction."""
    if max_length not in (1, 2, 3):
        raise ValueError("max_length must be 1, 2, or 3.")
    return max_length


def strip_ui_accelerators(text: str | None) -> str:
    """Remove common UI accelerator markers while preserving literal &&."""
    normalized = normalize_space(text)
    if "&" in normalized:
        amp_sentinel = "\u0000AMP\u0000"
        normalized = normalized.replace("&&", amp_sentinel)
        normalized = ACCELERATOR_AMPERSAND_PREFIX_RE.sub(r"\1", normalized)
        normalized = ACCELERATOR_AMPERSAND_INLINE_RE.sub("", normalized)
        normalized = normalized.replace(amp_sentinel, "&")

    if "_" in normalized:
        underscore_sentinel = "\u0000UNDERSCORE\u0000"
        normalized = normalized.replace("__", underscore_sentinel)
        normalized = ACCELERATOR_UNDERSCORE_PREFIX_RE.sub(r"\1", normalized)
        normalized = ACCELERATOR_UNDERSCORE_INLINE_RE.sub("", normalized)
        normalized = normalized.replace(underscore_sentinel, "_")

    return normalized


def strip_markup_and_url_noise(text: str | None) -> str:
    """Remove XML/HTML tags and replace URL-like payloads with a safe sentinel."""
    normalized = normalize_space(text)
    if not normalized:
        return ""
    normalized = XML_COMMENT_RE.sub(" ", normalized)
    normalized = XML_ATTR_TAG_RE.sub(URL_NOISE_SENTINEL, normalized)
    normalized = ATTRIBUTE_ASSIGNMENT_RE.sub(URL_NOISE_SENTINEL, normalized)
    normalized = URL_RE.sub(URL_NOISE_SENTINEL, normalized)
    normalized = XML_TAG_RE.sub(" ", normalized)
    normalized = URL_NOISE_SEQUENCE_RE.sub(URL_NOISE_SENTINEL, normalized)
    return normalize_space(normalized)


def normalize_ui_source_text(text: str | None) -> str:
    """Normalize UI text for extraction and strip excluded multiword source terms."""
    normalized = strip_ui_accelerators(strip_markup_and_url_noise(text))
    for pattern in EXCLUDED_MULTIWORD_SOURCE_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    return normalize_space(normalized)


def normalize_candidate_key(text: str | None) -> str:
    """Build a lowercase comparison key for candidate terms."""
    cleaned = normalize_ui_source_text(text).strip(".,:;!?\"'()[]{}")
    return cleaned.lower()


def strip_candidate_possessive_suffix(token: str) -> str:
    """Remove a trailing possessive suffix without changing letter case."""
    if token.endswith("'s") or token.endswith("’s"):
        return token[:-2]
    if token.endswith("'") or token.endswith("’"):
        return token[:-1]
    return token


def is_distinct_all_caps_token(token: str) -> bool:
    """Treat pure all-caps terms as a separate candidate family."""
    compact = strip_candidate_possessive_suffix(token).replace("-", "")
    letters = [char for char in compact if char.isalpha()]
    if not letters:
        return False
    return all(char.isupper() for char in letters)


def normalize_candidate_identity_token(token: str) -> str:
    """Normalize one token while preserving distinct all-caps identifiers."""
    cleaned = token.strip(".,:;!?\"'()[]{}")
    if not cleaned:
        return ""
    if is_distinct_all_caps_token(cleaned):
        return cleaned
    return cleaned.lower()


def normalize_candidate_identity_key(text: str | None) -> str:
    """Build a stable candidate key that keeps single-word all-caps terms distinct."""
    cleaned = normalize_ui_source_text(text).strip(".,:;!?\"'()[]{}")
    if not cleaned:
        return ""
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) == 1:
        return normalize_candidate_identity_token(tokens[0])
    return " ".join(token.lower() for token in tokens)


def normalize_vocabulary_match_text(text: str | None) -> str:
    """Normalize text for glossary matching without stripping excluded phrases."""
    return normalize_space(strip_ui_accelerators(text)).lower()


def tokenize_source_text(text: str) -> List[str]:
    """Tokenize normalized UI text into lowercase word-like units."""
    return [token.lower() for token in TOKEN_RE.findall(normalize_ui_source_text(text))]


def tokenize_candidate_source_text(text: str) -> List[str]:
    """Tokenize normalized UI text into candidate units for exact whole-label checks."""
    raw_tokens = TOKEN_RE.findall(normalize_ui_source_text(text))
    if len(raw_tokens) == 1:
        normalized = normalize_candidate_identity_token(raw_tokens[0])
        return [normalized] if normalized else []
    return [token.lower() for token in raw_tokens]


def is_placeholder_like(term: str) -> bool:
    """Reject terms that look like placeholders, markup, or protected syntax."""
    if not term:
        return True
    if "%" in term or "{" in term or "}" in term or "<" in term or ">" in term:
        return True
    if "&" in term or "_" in term:
        return True
    return False


def has_disallowed_term_prefix(term: str) -> bool:
    """Reject tokens that look like CLI options, macros, or number-led labels."""
    if not term:
        return False
    return term.startswith("%") or term.startswith("-") or term[0].isdigit()


def is_excluded_source_term(term: str) -> bool:
    """Filter out known brands, projects, protocols, and abbreviations."""
    normalized = normalize_candidate_key(term)
    if not normalized:
        return True
    return normalized in EXCLUDED_SOURCE_TERMS


def is_valid_single_token(token: str) -> bool:
    """Validate a single token as a possible term candidate."""
    if not token:
        return False
    normalized = normalize_candidate_key(token)
    if has_disallowed_term_prefix(token):
        return False
    if is_excluded_source_term(token):
        return False
    if normalized in LOW_VALUE_SINGLE_WORDS:
        return False
    if token.isdigit():
        return False
    if len(token) <= 1:
        return False
    return not is_placeholder_like(token)


def is_valid_phrase_tokens(tokens: List[str]) -> bool:
    """Validate a 2- or 3-word phrase before counting it as a candidate."""
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    phrase = " ".join(tokens)
    if phrase in FIXED_MULTIWORD_ALLOWLIST:
        return True
    if is_excluded_source_term(phrase):
        return False
    first_token = normalize_candidate_key(tokens[0])
    last_token = normalize_candidate_key(tokens[-1])
    if first_token in STOP_WORDS or last_token in STOP_WORDS:
        return False
    return all(is_valid_single_token(token) for token in tokens)


def should_include_phrase(
    phrase_tokens: List[str],
    *,
    max_length: int,
) -> bool:
    """Decide whether a valid phrase fits within the current max length."""
    return len(phrase_tokens) <= max_length and is_valid_phrase_tokens(phrase_tokens)


def extract_message_candidate_counts(
    source_text: str,
    *,
    max_length: int = 1,
) -> dict[str, int]:
    """Count candidate terms from one message up to the requested n-gram length."""
    max_length = validate_max_length(max_length)
    raw_tokens = TOKEN_RE.findall(normalize_ui_source_text(source_text))
    tokens = [token.lower() for token in raw_tokens]
    counts: dict[str, int] = {}

    for raw_token, token in zip(raw_tokens, tokens):
        candidate_token = raw_token if is_distinct_all_caps_token(raw_token) else token
        if is_valid_single_token(candidate_token):
            counts[candidate_token] = counts.get(candidate_token, 0) + 1

    if max_length == 1:
        return counts

    for size in range(2, max_length + 1):
        for index in range(0, len(tokens) - size + 1):
            phrase_tokens = tokens[index : index + size]
            if not should_include_phrase(phrase_tokens, max_length=max_length):
                continue
            phrase = " ".join(phrase_tokens)
            counts[phrase] = counts.get(phrase, 0) + 1

    return counts


def strip_english_possessive(token: str) -> str:
    """Remove a trailing English possessive marker from a token."""
    lower = token.lower()
    if lower.endswith("'s") or lower.endswith("’s"):
        return lower[:-2]
    if lower.endswith("'") or lower.endswith("’"):
        return lower[:-1]
    return lower


def singularize_english_token(token: str) -> str:
    """Apply a lightweight singularization rule for English nouns."""
    lower = token.lower()
    if len(lower) <= 3:
        return lower
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if any(lower.endswith(suffix) for suffix in ("ches", "shes", "sses", "xes", "zes")):
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        return lower[:-1]
    return lower


def build_vocabulary_matcher(source_term: str) -> re.Pattern[str]:
    """Build a regex matcher for one glossary source term."""
    normalized = normalize_vocabulary_match_text(source_term)
    escaped_parts = [re.escape(part) for part in normalized.split()]
    if not escaped_parts:
        return re.compile(r"$^")

    body = r"\s+".join(escaped_parts)
    compact = normalized.replace("-", "")
    if len(escaped_parts) == 1 and compact.isalpha():
        body = rf"{escaped_parts[0]}(?:'s|s)?"
    return re.compile(rf"(?<!\w){body}(?!\w)", re.UNICODE)


def build_scoped_vocabulary_entries(vocabulary_text: str | None) -> List[ScopedVocabularyEntry]:
    """Parse rich glossary text into matchable scoped vocabulary entries."""
    entries: List[ScopedVocabularyEntry] = []
    if not vocabulary_text:
        return entries

    for raw_line in vocabulary_text.splitlines():
        parsed = parse_vocabulary_fields(raw_line)
        if not parsed:
            continue
        source_term, target_term, part_of_speech, context_note = parsed
        entries.append(
            ScopedVocabularyEntry(
                source_term=source_term,
                target_term=target_term,
                part_of_speech=part_of_speech,
                context_note=context_note,
                matcher=build_vocabulary_matcher(source_term),
            )
        )

    entries.sort(key=lambda item: (-len(item.source_term.split()), -len(item.source_term), item.source_term.lower()))
    return entries


def build_relevant_vocabulary(
    source_text: str | None,
    scoped_vocabulary_entries: List[ScopedVocabularyEntry],
) -> List[dict[str, str]]:
    """Match the relevant subset of glossary entries for one source message."""
    searchable_source = normalize_vocabulary_match_text(source_text)
    if not searchable_source or not scoped_vocabulary_entries:
        return []

    relevant: List[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for item in scoped_vocabulary_entries:
        if not item.matcher.search(searchable_source):
            continue
        key = (item.source_term, item.target_term, item.part_of_speech, item.context_note)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        suggestion = {
            "source_term": item.source_term,
            "target_term": item.target_term,
        }
        if item.part_of_speech:
            suggestion["part_of_speech"] = item.part_of_speech
        if item.context_note:
            suggestion["context_note"] = item.context_note
        relevant.append(suggestion)
    return relevant


def parse_location_note(note: str) -> List[str]:
    """Extract normalized file:line locations from a note string."""
    normalized_note = normalize_space(note)
    match = LOCATION_NOTE_PREFIX_RE.search(normalized_note)
    if not match:
        return []

    matches: List[str] = []
    payload = match.group(1)
    for token in LOCATION_TOKEN_RE.findall(payload):
        normalized = token.replace("\\", "/")
        if normalized not in matches:
            matches.append(normalized)
    return matches


def build_location_scope(location: str) -> str:
    """Collapse a file location to a coarse module-like scope."""
    normalized = location.replace("\\", "/")
    file_path = normalized.split(":", 1)[0]
    parts = [part for part in file_path.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else ""


def build_source_message_key(
    source: str,
    context: str = "",
    note: str = "",
    source_file: str = "",
) -> str:
    """Build a stable dedupe key for a source message payload."""
    return "\u241f".join(
        (
            normalize_candidate_identity_key(source),
            normalize_space(context).lower(),
            normalize_space(note).lower(),
            normalize_space(source_file).replace("\\", "/").lower(),
        )
    )


def should_include_source_text(text: str) -> bool:
    """Skip empty or non-alphabetic strings before term extraction."""
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalpha() for ch in stripped)


def coerce_source_message(item: SourceMessage | Mapping[str, Any]) -> SourceMessage:
    """Normalize dict-like message payloads into SourceMessage objects."""
    if isinstance(item, SourceMessage):
        return item
    return SourceMessage(
        source=normalize_space(item.get("source", "")),
        context=normalize_space(item.get("context", "")),
        note=normalize_space(item.get("note", "")),
        source_file=normalize_space(item.get("source_file", "")).replace("\\", "/"),
    )


def build_source_messages_from_payloads(
    items: Iterable[SourceMessage | Mapping[str, Any]],
) -> List[SourceMessage]:
    """Normalize and dedupe incoming source messages."""
    results: List[SourceMessage] = []
    seen: set[str] = set()

    for raw_item in items:
        item = coerce_source_message(raw_item)
        if not should_include_source_text(item.source):
            continue
        key = build_source_message_key(item.source, item.context, item.note, item.source_file)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)

    return results


def collect_source_messages(entries: Iterable[Any], *, source_file: str = "") -> List[SourceMessage]:
    """Project localization entries into normalized source messages for local extraction."""
    payloads: List[dict[str, str]] = []

    def add_message(text: str, *, context: str | None, note: str | None) -> None:
        normalized = normalize_space(text)
        if not should_include_source_text(normalized):
            return
        payloads.append(
            {
                "source": normalized,
                "context": normalize_space(context),
                "note": normalize_space(note),
                "source_file": normalize_space(source_file).replace("\\", "/"),
            }
        )

    for entry in entries:
        if getattr(entry, "obsolete", False):
            continue
        if not getattr(entry, "include_in_term_extraction", True):
            continue
        context, note = get_entry_prompt_context_and_note(entry)
        add_message(getattr(entry, "msgid", "") or "", context=context, note=note)
        plural_text = getattr(entry, "msgid_plural", None)
        if plural_text:
            add_message(plural_text, context=context, note=note)

    return build_source_messages_from_payloads(payloads)


def is_meaningful_context(context: str) -> bool:
    """Ignore purely structural contexts like line numbers."""
    normalized = normalize_space(context)
    if not normalized:
        return False
    return STRUCTURAL_CONTEXT_RE.fullmatch(normalized) is None


def add_unique_limited(values: List[str], value: str, limit: int = 3) -> None:
    """Append a normalized value once while keeping lists compact."""
    cleaned = normalize_space(value)
    if not cleaned or cleaned in values:
        return
    if len(values) >= limit:
        return
    values.append(cleaned)


def canonicalize_candidate_key(candidate_key: str) -> str:
    """Apply conservative English canonicalization to a candidate key."""
    tokens = normalize_candidate_identity_key(candidate_key).split()
    if not tokens:
        return ""
    last_token = tokens[-1]
    if is_distinct_all_caps_token(last_token):
        normalized_last = strip_candidate_possessive_suffix(last_token)
    else:
        normalized_last = strip_english_possessive(last_token)
        normalized_last = singularize_english_token(normalized_last)
    return " ".join(tokens[:-1] + [normalized_last])


def build_vocabulary_translation_map(
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> Dict[str, str]:
    """Map normalized vocabulary source forms to known translations."""
    mapping: Dict[str, str] = {}
    for source, target in vocabulary_pairs or []:
        normalized_source = normalize_candidate_identity_key(source)
        normalized_target = normalize_space(target)
        if not normalized_source or not normalized_target:
            continue
        mapping.setdefault(normalized_source, normalized_target)
        canonical_source = canonicalize_candidate_key(normalized_source)
        if canonical_source:
            mapping.setdefault(canonical_source, normalized_target)
    return mapping


def build_vocabulary_exclusion_keys(
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> set[str]:
    """Build the set of vocabulary keys that should be excluded in missing mode."""
    keys: set[str] = set()
    for source_term, _target_term in vocabulary_pairs or []:
        normalized = normalize_candidate_identity_key(source_term)
        if not normalized:
            continue
        keys.add(normalized)
        canonical = canonicalize_candidate_key(normalized)
        if canonical:
            keys.add(canonical)
    return keys


def collect_raw_candidate_evidence(
    messages: List[SourceMessage],
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
    *,
    max_length: int = 1,
) -> Dict[str, _CandidateAccumulator]:
    """Collect per-message evidence before canonical term merging."""
    vocabulary_map = build_vocabulary_translation_map(vocabulary_pairs)
    evidence: Dict[str, _CandidateAccumulator] = {}

    for message in messages:
        message_key = build_source_message_key(
            message.source,
            message.context,
            message.note,
            message.source_file,
        )
        candidate_counts = extract_message_candidate_counts(
            message.source,
            max_length=max_length,
        )
        exact_candidate = normalize_candidate_identity_key(message.source)
        parsed_locations = parse_location_note(message.note)

        for term, count in candidate_counts.items():
            item = evidence.setdefault(
                term,
                _CandidateAccumulator(
                    source_term=term,
                    known_translation=vocabulary_map.get(term, ""),
                ),
            )
            item.occurrence_count += count
            item.message_keys.add(message_key)
            add_unique_limited(item.examples, message.source)
            if is_meaningful_context(message.context):
                add_unique_limited(item.contexts, message.context)
                item.context_keys.add(normalize_space(message.context).lower())
            if message.note:
                add_unique_limited(item.notes, message.note)
                item.note_keys.add(normalize_space(message.note).lower())
            if message.source_file:
                add_unique_limited(item.files, message.source_file, limit=10)
                item.file_keys.add(message.source_file)
            for location in parsed_locations:
                add_unique_limited(item.location_files, location, limit=10)
                item.location_file_keys.add(location)
                scope = build_location_scope(location)
                if scope:
                    add_unique_limited(item.location_scopes, scope, limit=10)
                    item.location_scope_keys.add(scope)
            add_unique_limited(item.surface_forms, term, limit=10)

        if exact_candidate and exact_candidate in candidate_counts:
            evidence[exact_candidate].exact_source_match_count += 1

    return evidence


def maybe_canonicalize_candidate(
    source_term: str,
    *,
    observed_terms: set[str],
    vocabulary_keys: set[str],
) -> str:
    """Collapse a candidate to an observed or known canonical variant when safe."""
    tokens = source_term.split()
    if not tokens:
        return source_term
    last_token = tokens[-1]
    if is_distinct_all_caps_token(last_token):
        normalized_last = strip_candidate_possessive_suffix(last_token)
    else:
        normalized_last = strip_english_possessive(last_token)
        normalized_last = singularize_english_token(normalized_last)
    if normalized_last == tokens[-1]:
        return source_term
    candidate = " ".join(tokens[:-1] + [normalized_last])
    if candidate in observed_terms or candidate in vocabulary_keys:
        return candidate
    return source_term


def build_candidate_alias_map(
    raw_terms: Iterable[str],
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
) -> Dict[str, str]:
    """Map raw observed terms to canonical term keys."""
    observed_terms = {
        normalize_candidate_identity_key(term)
        for term in raw_terms
        if normalize_candidate_identity_key(term)
    }
    vocabulary_keys = build_vocabulary_exclusion_keys(vocabulary_pairs)
    return {
        term: maybe_canonicalize_candidate(
            term,
            observed_terms=observed_terms,
            vocabulary_keys=vocabulary_keys,
        )
        for term in observed_terms
    }


def finalize_candidate_evidence(item: _CandidateAccumulator) -> CandidateEvidence:
    """Freeze mutable accumulator state into the public evidence dataclass."""
    return CandidateEvidence(
        source_term=item.source_term,
        occurrence_count=item.occurrence_count,
        message_count=len(item.message_keys),
        exact_source_match_count=item.exact_source_match_count,
        context_diversity=len(item.context_keys),
        file_count=len(item.file_keys),
        location_file_count=len(item.location_file_keys),
        location_scope_count=len(item.location_scope_keys),
        examples=list(item.examples),
        contexts=list(item.contexts),
        notes=list(item.notes),
        files=list(item.files),
        location_files=list(item.location_files),
        location_scopes=list(item.location_scopes),
        surface_forms=list(item.surface_forms),
        known_translation=item.known_translation,
    )


def collect_candidate_evidence(
    messages: List[SourceMessage],
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
    *,
    max_length: int = 1,
) -> Dict[str, CandidateEvidence]:
    """Collect and merge evidence into canonical candidate records."""
    raw_evidence = collect_raw_candidate_evidence(
        messages,
        vocabulary_pairs=vocabulary_pairs,
        max_length=max_length,
    )
    alias_map = build_candidate_alias_map(raw_evidence.keys(), vocabulary_pairs=vocabulary_pairs)
    vocabulary_map = build_vocabulary_translation_map(vocabulary_pairs)
    merged: Dict[str, _CandidateAccumulator] = {}

    for raw_term, raw_item in raw_evidence.items():
        canonical_term = alias_map.get(raw_term, raw_term)
        item = merged.setdefault(
            canonical_term,
            _CandidateAccumulator(
                source_term=canonical_term,
                known_translation=vocabulary_map.get(canonical_term, ""),
            ),
        )
        item.occurrence_count += raw_item.occurrence_count
        item.exact_source_match_count += raw_item.exact_source_match_count
        item.message_keys.update(raw_item.message_keys)
        item.context_keys.update(raw_item.context_keys)
        item.note_keys.update(raw_item.note_keys)
        item.file_keys.update(raw_item.file_keys)
        item.location_file_keys.update(raw_item.location_file_keys)
        item.location_scope_keys.update(raw_item.location_scope_keys)
        for value in raw_item.examples:
            add_unique_limited(item.examples, value)
        for value in raw_item.contexts:
            add_unique_limited(item.contexts, value)
        for value in raw_item.notes:
            add_unique_limited(item.notes, value)
        for value in raw_item.files:
            add_unique_limited(item.files, value, limit=10)
        for value in raw_item.location_files:
            add_unique_limited(item.location_files, value, limit=10)
        for value in raw_item.location_scopes:
            add_unique_limited(item.location_scopes, value, limit=10)
        for value in raw_item.surface_forms:
            add_unique_limited(item.surface_forms, value, limit=10)

    return {
        key: finalize_candidate_evidence(item)
        for key, item in merged.items()
    }


def build_strong_atomic_terms(
    evidence: Mapping[str, CandidateEvidence],
    vocabulary_keys: set[str],
) -> set[str]:
    """Identify strong one-word atoms used to penalize loose composed phrases."""
    terms: set[str] = set()

    for key in vocabulary_keys:
        if len(key.split()) == 1 and normalize_candidate_key(key) not in LOW_VALUE_SINGLE_WORDS:
            terms.add(key)

    for term, item in evidence.items():
        if len(term.split()) != 1:
            continue
        if normalize_candidate_key(term) in LOW_VALUE_SINGLE_WORDS or is_placeholder_like(term):
            continue
        if (
            item.exact_source_match_count > 0
            or item.message_count > 1
            or item.context_diversity > 0
            or item.location_scope_count > 0
        ):
            terms.add(term)

    return terms


def decide_candidate(
    item: CandidateEvidence,
    *,
    mode: DiscoveryMode,
    vocabulary_keys: set[str],
    strong_atomic_terms: set[str],
) -> CandidateEvidence:
    """Classify one candidate as accepted, borderline, or rejected."""
    decided = CandidateEvidence(
        source_term=item.source_term,
        occurrence_count=item.occurrence_count,
        message_count=item.message_count,
        exact_source_match_count=item.exact_source_match_count,
        context_diversity=item.context_diversity,
        file_count=item.file_count,
        location_file_count=item.location_file_count,
        location_scope_count=item.location_scope_count,
        examples=list(item.examples),
        contexts=list(item.contexts),
        notes=list(item.notes),
        files=list(item.files),
        location_files=list(item.location_files),
        location_scopes=list(item.location_scopes),
        surface_forms=list(item.surface_forms),
        known_translation=item.known_translation,
    )

    tokens = item.source_term.split()
    if not tokens or is_placeholder_like(item.source_term):
        decided.reasons.append("placeholder_or_empty")
        return decided

    if mode == "missing" and item.source_term in vocabulary_keys:
        decided.reasons.append("already_in_vocabulary")
        return decided

    if is_excluded_source_term(item.source_term):
        decided.reasons.append("explicit_excluded_source_term")
        return decided

    if len(tokens) == 1:
        token = tokens[0]
        if normalize_candidate_key(token) in LOW_VALUE_SINGLE_WORDS:
            decided.reasons.append("low_value_single_word")
            return decided

        if item.exact_source_match_count > 0:
            decided.score += 2
            decided.reasons.append("exact_ui_label")
        if item.message_count > 1:
            decided.score += 2
            decided.reasons.append("repeated_across_messages")
        if item.exact_source_match_count > 1:
            decided.score += 1
            decided.reasons.append("repeated_exact_label")
        if len(token) >= 4:
            decided.score += 1
            decided.reasons.append("content_word_candidate")
        if item.context_diversity > 0:
            decided.score += 1
            decided.reasons.append("has_meaningful_context")
        if item.context_diversity > 1:
            decided.score += 1
            decided.reasons.append("cross_context_usage")
        if item.notes:
            decided.score += 1
            decided.reasons.append("has_note_evidence")
        if item.location_file_count > 0:
            decided.score += 1
            decided.reasons.append("has_source_locations")
        if item.location_scope_count > 1:
            decided.score += 1
            decided.reasons.append("cross_module_usage")

        if decided.score >= 4:
            decided.accepted = True
            decided.decision = "accepted"
            return decided
        if decided.score >= 2:
            decided.decision = "borderline"
            decided.reasons.append("needs_review")
            return decided
        decided.reasons.append("insufficient_single_word_evidence")
        return decided

    if item.source_term in FIXED_MULTIWORD_ALLOWLIST:
        decided.score += 4
        decided.reasons.append("fixed_multiword_allowlist")
    if item.message_count > 1:
        decided.score += 2
        decided.reasons.append("repeated_phrase_across_messages")
    if item.exact_source_match_count > 1:
        decided.score += 1
        decided.reasons.append("repeated_exact_label")
    if item.context_diversity > 0:
        decided.score += 1
        decided.reasons.append("has_meaningful_context")
    if item.location_scope_count > 1:
        decided.score += 1
        decided.reasons.append("cross_module_usage")
    if item.notes:
        decided.score += 1
        decided.reasons.append("has_note_evidence")

    if (
        len(tokens) == 2
        and item.source_term not in FIXED_MULTIWORD_ALLOWLIST
        and all(token in strong_atomic_terms for token in tokens)
    ):
        decided.reasons.append("compositional_phrase")
        if decided.score >= 2:
            decided.decision = "borderline"
            decided.reasons.append("needs_review")
        else:
            decided.reasons.append("single_occurrence_multiword_phrase")
        return decided

    if decided.score >= 4:
        decided.accepted = True
        decided.decision = "accepted"
        return decided
    if decided.score >= 2:
        decided.decision = "borderline"
        decided.reasons.append("needs_review")
        return decided
    decided.reasons.append("single_occurrence_multiword_phrase")
    return decided


def sort_candidates(items: Iterable[CandidateEvidence]) -> List[CandidateEvidence]:
    """Sort candidates by strength first, then by stable lexical order."""
    return sorted(
        items,
        key=lambda item: (
            -item.score,
            -item.message_count,
            -item.occurrence_count,
            item.source_term,
        ),
    )


def extract_terms_locally(
    messages: Iterable[SourceMessage | Mapping[str, Any]],
    *,
    mode: DiscoveryMode = "missing",
    vocabulary_pairs: List[Tuple[str, str]] | None = None,
    max_length: int = 1,
) -> ExtractionResult:
    """Run the full local extraction pipeline over normalized source messages."""
    normalized_messages = build_source_messages_from_payloads(messages)
    evidence = collect_candidate_evidence(
        normalized_messages,
        vocabulary_pairs=vocabulary_pairs,
        max_length=max_length,
    )
    vocabulary_keys = build_vocabulary_exclusion_keys(vocabulary_pairs)
    strong_atomic_terms = build_strong_atomic_terms(evidence, vocabulary_keys)

    accepted: List[CandidateEvidence] = []
    borderline: List[CandidateEvidence] = []
    rejected: List[CandidateEvidence] = []
    for item in evidence.values():
        decided = decide_candidate(
            item,
            mode=mode,
            vocabulary_keys=vocabulary_keys,
            strong_atomic_terms=strong_atomic_terms,
        )
        if decided.accepted:
            accepted.append(decided)
        elif decided.decision == "borderline":
            borderline.append(decided)
        else:
            rejected.append(decided)

    return ExtractionResult(
        accepted_terms=sort_candidates(accepted),
        borderline_terms=sort_candidates(borderline),
        rejected_terms=sort_candidates(rejected),
    )


__all__ = [
    "EXCLUDED_SOURCE_TERMS",
    "FIXED_MULTIWORD_ALLOWLIST",
    "LOW_VALUE_SINGLE_WORDS",
    "STOP_WORDS",
    "CandidateEvidence",
    "Decision",
    "DiscoveryMode",
    "ExtractionResult",
    "ScopedVocabularyEntry",
    "SourceMessage",
    "build_candidate_alias_map",
    "build_location_scope",
    "build_relevant_vocabulary",
    "build_scoped_vocabulary_entries",
    "collect_source_messages",
    "build_source_message_key",
    "build_source_messages_from_payloads",
    "build_strong_atomic_terms",
    "build_vocabulary_exclusion_keys",
    "build_vocabulary_translation_map",
    "build_vocabulary_matcher",
    "canonicalize_candidate_key",
    "coerce_source_message",
    "collect_candidate_evidence",
    "collect_raw_candidate_evidence",
    "decide_candidate",
    "extract_message_candidate_counts",
    "extract_terms_locally",
    "finalize_candidate_evidence",
    "is_excluded_source_term",
    "is_meaningful_context",
    "is_placeholder_like",
    "is_valid_phrase_tokens",
    "is_valid_single_token",
    "maybe_canonicalize_candidate",
    "normalize_candidate_identity_key",
    "normalize_candidate_key",
    "normalize_space",
    "normalize_ui_source_text",
    "normalize_vocabulary_match_text",
    "parse_location_note",
    "should_include_source_text",
    "should_include_phrase",
    "singularize_english_token",
    "sort_candidates",
    "strip_candidate_possessive_suffix",
    "strip_english_possessive",
    "strip_ui_accelerators",
    "tokenize_candidate_source_text",
    "tokenize_source_text",
    "validate_max_length",
]
