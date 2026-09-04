"""Name extraction and lightweight conversational evidence.

The local model is the authority for deciding which words are people.  The
token fallback exists for operation without an LLM and for producing generic
turn-boundary evidence; it does not contain recording-specific names.
"""
from __future__ import annotations

from typing import Iterable, List

# These are language/transcription artifacts, not participant names.  The
# LLM name-listing pass is still the primary filter for real recordings.
# Only closed-class English words (function words, interrogatives,
# connectors, greetings) and generic collective nouns belong here — never
# recording-specific vocabulary.
NON_NAME_WORDS = {
    # pronouns / determiners / quantifiers
    "all", "another", "any", "anybody", "anyone", "anything", "both", "each",
    "either", "every", "everybody", "everyone", "everything", "few", "he",
    "her", "hers", "him", "his", "i", "it", "its", "many", "me", "mine",
    "most", "my", "neither", "no", "nobody", "none", "nothing", "one", "our",
    "ours", "she", "some", "somebody", "something", "such", "that", "the",
    "their", "them", "these", "they", "this", "those", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "we", "who", "whom",
    "whose", "whoever", "whatever", "whenever", "wherever", "which", "you",
    "your", "yours",
    # verbs / auxiliaries
    "am", "are", "be", "been", "being", "can", "could", "did", "do", "does",
    "had", "has", "have", "is", "let", "lets", "may", "might", "must", "shall",
    "should", "was", "were", "will", "would", "cannot", "dont", "didnt",
    "cant", "wont",
    # prepositions / conjunctions / connectors
    "about", "after", "against", "although", "and", "as", "at", "because",
    "before", "between", "but", "by", "consequently", "during", "either",
    "for", "from", "furthermore", "hence", "however", "if", "into", "indeed",
    "moreover", "nevertheless", "nor", "of", "off", "on", "onto", "or",
    "otherwise", "over", "per", "since", "so", "subsequently", "than", "then",
    "therefore", "through", "till", "to", "under", "until", "up", "via",
    "whilst", "whether", "with", "within", "without", "yet", "again",
    "additionally", "also", "anyway", "obviously", "perhaps", "maybe",
    "unfortunately", "surely", "instead", "meanwhile", "otherwise",
    # greetings / discourse / transcription artifacts
    "alright", "applause", "cheers", "evening", "father", "god", "good",
    "great", "guys", "hello", "here", "hey", "hi", "holy", "how", "laughter",
    "man", "morning", "afternoon", "welcome", "now", "oh", "okay", "piss",
    "please", "quick", "ramen", "right", "sorry", "sure", "thank",
    "thanks", "there", "uh", "um", "wait", "well", "when", "where", "why",
    "yeah", "yes",
    # generic collective / meeting nouns (never personal names on their own)
    "members", "council", "councillors", "councilors", "officers", "officer",
    "residents", "leader", "agenda", "committee", "people", "person",
    "speaker", "speakers", "item", "items", "report", "reports", "meeting",
    "meetings",
}

# Titles that immediately precede a personal name in ordinary English usage.
# "Councillor Perry", "Mr Coleman", "Dr Beadle".
HONORIFIC_TITLES = {
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "rev",
    "fr", "sir", "lady", "lord", "captain", "colonel", "sergeant", "judge",
    "alderman", "mayor", "councillor", "councilor", "cllr", "coun",
}



def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """Split words without assuming a particular sentence template."""
    spans: list[tuple[str, int, int]] = []
    start = None
    for index, char in enumerate(text):
        if char.isalpha() or (char == "'" and start is not None):
            if start is None:
                start = index
        elif start is not None:
            spans.append((text[start:index], start, index))
            start = None
    if start is not None:
        spans.append((text[start:], start, len(text)))
    return spans


def is_plausible_name(value: str) -> bool:
    """Reject labels/artifacts while allowing normal multi-part names."""
    if not isinstance(value, str):
        return False
    words = value.strip().split()
    if not 1 <= len(words) <= 4 or len(value.strip()) > 80:
        return False
    for word in words:
        normalized = word.strip("-'\u2019").casefold()
        if not normalized or normalized in NON_NAME_WORDS:
            return False
        if not any(char.isalpha() for char in normalized):
            return False
        if not (word[0].isupper() and word[1:].replace("-", "").replace("'", "").replace("\u2019", "").islower()):
            return False
    return True


def is_model_name_candidate(value: str) -> bool:
    """Validate a name classified by the LLM without requiring title case."""
    if not isinstance(value, str):
        return False
    words = value.strip().split()
    if not 1 <= len(words) <= 4 or len(value.strip()) > 80:
        return False
    for word in words:
        normalized = word.strip("-'\u2019").casefold()
        if not normalized or normalized in NON_NAME_WORDS:
            return False
        if not all(char.isalpha() or char in "-'\u2019" for char in word):
            return False
    return True


def extract_candidates(text: str) -> List[str]:
    """Return generic title-case tokens, deduplicated in transcript order."""
    candidates: list[str] = []
    seen: set[str] = set()
    for token, _, _ in _word_spans(text):
        if "'" in token and not token.casefold().endswith("'"):
            continue
        if is_plausible_name(token) and token not in seen:
            candidates.append(token)
            seen.add(token)
    return candidates


def should_trigger_llm(text: str, speaker_id: str | None = None) -> bool:
    """Retained for callers; candidate discovery itself is no longer trigger-based."""
    return bool(extract_candidates(text))


def _find_occurrences(text: str, name: str) -> list[tuple[int, int]]:
    """Find exact case-insensitive name occurrences without regex templates."""
    haystack = text.casefold()
    needle = name.casefold()
    if not needle:
        return []
    occurrences = []
    offset = 0
    while True:
        start = haystack.find(needle, offset)
        if start < 0:
            return occurrences
        end = start + len(needle)
        before_ok = start == 0 or not haystack[start - 1].isalnum()
        after_ok = end == len(haystack) or not haystack[end].isalnum()
        if before_ok and after_ok:
            occurrences.append((start, end))
        offset = end


def name_occurs_in_segments(name: str, aligned_segments: list[dict]) -> bool:
    """Confirm a model-listed name is literally present in the transcript."""
    return any(
        _find_occurrences(str(segment.get("text", "")), name)
        for segment in aligned_segments
    )


def _has_word_sequence(words: list[str], sequence: tuple[str, ...], name: str) -> bool:
    lowered = [word.casefold() for word in words]
    name_words = tuple(word.casefold() for word in name.split())
    sequence = tuple(word.casefold() for word in sequence)
    width = len(sequence) + len(name_words)
    for index in range(len(words) - width + 1):
        if tuple(lowered[index : index + len(sequence)]) == sequence and tuple(lowered[index + len(sequence) : index + width]) == name_words:
            # The introducing sequence must be followed by a name-shaped
            # token: "I'm Dan" is a self-identification, "I'm actually" and
            # "this is full council" are not.
            if not words[index + len(sequence)].strip("-'\u2019")[:1].isupper():
                return False
            return True
    return False


def _name_is_vocative(text: str, occurrence: tuple[int, int]) -> bool:
    start, end = occurrence
    prefix = text[:start].rstrip()
    suffix = text[end:].lstrip()
    if not (prefix.endswith(",") or suffix.startswith(",")):
        return False
    previous = _previous_word(text, start)
    if previous is None:
        return True
    normalized = previous.strip("-'\u2019").casefold()
    if normalized in ADDRESS_PREFIX_WORDS or normalized in HONORIFIC_TITLES:
        return True
    # A trailing vocative sits at the end of its clause: ", Dan." / ", Dan?" /
    # ", Dan" (end of segment).  A mid-clause comma subject ("At a stroke,
    # flooding in ...") is followed by more words and is not an addressee.
    if prefix.endswith(","):
        if suffix == "":
            return True
        return suffix[0] in ".!?"
    return False


def _previous_word(text: str, index: int) -> str | None:
    spans = _word_spans(text[:index])
    return spans[-1][0] if spans else None


def _honorific_prefixed(text: str, occurrence: tuple[int, int]) -> bool:
    start, _ = occurrence
    previous = _previous_word(text, start)
    if previous is None:
        return False
    return previous.strip("-'\u2019").casefold() in HONORIFIC_TITLES


# Words that immediately precede a trailing vocative ("Hey Dan,",
# "Thank you, Chairman.").  Deliberately narrow: ordinary prepositions and
# nouns must not qualify, or object nouns ("risk of flooding,") read as
# addressees.
ADDRESS_PREFIX_WORDS = {
    "hey", "hi", "hello", "thank", "thanks", "you", "sorry", "well", "okay",
    "right", "now", "so", "but", "and", "yes", "yeah", "no", "good", "evening",
    "morning", "afternoon", "welcome", "please", "um", "uh", "oh", "listen",
    "look", "come", "anyway", "cheers", "congratulations", "excuse",
}

# Word sequences that introduce the following token(s) as a person:
# "my name is Ada", "I'm Ada", "this is Ada", "introduce Ada",
# "we'll hear from Ada", "speaking is Ada".
NAME_INTRODUCING_SEQUENCES = (
    ("my", "name", "is"),
    ("i", "am"),
    ("i'm",),
    ("this", "is"),
    ("introduce",),
    ("introducing",),
    ("hear", "from"),
    ("speaking", "is"),
    ("call", "me"),
    ("on", "behalf", "of"),
)


def name_usage_supported(name: str, aligned_segments: list[dict]) -> bool:
    """Confirm a candidate name is used like a name somewhere in the transcript.

    A capitalized word appearing mid-sentence is not evidence of a person
    (sentence-initial "However", place names such as "Langford Road").  The
    name must appear in a person-typical construction: vocative address with
    a proper boundary, an honorific title (before or inside the name), or an
    introducing sequence.  A sentence-initial vocative alone ("However, the
    density ...") is weak evidence and additionally requires a capitalized
    mid-sentence occurrence of the same name.
    """
    if not isinstance(name, str) or not name.strip():
        return False
    name_words = name.split()
    if all(word.strip("-'\u2019").casefold() in NON_NAME_WORDS for word in name_words):
        return False
    if len(name_words) == 1 and name_words[0].strip("-'\u2019").casefold() in HONORIFIC_TITLES:
        # "Mr", "Councillor" on their own are titles, not participants.
        return False
    title_led = len(name_words) > 1 and name_words[0].strip("-'\u2019").casefold() in HONORIFIC_TITLES
    strong = False
    weak = False
    mid_capitalized = False
    for segment in aligned_segments:
        text = str(segment.get("text", ""))
        occurrences = _find_occurrences(text, name)
        if not occurrences:
            continue
        if title_led:
            # "Councillor Perry", "Mr Lee": a title-led string found verbatim
            # in the transcript is itself a person reference.
            return True
        words = [token for token, _, _ in _word_spans(text)]
        if any(_has_word_sequence(words, sequence, name) for sequence in NAME_INTRODUCING_SEQUENCES):
            return True
        for occurrence in occurrences:
            if _honorific_prefixed(text, occurrence):
                strong = True
                continue
            if _name_is_vocative(text, occurrence):
                if text[: occurrence[0]].strip():
                    strong = True
                else:
                    weak = True
            if occurrence[0] > 0 and text[: occurrence[0]].strip() and text[occurrence[0]].isupper():
                mid_capitalized = True
    return strong or (weak and mid_capitalized)


def heuristic_evidence(aligned_segments: list[dict], candidate_names: Iterable[str] | None = None) -> List[dict]:
    """Produce evidence using candidate names and turn structure only.

    No recording-specific names or English greeting regexes are used here.
    Candidate names come from the LLM listing pass in the production pipeline.
    """
    if candidate_names is None:
        candidate_names = (
            candidate
            for segment in aligned_segments
            for candidate in extract_candidates(segment.get("text", ""))
        )
    names = list(dict.fromkeys(name for name in candidate_names if is_plausible_name(name)))
    evidence: list[dict] = []

    for index, segment in enumerate(aligned_segments):
        text = str(segment.get("text", ""))
        speaker = segment.get("speaker_id", segment.get("speaker", ""))
        words = [token for token, _, _ in _word_spans(text)]
        for name in names:
            occurrences = _find_occurrences(text, name)
            if not occurrences:
                continue

            self_identified = any(
                _has_word_sequence(words, sequence, name)
                for sequence in (("i", "am"), ("i'm",), ("my", "name", "is"), ("this", "is"))
            )
            introduced = any(
                _has_word_sequence(words, sequence, name)
                for sequence in (("meet",), ("introduce",))
            )
            if self_identified:
                evidence.append(
                    {
                        "type": "self_identification",
                        "speaker": speaker,
                        "name": name,
                        "confidence": 0.98,
                        "evidence": f"{speaker} self-identifies as {name}: \"{text[:80]}\"",
                        "segment_id": segment.get("segment_id"),
                        "segment_index": index,
                    }
                )
            elif introduced:
                evidence.append(
                    {
                        "type": "introduction",
                        "speaker": speaker,
                        "name": name,
                        "confidence": 0.85,
                        "evidence": f"{speaker} introduces {name}: \"{text[:80]}\"",
                        "segment_id": segment.get("segment_id"),
                        "segment_index": index,
                    }
                )

            if not self_identified and any(_name_is_vocative(text, occurrence) for occurrence in occurrences):
                evidence.append(
                    {
                        "type": "direct_address",
                        "speaker": speaker,
                        "target_name": name,
                        "confidence": 0.70,
                        "evidence": f"{speaker} appears to address {name}: \"{text[:80]}\"",
                        "segment_id": segment.get("segment_id"),
                        "segment_index": index,
                    }
                )

    # A following turn from a different voice is evidence for the addressed name.
    # The responder may be a few segments later (whisper often splits one turn,
    # and the addressed speaker may pause before replying), so scan forward past
    # consecutive segments from the same speaker.
    for item in list(evidence):
        if item["type"] != "direct_address":
            continue
        index = item["segment_index"]
        for look in range(index + 1, min(index + 5, len(aligned_segments))):
            following = aligned_segments[look]
            following_speaker = following.get("speaker_id", following.get("speaker", ""))
            if following_speaker and following_speaker != item["speaker"]:
                evidence.append(
                    {
                        "type": "response_to_address",
                        "speaker": following_speaker,
                        "candidate_name": item["target_name"],
                        "confidence": float(item.get("confidence", 0.70)),
                        "evidence": f"{following_speaker} responded after {item['speaker']} addressed {item['target_name']}",
                        "segment_index": look,
                        "segment_id": following.get("segment_id"),
                        "source_evidence": item,
                    }
                )
                break
    return evidence


def extract_entities(aligned_segments: list[dict]) -> dict:
    """Return generic fallback entities and evidence for non-LLM callers."""
    all_names: list[str] = []
    for segment in aligned_segments:
        all_names.extend(extract_candidates(segment.get("text", "")))
    names = list(dict.fromkeys(all_names))
    entities = [{"name": name, "type": "person"} for name in names]
    evidence = heuristic_evidence(aligned_segments, names)
    return {
        "entities": entities,
        "evidence": evidence,
        "needs_llm": bool(names),
    }
