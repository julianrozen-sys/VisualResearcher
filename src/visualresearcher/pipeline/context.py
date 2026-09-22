"""Stage 3: project context, and the per-segment analysis (CLAUDE.md §7, §8).

Two jobs:

* :func:`build_context` produces ``project_context.json`` -- what this video is
  about, which franchise and era, who appears in it, what the terminology is.
  Domain packs supply what they can for free; the LLM is asked only for what
  they could not.
* :func:`analyze_segments` fills in each segment's ``topic``, ``entities``,
  ``event``, ``location``, ``interpretation`` and ``visual_intent``.

Segment analysis batches **20 segments per LLM call** (§20). A 300-segment
video is 15 calls, not 300. That is the difference between a few cents and a
bill worth arguing about.
"""

from __future__ import annotations

import json
import re

from ..logging_setup import get_logger
from ..providers.base import ProviderError
from ..providers.llm.base import LLMProvider
from ..schemas import (
    EntityCorrection,
    ProjectContext,
    Segment,
    SubShot,
    Transcript,
    VisualIntent,
    Work,
)
from .entities import DomainPack, apply_corrections

__all__ = [
    "build_context",
    "analyze_segments",
    "BATCH_SIZE",
    "merge_pack_context",
]

log = get_logger("pipeline.context")

#: §20 -- "Batch LLM calls ~20 segments per call".
BATCH_SIZE = 20

_INTENTS = {i.value for i in VisualIntent}


def merge_pack_context(
    context: ProjectContext, packs: list[DomainPack], text: str
) -> ProjectContext:
    """Fold everything the active domain packs already know into the context.

    Free and reliable, so it runs before the model is consulted and its values
    are not overwritten by one.
    """
    lowered = text.lower()
    for pack in packs:
        context.domain_packs.append(pack.name)
        if pack.franchise and not context.franchise:
            context.franchise = pack.franchise
        if pack.era and not context.era:
            context.era = pack.era
        if pack.title and not context.subject:
            context.subject = pack.title
        if pack.search_tag and not context.search_tag:
            context.search_tag = pack.search_tag
        for term in pack.terminology:
            if term.lower() in lowered and term not in context.terminology:
                context.terminology.append(term)
        for work in pack.works:
            title = str(work.get("title", "")) if isinstance(work, dict) else str(work)
            if title and not any(w.title == title for w in context.works):
                kind = work.get("type", "unknown") if isinstance(work, dict) else "unknown"
                context.works.append(Work(title=title, type=str(kind)))
        for entity in pack.entities:
            # Only list an entity that actually turns up in this narration.
            spellings = [s.lower() for s in entity.all_spellings()]
            if not any(s in lowered for s in spellings):
                continue
            bucket = {
                "character": context.characters,
                "person": context.people,
                "place": context.places,
                "event": context.events,
                "terminology": context.terminology,
            }.get(entity.type, context.characters)
            if entity.canonical not in bucket:
                bucket.append(entity.canonical)
    return context


def build_context(
    transcript: Transcript,
    packs: list[DomainPack],
    *,
    llm: LLMProvider | None = None,
    corrections: list[EntityCorrection] | None = None,
) -> ProjectContext:
    """Assemble ``project_context.json``.

    The LLM fills only the gaps the packs left. A provider failure leaves the
    pack-derived context in place rather than failing the job (§8).
    """
    context = ProjectContext(provider="domain_packs")
    merge_pack_context(context, packs, transcript.text)

    if llm is not None:
        try:
            context = _enrich_with_llm(context, transcript, llm)
        except ProviderError as exc:
            log.warning("project context: LLM unavailable, using domain packs only (%s)", exc)

    if corrections:
        context.entity_corrections = list(corrections)
        # A resolved name belongs in the cast list under its correct spelling.
        for correction in corrections:
            if not correction.applied:
                continue
            if correction.resolved not in context.characters:
                context.characters.append(correction.resolved)

    log.info(
        "project context: subject=%r franchise=%r era=%r "
        "(%d characters, %d places, %d terms, %d corrections)",
        context.subject,
        context.franchise,
        context.era,
        len(context.characters),
        len(context.places),
        len(context.terminology),
        len(context.entity_corrections),
    )
    return context


def _enrich_with_llm(
    context: ProjectContext, transcript: Transcript, llm: LLMProvider
) -> ProjectContext:
    system = (
        "You identify what a video essay is about from its narration. Answer "
        "only from the transcript; do not speculate beyond it. Leave a field "
        "empty rather than guessing."
    )
    user = f"TRANSCRIPT:\n{transcript.text}"
    schema = {
        "subject": "string",
        "franchise": "string",
        "era": "string",
        "characters": ["string"],
        "people": ["string"],
        "places": ["string"],
        "events": ["string"],
        "terminology": ["string"],
        "works": [{"title": "string", "type": "string"}],
    }
    response = llm.complete_json(
        system=system, user=user, task="project_context", schema_hint=schema
    )

    # Scalars: only fill a blank. The packs are more reliable than the model.
    for attr in ("subject", "franchise", "era"):
        value = str(response.get(attr, "") or "").strip()
        if value and not getattr(context, attr):
            setattr(context, attr, value)

    for attr in ("characters", "people", "places", "events", "terminology"):
        existing = getattr(context, attr)
        known = {v.lower() for v in existing}
        for value in response.get(attr, []) or []:
            text = str(value).strip()
            if text and text.lower() not in known:
                existing.append(text)
                known.add(text.lower())

    for work in response.get("works", []) or []:
        if not isinstance(work, dict):
            continue
        title = str(work.get("title", "")).strip()
        if title and not any(w.title == title for w in context.works):
            context.works.append(Work(title=title, type=str(work.get("type", "unknown"))))

    context.provider = f"domain_packs+{llm.name}"
    return context


# ---------------------------------------------------------------------------
# Per-segment analysis
# ---------------------------------------------------------------------------


#: Words that begin a sentence without being a name. Without this every
#: sentence-initial "And", "They" or "But" becomes a search term.
_NOT_NAMES = frozenset(
    """a an and as at but by for from he her him his however in into it its meanwhile
    of on or she that the their them they this to was were what when where which who
    with you your not no so if then than there here now also after before while during
    all any both each few more most other some such only own same too very can will
    just don should could would may might must one two three first second next last
    because although though since until unless whether both either neither""".split()
)

#: A load-bearing noun has to be worth searching for. Two letters is an
#: initial, not a subject.
_MIN_NAME_LEN = 3


def _capitalised_midsentence(texts: list[str]) -> set[str]:
    """Words seen capitalised somewhere other than the start of a sentence.

    This is the evidence that a capitalised word is a *name* rather than a
    word that happens to start a sentence. "Republic" appears mid-sentence all
    through this transcript; "Hello", "Why" and "Starting" never do. Without
    this test the extractor emitted exactly those three as subjects, and a
    query for "Star Wars Hello" is worse than the blob it replaced.
    """
    seen: set[str] = set()
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for position, raw in enumerate(sentence.split()):
                if position == 0:
                    continue
                word = raw.strip(".,;:!?\"'()[]")
                if len(word) >= _MIN_NAME_LEN and word[:1].isupper():
                    seen.add(word.lower())
    return seen


def _proper_noun_phrases(text: str, evidence: set[str]) -> list[str]:
    """Capitalised phrases that look like names, in order of appearance.

    The narration is the only place some subjects appear: this transcript says
    "the Republic" and "the Core Worlds" constantly, and neither is in the
    project's character or terminology lists, so nothing matched them and the
    segment went out with no entity at all.

    A word that opens a sentence is capitalised by position, not by nature, so
    it counts only when `evidence` has also seen it capitalised mid-sentence.
    """
    phrases: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        current: list[str] = []
        for position, raw in enumerate(words):
            word = raw.strip(".,;:!?\"'()[]")
            lowered = word.lower()
            looks_capital = len(word) >= _MIN_NAME_LEN and word[:1].isupper()
            is_name = (
                looks_capital
                and lowered not in _NOT_NAMES
                and (position > 0 or lowered in evidence)
            )
            if is_name:
                current.append(word)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def _is_searchable_subject(name: str) -> bool:
    """Is this worth putting in an image query?

    Backfilling found `BBY` -- the in-universe date notation -- and carried it
    into a query as "Star Wars BBY", which is worse than the blob it replaced:
    it is a unit of time, and no image depicts one. A subject has to be
    something a picture can be *of*.
    """
    word = name.strip()
    if len(word) < _MIN_NAME_LEN:
        return False
    # Bare acronyms (BBY, ABY, CIS) name eras and organisations by initials;
    # a search engine has nothing visual to match them against.
    if word.isupper() and len(word.split()) == 1:
        return False
    if any(ch.isdigit() for ch in word):
        return False
    return True


#: Third-person references that point at something the segment never names.
_CONTINUATION = frozenset(
    "he she they him her them his hers their theirs its it this that these those".split()
)


def _looks_like_continuation(narration: str) -> bool:
    """Does this segment lean on something the previous one established?"""
    words = [w.strip(".,;:!?\"'").lower() for w in narration.split()[:12]]
    return any(w in _CONTINUATION for w in words)


def backfill_entities(
    segments: list[Segment],
    context: ProjectContext,
    corrections: list[EntityCorrection] | None = None,
    *,
    lookback: int = 2,
) -> int:
    """Give every segment an entity to anchor its queries and CLIP prompt on.

    `segment.entities` feeds three things at once -- the queries, the CLIP
    prompt and §11's entity bonus -- so a segment with an empty list is
    penalised three times over. On the first long real narration 35 of 170
    segments had none, and their best candidate scored a median 0.435 against
    0.576 for the rest: not one of them could clear the confidence gate.

    Two reasons a segment ends up empty, and they need different answers:

    * **Its subject is not in the project vocabulary.** "the perks of being
      part of the Republic" names a subject the character and terminology
      lists never mention, so matching against them finds nothing. Reading the
      capitalised phrases out of the narration itself does.
    * **Its subject is a pronoun.** "They feared that a standing military..."
      names nobody, because the previous segment already did. Only the
      preceding segments can say who "they" are.

    Returns the number of segments that gained entities. Each one records why
    in its notes, so a surprising query can be traced back to the guess that
    produced it.
    """
    corrections = corrections or []
    # Built from the whole narration, not one segment: a name that opens the
    # only sentence it appears in is indistinguishable from a sentence-opener.
    evidence = _capitalised_midsentence([s.narration for s in segments])
    evidence |= {w.lower() for name in
                 (*context.characters, *context.places, *context.terminology, *context.events)
                 for w in str(name).split()}
    recent: list[str] = []
    filled = 0

    for segment in sorted(segments, key=lambda s: s.index):
        if segment.entities:
            recent = [e for e in segment.entities if _is_searchable_subject(e)][:lookback]
            continue

        narration = apply_corrections(segment.narration, corrections)
        found = [
            phrase
            for phrase in _proper_noun_phrases(narration, evidence)
            if _is_searchable_subject(phrase)
        ]
        if found:
            segment.entities = found[:3]
            segment.notes.append(
                "entities read from the narration's own proper nouns "
                "(nothing in the project vocabulary matched)"
            )
            filled += 1
        elif recent and _looks_like_continuation(narration):
            segment.entities = [e for e in recent if _is_searchable_subject(e)]
        if not segment.entities and recent and _looks_like_continuation(narration):
            pass  # nothing carryable; a blob query still beats a wrong subject
        elif segment.entities and not found:
            segment.notes.append(
                "entities carried forward from the preceding segment(s): "
                f"{', '.join(segment.entities)} -- this segment names no subject of its own"
            )
            filled += 1

        if segment.entities:
            recent = [e for e in segment.entities if _is_searchable_subject(e)][:lookback]

    if filled:
        log.info("backfilled entities for %d segment(s) that had none", filled)
    return filled


def analyze_segments(
    segments: list[Segment],
    context: ProjectContext,
    *,
    llm: LLMProvider | None = None,
    corrections: list[EntityCorrection] | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[Segment]:
    """Fill in topic, entities, event, location, interpretation and intent.

    Mutates and returns ``segments``. Segments in a batch whose call fails are
    marked ``degraded`` and keep their heuristic fallback -- one bad batch
    never fails the job (§8).
    """
    corrections = corrections or []
    for segment in segments:
        _heuristic_analysis(segment, context, corrections)

    if llm is None:
        log.info("segment analysis: no LLM provider; heuristics only")
        return segments

    batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
    log.info(
        "segment analysis: %d segment(s) in %d LLM call(s) of up to %d",
        len(segments),
        len(batches),
        batch_size,
    )
    for number, batch in enumerate(batches, start=1):
        try:
            _analyze_batch(batch, context, llm)
        except ProviderError as exc:
            log.warning(
                "segment analysis: batch %d/%d failed, segments %d-%d degraded (%s)",
                number,
                len(batches),
                batch[0].index,
                batch[-1].index,
                exc,
            )
            for segment in batch:
                segment.status = "degraded"
                segment.notes.append(f"segment analysis unavailable: {exc}")

    # Last: whatever the model left without a subject still needs one, because
    # queries, the CLIP prompt and the entity bonus all read this field.
    backfill_entities(segments, context, corrections)
    return segments


def _analyze_batch(batch: list[Segment], context: ProjectContext, llm: LLMProvider) -> None:
    system = (
        "You plan the visuals for a video essay. For each narration segment, "
        "say what it is about and what should be on screen. Be concrete and "
        "specific: name the character, the place, the moment. If the narration "
        "describes a particular event, say so; if it is general commentary, "
        "say that instead. The interpretation should read as an instruction to "
        "a researcher looking for images, and should name a fallback."
    )
    known = {
        "subject": context.subject,
        "franchise": context.franchise,
        "era": context.era,
        "characters": context.characters[:30],
        "places": context.places[:30],
        "terminology": context.terminology[:30],
    }
    payload = [{"index": s.index, "narration": s.narration} for s in batch]
    user = (
        f"PROJECT: {json.dumps(known, ensure_ascii=False)}\n\n"
        f"visual_intent must be one of: {', '.join(sorted(_INTENTS))}\n\n"
        f"SEGMENTS:{json.dumps(payload, ensure_ascii=False)}"
    )
    schema = {
        "segments": [
            {
                "index": 0,
                "topic": "string",
                "entities": ["string"],
                "event": "string",
                "location": "string",
                "interpretation": "string",
                "visual_intent": "EXACT_EVENT",
                "confidence": 0.0,
                "sub_shots": [{"start": 0.0, "end": 0.0, "topic": "string"}],
            }
        ]
    }
    response = llm.complete_json(
        system=system, user=user, task="segment_analysis", schema_hint=schema, max_tokens=8000
    )

    by_index = {s.index: s for s in batch}
    for item in response.get("segments", []) or []:
        if not isinstance(item, dict):
            continue
        segment = by_index.get(item.get("index"))
        if segment is None:
            continue
        _apply_analysis(segment, item)


def _apply_analysis(segment: Segment, item: dict) -> None:
    """Copy one analysis object onto a segment, validating as it goes."""
    for attr in ("topic", "event", "location", "interpretation"):
        value = str(item.get(attr, "") or "").strip()
        if value:
            setattr(segment, attr, value)

    entities = [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
    if entities:
        segment.entities = entities

    intent = str(item.get("visual_intent", "") or "").strip().upper()
    if intent in _INTENTS:
        segment.visual_intent = VisualIntent(intent)
    elif intent:
        segment.notes.append(f"unknown visual_intent {intent!r} from the model; kept the default")

    try:
        confidence = float(item.get("confidence", segment.confidence))
        segment.confidence = round(min(max(confidence, 0.0), 1.0), 3)
    except (TypeError, ValueError):
        pass

    sub_shots: list[SubShot] = []
    for shot in item.get("sub_shots") or []:
        if not isinstance(shot, dict):
            continue
        try:
            start = float(shot["start"])
            end = float(shot["end"])
        except (KeyError, TypeError, ValueError):
            continue
        # A sub-shot outside its own segment is nonsense; clamp it.
        start = min(max(start, segment.start), segment.end)
        end = min(max(end, start), segment.end)
        if end > start:
            sub_shots.append(SubShot(start=start, end=end, topic=str(shot.get("topic", ""))))
    if sub_shots:
        segment.sub_shots = sub_shots


def _heuristic_analysis(
    segment: Segment, context: ProjectContext, corrections: list[EntityCorrection]
) -> None:
    """A defensible analysis with no model at all.

    Matches known names from the project context against the narration and
    picks an intent from what it finds. This is what ``VR_OFFLINE=1`` runs on,
    and it is also the floor a failed LLM batch falls back to.
    """
    corrected = apply_corrections(segment.narration, corrections)
    lowered = corrected.lower()

    found_characters = [c for c in context.characters if c.lower() in lowered]
    found_places = [p for p in context.places if p.lower() in lowered]
    found_terms = [t for t in context.terminology if t.lower() in lowered]

    segment.entities = found_characters + found_terms
    if found_places:
        segment.location = found_places[0]

    if not segment.topic:
        first_clause = corrected.split(".")[0].strip()
        segment.topic = " ".join(first_clause.split()[:12])

    if not segment.interpretation:
        if found_characters and found_places:
            segment.interpretation = (
                f"{found_characters[0]} at {found_places[0]}. "
                f"Fall back to a portrait of {found_characters[0]} or a wide shot of "
                f"{found_places[0]}."
            )
        elif found_characters:
            segment.interpretation = (
                f"{found_characters[0]} in this moment. "
                "Fall back to a portrait or a known scene featuring them."
            )
        elif found_places:
            segment.interpretation = (
                f"Establishing shot of {found_places[0]}. Fall back to a wide environment shot."
            )
        else:
            segment.interpretation = (
                "No named subject in this line; use broad B-roll for the surrounding topic."
            )

    if segment.visual_intent == VisualIntent.B_ROLL:
        if found_characters and found_places:
            segment.visual_intent = VisualIntent.GAME_SCENE
        elif found_characters:
            segment.visual_intent = VisualIntent.CHARACTER
        elif found_places:
            segment.visual_intent = VisualIntent.LOCATION

    if not segment.confidence:
        # Heuristics are worth stating a low number for: §14 will route these
        # to review rather than auto-accepting them.
        hits = len(found_characters) + len(found_places) + len(found_terms)
        segment.confidence = round(min(0.55, 0.2 + 0.1 * hits), 3)
