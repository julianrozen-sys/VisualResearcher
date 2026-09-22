"""Stage 6: query generation (CLAUDE.md §7, §22 P2).

Turns a segment's analysis into the search queries that P3 and P5 will run.
The acceptance bar is at least four queries per segment **of differing kinds**,
because four rewordings of the same phrase return the same thirty images and
the ranking stage then has nothing to choose between.

Two rules from §10 land here, and nowhere else:

* corrections at or above 0.75 are applied to query text (§10.2 -- queries
  only, never the transcript);
* below 0.75, both spellings are searched and ranking decides (§10.3).

Queries are ordered most specific first, because the downstream budget is
finite and the exact-event query is the one most likely to find the shot the
narration is actually describing.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..logging_setup import get_logger
from ..schemas import EntityCorrection, ProjectContext, Query, QueryKind, Segment, VisualIntent
from .entities import apply_corrections, both_spellings

__all__ = ["generate_queries", "build_segment_queries", "MIN_QUERIES"]

log = get_logger("pipeline.queries")

#: §22 P2 -- at least this many queries per segment, of differing kinds.
MIN_QUERIES = 4

#: Which kinds to aim for per intent, most specific first. Diversity of *kind*
#: is what matters; the text is built from whatever the segment actually has.
_INTENT_PLAN: dict[VisualIntent, tuple[QueryKind, ...]] = {
    VisualIntent.EXACT_EVENT: (
        QueryKind.EXACT_EVENT,
        QueryKind.CHARACTER_SETTING,
        QueryKind.QUEST,
        QueryKind.YOUTUBE,
        QueryKind.LOCATION,
        QueryKind.FALLBACK,
    ),
    VisualIntent.CHARACTER: (
        QueryKind.CHARACTER_SETTING,
        QueryKind.EXACT_EVENT,
        QueryKind.YOUTUBE,
        QueryKind.OBJECT,
        QueryKind.FALLBACK,
    ),
    VisualIntent.LOCATION: (
        QueryKind.LOCATION,
        QueryKind.CHARACTER_SETTING,
        QueryKind.YOUTUBE,
        QueryKind.FALLBACK,
    ),
    VisualIntent.GAME_SCENE: (
        QueryKind.EXACT_EVENT,
        QueryKind.CHARACTER_SETTING,
        QueryKind.QUEST,
        QueryKind.YOUTUBE,
        QueryKind.LOCATION,
        QueryKind.FALLBACK,
    ),
    VisualIntent.FILM_SCENE: (
        QueryKind.EXACT_EVENT,
        QueryKind.CHARACTER_SETTING,
        QueryKind.YOUTUBE,
        QueryKind.FALLBACK,
    ),
    VisualIntent.COMIC: (
        QueryKind.CHARACTER_SETTING,
        QueryKind.EXACT_EVENT,
        QueryKind.OBJECT,
        QueryKind.FALLBACK,
    ),
    VisualIntent.ABSTRACT: (
        QueryKind.FALLBACK,
        QueryKind.LOCATION,
        QueryKind.OBJECT,
        QueryKind.YOUTUBE,
    ),
    VisualIntent.B_ROLL: (
        QueryKind.FALLBACK,
        QueryKind.LOCATION,
        QueryKind.CHARACTER_SETTING,
        QueryKind.YOUTUBE,
    ),
}


#: Filler that carries no visual meaning. A search for "of the and that" is a
#: search for nothing, and these words dominate spoken narration.
_STOPWORDS = frozenset(
    """a about above after again against all also am an and any are aren as at be because
    been before being below between both but by can cannot could couldn did didn do does
    doesn doing don down during each few for from further had hadn has hasn have haven
    having he her here hers herself him himself his how i if in into is isn it its itself
    just me more most mustn my myself no nor not now of off on once only or other ought
    our ours ourselves out over own same shan she should shouldn so some such than that
    the their theirs them themselves then there these they this those through to too under
    until up very was wasn we were weren what when where which while who whom why with
    won would wouldn you your yours yourself yourselves
    actually anyway basically course got kind lot maybe mean really say says said see
    sort thing things well going get go come came like little bit much many one two
    first second next last time way back even still never always something anything
    everything nothing someone anyone everyone able make made take taken put
    course right okay yeah yes look looks looking know knew think thought want wanted
    will shall must might may can could would should being been have has had does did
    used using use makes making gets getting give given gave keep kept let lets
    every each both either neither enough quite rather almost nearly
    beyond mentioned mention mentions compares compare compared comparing
    described describe describes description discussed discuss discussing
    talked talking speak speaks speaking spoken tell tells telling told
    example examples instance instances means meaning meant includes include
    including consider considered considering imagine suppose perhaps
    essentially effectively arguably obviously clearly simply merely
    whole entire entirely various several certain particular specific
    beginning start started starts ended ending finish finished
    happened happens happen occur occurs occurred
    called calls call named names naming known knows
    seen saw watched watching notice noticed
    remember remembered forget forgot understand understood
    reason reasons result results cause causes caused
    question questions answer answers point points part parts
    today yesterday tomorrow later earlier before afterwards
    video channel episode series subscribe comment comments like likes
    welcome thanks thank hello goodbye""".split()
)

#: What a model writes when it has nothing to say. Searching these literally
#: -- real runs issued "Star Wars N/A" -- spends a provider call on noise.
_NULL_MARKERS = frozenset(
    {"n/a", "na", "none", "unknown", "unspecified", "null", "nil", "tbd", "-", "--", "?"}
)


def strip_null_markers(text: str) -> str:
    """Drop a model's placeholder words from query text."""
    kept = [w for w in text.split() if w.lower().strip(".,;:") not in _NULL_MARKERS]
    return " ".join(kept).strip()


#: Below this a word is usually a preposition or an artefact of transcription.
_MIN_KEYWORD_LEN = 4

#: How many narration words to carry into a query. Long queries return fewer
#: and worse results on image search, so this stays tight.
_MAX_KEYWORDS = 6


def _vocabulary(context: ProjectContext) -> list[str]:
    """Every named thing the project knows about, longest phrase first.

    Longest-first matters: "Dark Council" must win over "Council", and
    "Sith Warrior" over "Sith", or the query names a category instead of the
    thing the narration actually mentioned.
    """
    terms = {
        str(name).strip()
        for group in (context.characters, context.places, context.terminology, context.events)
        for name in (group or [])
        if str(name).strip()
    }
    return sorted(terms, key=lambda s: (-len(s), s.lower()))


def _capitalised_phrases(text: str) -> list[str]:
    """Runs of capitalised words that are not merely starting a sentence."""
    phrases: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        run: list[str] = []
        for position, raw in enumerate(words):
            word = raw.strip(".,;:!?\"'()[]")
            ordinary = word.lower() in _STOPWORDS
            looks_named = len(word) >= _MIN_KEYWORD_LEN and word[:1].isupper()
            # A sentence's first word is capitalised whatever it is, so it
            # normally proves nothing -- except when the next word is a name
            # too. Without this "Jaesa Willsaam said..." searched for
            # "Willsaam", losing the half of the name people recognise.
            if position == 0 and looks_named and not ordinary and len(words) > 1:
                nxt = words[1].strip(".,;:!?\"'()[]")
                nxt_named = len(nxt) >= _MIN_KEYWORD_LEN and nxt[:1].isupper()
                if nxt_named and nxt.lower() not in _STOPWORDS:
                    run.append(word)
                    continue
            if looks_named and not ordinary and position > 0:
                run.append(word)
            else:
                if run:
                    phrases.append(" ".join(run))
                    run = []
        if run:
            phrases.append(" ".join(run))
    return phrases


def narration_subjects(
    segment: Segment, context: ProjectContext, corrections: list[EntityCorrection]
) -> list[str]:
    """The named things this segment talks about: who, where, what.

    This is what a picture can actually be *of*. A query built from the
    sentence's verbs -- "knowledge escape grasp" -- returns stock photography;
    one built from "Darth Baras Dark Council" returns the subject.

    Matching is case-insensitive and phrase-based against the project's own
    vocabulary, because the things that matter are often lowercase in speech:
    "lightsaber", "the Force" and "holocron" never match a capitalisation
    rule, and those are exactly the weapons and objects worth searching for.
    """
    text = apply_corrections(segment.narration or "", corrections)
    lowered = text.lower()
    found: list[str] = []
    covered: list[str] = []

    for term in _vocabulary(context):
        key = term.lower()
        # "the Force" is how the vocabulary spells it; narration says "Force
        # can be used...". Matching the bare noun too is the difference
        # between finding the weapon or object and finding nothing.
        bare = re.sub(r"^(the|a|an)\s+", "", key)
        hit = key in lowered or (bare != key and re.search(rf"\b{re.escape(bare)}\b", lowered))
        if hit and not any(key in seen or bare in seen for seen in covered):
            found.append(term)
            covered.append(key)

    # Names the project vocabulary has never heard of still deserve a search.
    for phrase in _capitalised_phrases(text):
        key = phrase.lower()
        if not any(key in seen or seen in key for seen in covered):
            found.append(phrase)
            covered.append(key)

    return found


def narration_actions(
    segment: Segment, context: ProjectContext, corrections: list[EntityCorrection]
) -> list[str]:
    """What is happening in the sentence, minus the names of who is doing it.

    A name alone finds portraits. "Sidious" returns headshots; "Sidious Force
    lightning throne room" returns the scene being described. The action and
    the concepts are what make an image the *right* one rather than merely
    one of the right person, so they are searched alongside the subject
    instead of being discarded the moment a name is found.
    """
    subject_words = {
        w.lower()
        for subject in narration_subjects(segment, context, corrections)
        for w in subject.split()
    }
    return [w for w in narration_keywords(segment, context, corrections)
            if w.lower() not in subject_words]


def narration_keywords(
    segment: Segment, context: ProjectContext, corrections: list[EntityCorrection]
) -> list[str]:
    """Content words from the narration: the concepts and actions in it.

    Filler is dropped, so what is left is what the sentence is *about*.
    Used both on its own, for a line that names nobody, and alongside the
    named subjects to describe what those subjects are doing.
    """
    text = apply_corrections(segment.narration or "", corrections)
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.split():
        word = raw.strip(".,;:!?\"'()[]-")
        key = word.lower()
        if len(word) < _MIN_KEYWORD_LEN or key in _STOPWORDS or key in seen:
            continue
        if not any(ch.isalpha() for ch in word):
            continue
        # "we've", "don't", "Sidious's" -- artefacts of speech, not subjects.
        if "'" in word or chr(8217) in word:
            continue
        seen.add(key)
        out.append(word)
    return out[:_MAX_KEYWORDS]


def _has_subject(text: str, tag: str) -> bool:
    """Does this query say anything beyond the franchise and stock phrasing?

    "Star Wars environment wide shot" passes every other check and is useless:
    it names no subject, so the provider returns generic art. Queries that
    reduce to the tag plus boilerplate are dropped rather than issued.
    """
    boilerplate = {"environment", "wide", "shot", "concept", "art", "close", "up",
                   "cutscene", "scene", "establishing"}
    tag_words = {w.lower() for w in tag.split()}
    rest = [w for w in text.lower().split() if w not in tag_words and w not in boilerplate]
    return bool(rest)


def _clean(*parts: str) -> str:
    """Join non-empty parts, collapse whitespace, drop duplicated words."""
    words: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for word in str(part).split():
            key = word.lower().strip(".,;:")
            if key and key not in seen:
                words.append(word)
                seen.add(key)
    return " ".join(words).strip()


def _subject_tag(context: ProjectContext) -> str:
    """A short qualifier that keeps results in the right universe.

    Searching "Baras Dark Council" without it returns a spread of unrelated
    fiction; with it the first page is the right game.

    An explicit ``search_tag`` from the domain pack wins, because the short
    form is what search engines respond to: §7's example queries read
    "Darth Baras Dark Council SWTOR", not "... Star Wars: The Old Republic".
    Falling back to the franchise rather than the subject keeps the tag short
    when no pack supplied one.
    """
    for candidate in (context.search_tag, context.franchise, context.subject):
        if candidate:
            words = candidate.split()
            return " ".join(words[:3])
    return ""


def build_segment_queries(
    segment: Segment,
    context: ProjectContext,
    corrections: list[EntityCorrection],
    *,
    min_queries: int = MIN_QUERIES,
) -> list[Query]:
    """Build the query list for one segment, most specific first."""
    tag = _subject_tag(context)
    entities = [apply_corrections(e, corrections) for e in segment.entities]
    primary = entities[0] if entities else ""
    secondary = entities[1] if len(entities) > 1 else ""
    location = apply_corrections(segment.location, corrections)
    event = apply_corrections(segment.event, corrections)
    topic = apply_corrections(segment.topic or segment.narration[:80], corrections)

    subjects = narration_subjects(segment, context, corrections)
    keywords = narration_keywords(segment, context, corrections)
    actions = narration_actions(segment, context, corrections)
    # Two questions, two queries. "Who or what is this about" finds the
    # subject; "what is happening" finds the shot. Searching only the first
    # returns portraits of the right character doing nothing in particular.
    if subjects:
        narration_text = _clean(" ".join(subjects[:2]), " ".join(actions[:3]), tag)
        subject_only = _clean(" ".join(subjects[:3]), tag)
    else:
        narration_text = _clean(" ".join(keywords[:5]), tag)
        subject_only = ""
    builders: dict[QueryKind, str] = {
        QueryKind.NARRATION: narration_text,
        QueryKind.EXACT_EVENT: _clean(primary, event or topic, location),
        QueryKind.CHARACTER_SETTING: _clean(primary, secondary, location, tag),
        QueryKind.QUEST: _clean(tag, primary, event or topic),
        QueryKind.YOUTUBE: _clean(primary, event or topic, tag, "cutscene scene"),
        QueryKind.LOCATION: _clean(location or primary, tag, "environment wide shot"),
        QueryKind.OBJECT: _clean(primary, tag, "concept art close up"),
        QueryKind.FALLBACK: _clean(tag, primary or location or topic),
    }

    plan = (QueryKind.NARRATION, *_INTENT_PLAN.get(
        segment.visual_intent, _INTENT_PLAN[VisualIntent.B_ROLL]
    ))
    queries: list[Query] = []
    used_text: set[str] = set()

    # Queries that name no subject are held back rather than dropped: they are
    # a poor use of a provider call, but §22 P2 still wants four per segment,
    # and a weak query beats padding the list with a duplicate.
    subjectless: list[Query] = []
    for kind in plan:
        text = strip_null_markers(builders.get(kind, "").strip())
        if not text:
            continue
        key = text.lower()
        if key in used_text:
            continue  # same words under a different label is not a different query
        used_text.add(key)
        if _has_subject(text, tag):
            queries.append(Query(kind=kind, text=text))
        else:
            subjectless.append(Query(kind=kind, text=text))

    # §10.3: for a correction we did not apply, search the other spelling too.
    for query in list(queries):
        for variant in both_spellings(query.text, corrections)[1:]:
            key = variant.lower()
            if key not in used_text:
                used_text.add(key)
                queries.append(Query(kind=query.kind, text=variant))

    # The bare subject, as its own search: a portrait is sometimes exactly
    # what a segment wants, and it is a genuinely different result set.
    if subject_only and subject_only.lower() not in used_text:
        used_text.add(subject_only.lower())
        queries.insert(1, Query(kind=QueryKind.NARRATION, text=subject_only))

    # Only now, if the good queries did not reach the minimum, bring back the
    # ones that named nothing -- least-bad first.
    while len(queries) < min_queries and subjectless:
        queries.append(subjectless.pop(0))

    # Top up from the narration itself rather than returning fewer than the
    # minimum. These are weak queries, which is why they come last.
    if len(queries) < min_queries:
        extras = [_clean(s, tag) for s in subjects[3:6]]
        extras += _narration_fallbacks(segment, tag, keywords)
        for extra in extras:
            if len(queries) >= min_queries:
                break
            extra = strip_null_markers(extra)
            key = extra.lower()
            if extra and key not in used_text:
                used_text.add(key)
                queries.append(Query(kind=QueryKind.FALLBACK, text=extra))

    # Last resort. A sentence with no name, no concept and no franchise tag
    # still has its own words, and §22 P2 wants four queries whatever the
    # segment contains.
    if len(queries) < min_queries and segment.narration:
        words = segment.narration.split()
        for chunk in (words[:8], words[8:16]):
            if len(queries) >= min_queries:
                break
            text = strip_null_markers(_clean(tag, " ".join(chunk)))
            key = text.lower()
            if text and key not in used_text:
                used_text.add(key)
                queries.append(Query(kind=QueryKind.FALLBACK, text=text))

    return queries


def _narration_fallbacks(segment: Segment, tag: str, keywords: list[str]) -> list[str]:
    """Extra query text from the narration, for topping up to MIN_QUERIES.

    Built from the same filtered keywords as the narration query rather than
    raw word slices: the old version took the first six words longer than
    three characters, which on spoken narration is mostly filler.
    """
    out = []
    if keywords:
        out.append(_clean(" ".join(keywords[:3]), tag))
        if len(keywords) > 3:
            out.append(_clean(" ".join(keywords[3:]), tag))
    return [o for o in out if o]


def generate_queries(
    segments: list[Segment],
    context: ProjectContext,
    corrections: list[EntityCorrection],
    settings: Settings | None = None,
    *,
    min_queries: int = MIN_QUERIES,
) -> list[Segment]:
    """Attach queries to every segment. Mutates and returns ``segments``."""
    thin = 0
    for segment in segments:
        segment.queries = build_segment_queries(
            segment, context, corrections, min_queries=min_queries
        )
        kinds = {str(q.kind) for q in segment.queries}
        if len(segment.queries) < min_queries or len(kinds) < 2:
            thin += 1
            segment.notes.append(
                f"only {len(segment.queries)} quer(y/ies) in {len(kinds)} kind(s); "
                "the narration gave little to work with"
            )

    totals = [len(s.queries) for s in segments] or [0]
    log.info(
        "query generation: %d segment(s), %.1f queries each on average (min %d, max %d)",
        len(segments),
        sum(totals) / len(totals),
        min(totals),
        max(totals),
    )
    if thin:
        log.warning("%d segment(s) produced fewer than %d queries", thin, min_queries)
    return segments
