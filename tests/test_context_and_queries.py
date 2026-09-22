"""Project context and query generation (CLAUDE.md §7, §20, §22 P2).

The P2 acceptance bar is "at least 4 queries per segment of differing kinds",
which is tested here literally, along with §20's batching requirement and the
§10.2/§10.3 rules about where corrections may and may not be applied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visualresearcher.pipeline.context import (
    BATCH_SIZE,
    analyze_segments,
    build_context,
    merge_pack_context,
)
from visualresearcher.pipeline.entities import (
    DomainPack,
    PackEntity,
    load_domain_packs,
    resolve_entities,
    select_packs,
)
from visualresearcher.pipeline.queries import (
    MIN_QUERIES,
    build_segment_queries,
    generate_queries,
)
from visualresearcher.providers.base import Availability, ProviderError
from visualresearcher.providers.llm.base import LLMProvider, LLMResponse
from visualresearcher.providers.llm.fake import FakeLLMProvider
from visualresearcher.schemas import (
    ProjectContext,
    Segment,
    Transcript,
    TranscriptSegment,
    VisualIntent,
)

PACK_DIR = Path(__file__).resolve().parents[1] / "domain_packs"

SWTOR_TEXT = (
    "The Sith Warrior walks into the Dark Council chamber on Korriban. "
    "Barriss stands at the centre, the Voice of the Emperor, masked as always. "
    "Valron watches from the far side. Sanx betrayed the safe house. "
    "Drog fell on Hoth. The lightsaber ignites and the Sith Academy falls silent."
)


@pytest.fixture
def packs():
    return select_packs(SWTOR_TEXT, load_domain_packs(PACK_DIR))


@pytest.fixture
def transcript():
    return Transcript(
        language="en",
        duration=60.0,
        text=SWTOR_TEXT,
        segments=[TranscriptSegment(id=0, start=0.0, end=60.0, text=SWTOR_TEXT)],
    )


@pytest.fixture
def corrections(packs):
    return resolve_entities(SWTOR_TEXT, packs)


@pytest.fixture
def context(transcript, packs, corrections):
    return build_context(transcript, packs, corrections=corrections)


def _segment(index=31, **kwargs) -> Segment:
    base = {
        "index": index,
        "start": 229.0,
        "end": 237.0,
        "narration": "Barriss is unmasked before the Dark Council on Korriban.",
    }
    base.update(kwargs)
    return Segment(**base)


class _StubLLM(LLMProvider):
    name = "stub"

    def __init__(self, payload=None, *, fail=False):
        self.payload = payload if payload is not None else {}
        self.fail = fail
        self.calls: list[dict] = []

    def availability(self):
        return Availability.available("stub")

    def complete_json(self, *, system, user, task, schema_hint=None, max_tokens=4096):
        self.calls.append({"task": task, "user": user})
        if self.fail:
            raise ProviderError("stub failure")
        payload = self.payload(user) if callable(self.payload) else self.payload
        return LLMResponse(payload, model="stub")


# ---------------------------------------------------------------------------
# Pack selection ordering
# ---------------------------------------------------------------------------


def test_the_more_specific_pack_leads(packs):
    """swtor matches more triggers than star_wars, so it must set the subject.

    Both packs fire on this narration. If the generic one leads, the subject
    tag appended to every query degrades from "Star Wars: The Old Republic"
    to "Star Wars".
    """
    assert [p.name for p in packs][0] == "swtor", (
        f"expected swtor to lead, got {[p.name for p in packs]}"
    )


def test_the_specific_pack_sets_the_subject(context):
    assert context.subject == "Star Wars: The Old Republic"
    assert context.franchise == "Star Wars"
    assert context.era == "Old Republic"


def test_pack_ordering_is_deterministic():
    text = SWTOR_TEXT
    all_packs = load_domain_packs(PACK_DIR)
    first = [p.name for p in select_packs(text, all_packs)]
    second = [p.name for p in select_packs(text, list(reversed(all_packs)))]
    assert first == second, "pack order must not depend on directory listing order"


# ---------------------------------------------------------------------------
# Project context
# ---------------------------------------------------------------------------


def test_context_lists_only_entities_that_appear(context):
    assert "Darth Baras" in context.characters
    # Darth Nox is in the pack but not in this narration.
    assert "Darth Nox" not in context.characters


def test_context_picks_up_places_and_terminology(context):
    assert "Korriban" in context.places
    assert "Dark Council" in context.terminology


def test_context_records_the_active_packs(context):
    assert set(context.domain_packs) == {"swtor", "star_wars"}


def test_context_carries_the_corrections(context, corrections):
    assert len(context.entity_corrections) == len(corrections)
    originals = {c.original for c in context.entity_corrections}
    assert {"Barriss", "Valron", "Sanx", "Drog"} <= originals


def test_a_resolved_name_joins_the_cast_list(context):
    for resolved in ("Darth Baras", "Darth Vowrawn", "Colonel Senks", "Lord Draahg"):
        assert resolved in context.characters


def test_the_llm_only_fills_gaps_the_packs_left(transcript, packs):
    llm = _StubLLM(
        {
            "subject": "Something Else Entirely",
            "era": "Wrong Era",
            "events": ["The duel beneath the academy"],
        }
    )
    context = build_context(transcript, packs, llm=llm)
    assert context.subject == "Star Wars: The Old Republic", (
        "the pack is more reliable than the model; a filled field must not be overwritten"
    )
    assert context.era == "Old Republic"
    assert "The duel beneath the academy" in context.events, "empty fields should be filled"


def test_an_llm_failure_leaves_the_pack_context_intact(transcript, packs):
    context = build_context(transcript, packs, llm=_StubLLM(fail=True))
    assert context.subject == "Star Wars: The Old Republic"
    assert "Darth Baras" in context.characters


def test_merge_pack_context_is_idempotent(transcript, packs):
    context = ProjectContext()
    merge_pack_context(context, packs, transcript.text)
    first = context.model_dump()
    merge_pack_context(context, packs, transcript.text)
    second = context.model_dump()
    second["domain_packs"] = first["domain_packs"]  # this list is append-only by design
    assert first == second, "merging twice must not duplicate characters or terminology"


# ---------------------------------------------------------------------------
# Segment analysis, including §20 batching
# ---------------------------------------------------------------------------


def test_segment_analysis_batches_twenty_per_call(context):
    """§20: a 300-segment video must not cost a fortune."""
    segments = [_segment(index=i) for i in range(1, 46)]
    llm = _StubLLM({"segments": []})
    analyze_segments(segments, context, llm=llm)
    assert len(llm.calls) == 3, f"45 segments should be 3 calls of 20, got {len(llm.calls)}"
    assert BATCH_SIZE == 20


def test_a_three_hundred_segment_video_costs_fifteen_calls(context):
    segments = [_segment(index=i) for i in range(1, 301)]
    llm = _StubLLM({"segments": []})
    analyze_segments(segments, context, llm=llm)
    assert len(llm.calls) == 15


def test_analysis_is_applied_to_the_right_segments(context):
    segments = [_segment(index=1), _segment(index=2)]
    llm = _StubLLM(
        {
            "segments": [
                {
                    "index": 2,
                    "topic": "Second only",
                    "visual_intent": "EXACT_EVENT",
                    "confidence": 0.9,
                }
            ]
        }
    )
    analyze_segments(segments, context, llm=llm)
    assert segments[1].topic == "Second only"
    assert segments[1].visual_intent == VisualIntent.EXACT_EVENT
    assert segments[0].topic != "Second only"


def test_a_failed_batch_degrades_only_its_own_segments(context):
    segments = [_segment(index=i) for i in range(1, 41)]

    calls = {"n": 0}

    class _FlakyLLM(_StubLLM):
        def complete_json(self, *, system, user, task, schema_hint=None, max_tokens=4096):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderError("first batch failed")
            return LLMResponse({"segments": []}, model="stub")

    analyze_segments(segments, context, llm=_FlakyLLM())
    assert all(s.status == "degraded" for s in segments[:20])
    assert all(s.status == "ok" for s in segments[20:]), (
        "one failed batch must not degrade the rest (§8)"
    )


def test_a_degraded_segment_keeps_its_heuristic_analysis(context):
    segments = [_segment(index=1)]
    analyze_segments(segments, context, llm=_StubLLM(fail=True))
    assert segments[0].status == "degraded"
    assert segments[0].interpretation.strip(), "the heuristic floor must survive"
    assert segments[0].topic.strip()


def test_an_unknown_visual_intent_is_rejected_and_noted(context):
    segments = [_segment(index=1)]
    llm = _StubLLM({"segments": [{"index": 1, "visual_intent": "INTERPRETIVE_DANCE"}]})
    analyze_segments(segments, context, llm=llm)
    assert segments[0].visual_intent in set(VisualIntent)
    assert any("INTERPRETIVE_DANCE" in n for n in segments[0].notes)


def test_sub_shots_are_clamped_to_their_segment(context):
    segments = [_segment(index=1)]  # 229.0 - 237.0
    llm = _StubLLM(
        {
            "segments": [
                {
                    "index": 1,
                    "sub_shots": [
                        {"start": 100.0, "end": 500.0, "topic": "way out of range"},
                        {"start": 231.0, "end": 233.0, "topic": "fine"},
                    ],
                }
            ]
        }
    )
    analyze_segments(segments, context, llm=llm)
    for shot in segments[0].sub_shots:
        assert 229.0 <= shot.start <= shot.end <= 237.0


def test_heuristics_alone_still_produce_an_interpretation(context):
    segments = [_segment(index=1)]
    analyze_segments(segments, context, llm=None)
    assert segments[0].interpretation.strip()
    assert segments[0].confidence > 0
    assert segments[0].confidence <= 0.55, (
        "a heuristic analysis must not claim high confidence; §14 routes it to review"
    )


def test_heuristics_use_corrected_names(context, corrections):
    """§10.2 in the other direction: analysis matches on the resolved name."""
    segments = [_segment(index=1, narration="Barriss faces the Dark Council.")]
    analyze_segments(segments, context, llm=None, corrections=corrections)
    assert "Darth Baras" in segments[0].entities


def test_the_offline_fake_produces_valid_analysis_for_every_segment(context):
    segments = [_segment(index=i) for i in range(1, 6)]
    analyze_segments(segments, context, llm=FakeLLMProvider(fixture_dir=Path("nowhere")))
    for segment in segments:
        assert segment.topic.strip()
        assert segment.interpretation.strip()
        assert segment.visual_intent in set(VisualIntent)


# ---------------------------------------------------------------------------
# Query generation -- the P2 acceptance criterion
# ---------------------------------------------------------------------------


def test_every_segment_gets_at_least_four_queries_of_differing_kinds(context, corrections):
    """§22 P2: ">=4 queries/segment of differing kinds"."""
    segments = [
        _segment(index=1, narration="Barriss is unmasked before the Dark Council."),
        _segment(index=2, narration="Drog fell on Hoth, and the last witness died."),
        _segment(index=3, narration="Korriban holds its breath beneath the academy."),
        _segment(index=4, narration="The vote turns against him in that instant."),
    ]
    analyze_segments(segments, context, llm=None, corrections=corrections)
    generate_queries(segments, context, corrections)

    for segment in segments:
        assert len(segment.queries) >= MIN_QUERIES, (
            f"segment {segment.index} has only {len(segment.queries)} queries"
        )
        kinds = {str(q.kind) for q in segment.queries}
        assert len(kinds) >= MIN_QUERIES - 1, (
            f"segment {segment.index} has {len(segment.queries)} queries but only "
            f"{len(kinds)} distinct kinds: {sorted(kinds)} -- rewordings of one "
            "query return the same results"
        )


def test_queries_are_never_empty_or_whitespace(context, corrections):
    segments = [
        _segment(index=i, narration=n)
        for i, n in enumerate(
            [
                "Barriss is unmasked.",
                "It happened.",
                "Then everything changed forever.",
            ],
            start=1,
        )
    ]
    analyze_segments(segments, context, llm=None, corrections=corrections)
    generate_queries(segments, context, corrections)
    for segment in segments:
        for query in segment.queries:
            assert query.text.strip(), f"segment {segment.index} has an empty query"


def test_queries_use_the_corrected_spelling(context, corrections):
    """§10.2 -- corrections apply to search queries."""
    segment = _segment(entities=["Barriss"], visual_intent=VisualIntent.EXACT_EVENT)
    queries = build_segment_queries(segment, context, corrections)
    joined = " | ".join(q.text for q in queries)
    assert "Darth Baras" in joined
    assert "Barriss" not in joined, "the misheard spelling must not reach the search"


def test_a_low_confidence_correction_yields_both_spellings():
    """§10.3 -- below 0.75, search both and let ranking decide."""
    packs = [
        DomainPack(
            name="t",
            triggers=("x",),
            min_triggers=1,
            entities=(
                PackEntity(
                    canonical="Correct Name",
                    aliases=("Wrongname",),
                    confidence=0.55,
                    reason="uncertain",
                ),
            ),
        )
    ]
    corrections = resolve_entities("Wrongname was there. x", packs)
    context = ProjectContext(subject="Test Subject")
    segment = _segment(entities=["Wrongname"], visual_intent=VisualIntent.CHARACTER)
    queries = build_segment_queries(segment, context, corrections)
    joined = " | ".join(q.text for q in queries)
    assert "Wrongname" in joined
    assert "Correct Name" in joined, "both spellings must be searched below the threshold"


def test_queries_carry_the_subject_tag(context, corrections):
    """Without a qualifier the results drift into unrelated fiction."""
    segment = _segment(entities=["Darth Baras"], location="Korriban")
    queries = build_segment_queries(segment, context, corrections)
    tag = context.search_tag or context.franchise
    assert tag, "the context must supply some qualifier"
    assert any(tag in q.text for q in queries)


def test_a_youtube_query_is_generated_for_visual_intents_that_want_footage(context, corrections):
    for intent in (VisualIntent.EXACT_EVENT, VisualIntent.GAME_SCENE, VisualIntent.FILM_SCENE):
        segment = _segment(entities=["Darth Baras"], visual_intent=intent, event="unmasking")
        kinds = {str(q.kind) for q in build_segment_queries(segment, context, corrections)}
        assert "youtube" in kinds, f"{intent} should produce a youtube query"


def test_no_two_queries_have_identical_text(context, corrections):
    segment = _segment(entities=["Darth Baras"], location="Korriban", event="unmasking")
    queries = build_segment_queries(segment, context, corrections)
    texts = [q.text.lower() for q in queries]
    assert len(texts) == len(set(texts)), f"duplicate query text: {texts}"


def test_query_generation_is_deterministic(context, corrections):
    segment_a = _segment(entities=["Darth Baras"], location="Korriban")
    segment_b = _segment(entities=["Darth Baras"], location="Korriban")
    assert [
        (str(q.kind), q.text) for q in build_segment_queries(segment_a, context, corrections)
    ] == [(str(q.kind), q.text) for q in build_segment_queries(segment_b, context, corrections)]


def test_a_segment_with_nothing_to_go_on_still_gets_the_minimum(context):
    segment = _segment(narration="And then it happened, just like that.", entities=[])
    queries = build_segment_queries(segment, context, [])
    assert len(queries) >= MIN_QUERIES


def test_thin_segments_are_noted_not_silently_accepted(context):
    segment = Segment(index=1, start=0.0, end=8.0, narration="")
    generate_queries([segment], ProjectContext(), [])
    assert segment.notes, "a segment that could not be given proper queries must say so"


def test_the_search_tag_uses_the_packs_short_form(context):
    """§7's example queries read "... SWTOR", not "... Star Wars: The Old Republic"."""
    assert context.search_tag == "SWTOR"


def test_queries_carry_the_short_search_tag(context, corrections):
    segment = _segment(entities=["Darth Baras"], location="Korriban")
    joined = " | ".join(q.text for q in build_segment_queries(segment, context, corrections))
    assert "SWTOR" in joined
    assert "The Old Republic" not in joined, (
        "the long title makes queries worse; the pack's short tag should win"
    )


def test_the_tag_falls_back_to_the_franchise_when_no_pack_supplied_one():
    ctx = ProjectContext(franchise="Some Franchise", subject="A Very Long Subject Title Indeed")
    segment = _segment(entities=["Someone"])
    joined = " | ".join(q.text for q in build_segment_queries(segment, ctx, []))
    assert "Some Franchise" in joined


def test_no_tag_at_all_still_produces_queries():
    segment = _segment(entities=["Someone"], narration="Someone did a thing here today.")
    queries = build_segment_queries(segment, ProjectContext(), [])
    assert len(queries) >= MIN_QUERIES
    assert all(q.text.strip() for q in queries)
