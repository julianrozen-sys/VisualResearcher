"""Segmentation tests (CLAUDE.md §21).

The filename-ordering test here is one of the two the brief singles out, so it
is written to fail loudly on the classic bug: ``str(index)`` without
zero-padding, where segment 100 sorts before segment 99.
"""

from __future__ import annotations

import random

import pytest

from visualresearcher.pipeline.segment import (
    BoundaryKind,
    classify_boundary,
    segment_transcript,
)
from visualresearcher.schemas import Segment
from visualresearcher.utils.files import pad_width, segment_dirname, selected_filename

from .conftest import make_transcript

# 40 sentences x ~7 words x 0.35s + pauses is a bit over four minutes: long
# enough to exercise the optimiser properly, short enough to stay fast.
SENTENCES = [
    "The Sith Warrior walks into the council chamber alone.",
    "Every member of the Dark Council is already seated.",
    "Baras stands at the centre wearing that heavy mask.",
    "He claims to speak with the authority of the Emperor.",
    "Nobody in the room is willing to challenge him openly.",
    "The apprentice has returned from Korriban carrying proof.",
    "Vowrawn watches from the far side and says nothing at all.",
    "The accusation is finally spoken out loud in the chamber.",
    "Baras reaches for his lightsaber in the same moment.",
    "The vote turns against him before the blade is drawn.",
] * 4


@pytest.fixture
def transcript():
    return make_transcript(SENTENCES)


def test_all_segments_within_configured_bounds(transcript, settings):
    segments = segment_transcript(transcript, settings)
    cfg = settings.segmentation
    assert len(segments) > 10, "fixture should produce many segments"
    for seg in segments:
        assert cfg.min_s <= seg.duration <= cfg.max_s, (
            f"segment {seg.index} is {seg.duration:.2f}s, outside "
            f"{cfg.min_s}-{cfg.max_s}s: {seg.narration!r}"
        )


def test_segments_are_contiguous_and_gapless(transcript, settings):
    segments = segment_transcript(transcript, settings)
    for prev, nxt in zip(segments[:-1], segments[1:], strict=True):
        assert prev.end == nxt.start, (
            f"gap between segment {prev.index} (ends {prev.end}) and "
            f"{nxt.index} (starts {nxt.start})"
        )
    assert segments[0].index == 1
    assert [s.index for s in segments] == list(range(1, len(segments) + 1))


def test_no_segment_boundary_falls_inside_a_word(transcript, settings):
    """A cut must never land in the middle of a spoken word."""
    segments = segment_transcript(transcript, settings)
    words = transcript.all_words()
    interior = [(w.start, w.end) for w in words]
    for seg in segments[1:]:
        for start, end in interior:
            assert not (start < seg.start < end), (
                f"segment {seg.index} starts at {seg.start}, inside the word spanning {start}-{end}"
            )


def test_boundaries_prefer_sentence_ends_over_hitting_the_target(transcript, settings):
    """Rule three: never split mid-phrase just to reach 8s."""
    segments = segment_transcript(transcript, settings)
    words = transcript.all_words()

    # Map each interior boundary back to the word index it cut at.
    cut_kinds: list[str] = []
    for seg in segments[1:]:
        cut = next(
            (i for i, w in enumerate(words) if w.start >= seg.start - 1e-6),
            None,
        )
        assert cut is not None
        cut_kinds.append(classify_boundary(words, cut))

    mid_phrase = [k for k in cut_kinds if k == BoundaryKind.WORD]
    sentence = [k for k in cut_kinds if k == BoundaryKind.SENTENCE]
    assert not mid_phrase, (
        f"{len(mid_phrase)} of {len(cut_kinds)} boundaries were mid-phrase; "
        "the optimiser is favouring duration over sentence ends"
    )
    assert len(sentence) == len(cut_kinds)


def test_silence_gap_counts_as_a_boundary(settings):
    """A long pause is a legal boundary even without punctuation (§9).

    Sentences are 15 words at 0.5s = 7.5s, so every pause lands squarely
    inside the 6-10s window and the optimiser has a real choice to make.
    """
    transcript = make_transcript(
        [" ".join(f"w{i:02d}" for i in range(15))] * 6,
        word_s=0.5,
        pause_s=0.8,
    )
    segments = segment_transcript(transcript, settings)
    words = transcript.all_words()
    assert len(segments) >= 5
    for seg in segments[1:]:
        cut = next(i for i, w in enumerate(words) if w.start >= seg.start - 1e-6)
        kind = classify_boundary(words, cut)
        assert kind in {BoundaryKind.SENTENCE, BoundaryKind.SILENCE}, (
            f"segment {seg.index} cut at a {kind} boundary although a "
            "silence was available in range"
        )


def test_no_natural_boundary_in_range_falls_back_to_a_word_boundary(settings):
    """When nothing better exists the cut is still between words, never inside one.

    One 40-word sentence with no pauses: the optimiser has no sentence,
    clause or silence boundary to reach for, and must not invent one.
    """
    transcript = make_transcript(
        [" ".join(f"w{i:02d}" for i in range(40))], word_s=0.5, pause_s=0.0
    )
    segments = segment_transcript(transcript, settings)
    words = transcript.all_words()
    assert len(segments) >= 2
    for seg in segments[1:]:
        assert any(abs(w.start - seg.start) < 1e-6 for w in words), (
            f"segment {seg.index} starts at {seg.start}, which is not a word boundary"
        )


def test_awkward_total_duration_lets_the_tail_run_short_not_long(settings):
    """11s of unbroken speech cannot split into two 6-10s halves.

    The rule that must hold is that no segment exceeds max_s. The correct
    concession is a short final segment, not an over-long earlier one.
    """
    transcript = make_transcript(
        [" ".join(f"w{i:02d}" for i in range(22))], word_s=0.5, pause_s=0.0
    )
    total = transcript.all_words()[-1].end - transcript.all_words()[0].start
    assert abs(total - 11.0) < 1e-6, "fixture should be exactly 11s"

    segments = segment_transcript(transcript, settings)
    assert len(segments) == 2, f"expected a 2-way split, got {len(segments)}"
    for seg in segments:
        assert seg.duration <= settings.segmentation.max_s + 1e-6, (
            f"segment {seg.index} is {seg.duration}s, over max_s"
        )
    assert segments[0].duration >= settings.segmentation.min_s
    assert segments[-1].duration < settings.segmentation.min_s
    for prev, nxt in zip(segments[:-1], segments[1:], strict=True):
        assert prev.end == nxt.start


def test_empty_transcript_produces_no_segments(settings):
    from visualresearcher.schemas import Transcript

    assert segment_transcript(Transcript(), settings) == []


# ---------------------------------------------------------------------------
# Filename ordering -- the test the brief calls out by name.
# ---------------------------------------------------------------------------


def _fake_segments(count: int) -> list[Segment]:
    return [Segment(index=i, start=float(i * 8), end=float(i * 8 + 8)) for i in range(1, count + 1)]


@pytest.mark.parametrize("count", [300, 999])
def test_segment_folder_names_sort_chronologically_as_strings(count):
    segments = _fake_segments(count)
    width = pad_width(count)
    names = [segment_dirname(s.index, s.start, s.end, width) for s in segments]

    shuffled = names[:]
    random.Random(0).shuffle(shuffled)
    assert sorted(shuffled) == names, (
        f"lexicographic order diverged from chronological order at {count} segments"
    )


@pytest.mark.parametrize("count", [300, 1500])
def test_selected_filenames_sort_chronologically_as_strings(count):
    """``selected/`` is flat, so string order IS the drag-into-CapCut order (§6)."""
    segments = _fake_segments(count)
    width = pad_width(count)
    names = [selected_filename(s.index, 1, f"topic {s.index}", "jpg", width) for s in segments]

    shuffled = names[:]
    random.Random(1).shuffle(shuffled)
    assert sorted(shuffled) == names, (
        f"selected/ filenames stopped sorting chronologically at {count} segments"
    )


def test_padding_is_three_digits_for_normal_projects():
    """§6 specifies {segment:03d}; that must hold for every realistic size."""
    assert pad_width(1) == 3
    assert pad_width(150) == 3
    assert pad_width(999) == 3
    # Past 999 the width has to grow or lexicographic ordering breaks.
    assert pad_width(1000) == 4


def test_three_digit_padding_would_break_past_999():
    """Guards the reason pad_width widens: proves the naive version is wrong."""
    names = [f"{i:03d}_x" for i in (99, 100, 1000, 1001)]
    assert sorted(names) != names, (
        "expected fixed 3-digit padding to mis-sort past 999 -- if this passes, "
        "pad_width's widening is unnecessary"
    )


def test_segment_dirname_matches_the_documented_shape():
    assert segment_dirname(31, 229.0, 237.0, 3) == "031_03m49s-03m57s"


def test_selected_filename_matches_the_documented_shape():
    assert (
        selected_filename(31, 1, "Darth Baras Dark Council", "JPG", 3)
        == "031_1_darth_baras_dark_council.jpg"
    )
