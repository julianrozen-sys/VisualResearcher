"""Transcription tests (CLAUDE.md §21).

Word timestamps monotonic; silent audio does not crash; SRT is valid.
"""

from __future__ import annotations

import re

from visualresearcher.pipeline.transcribe import (
    enforce_monotonic,
    render_srt,
    transcribe,
)
from visualresearcher.project import ProjectPaths
from visualresearcher.providers.transcription.fake import (
    FakeTranscriptionProvider,
    audio_duration_s,
)
from visualresearcher.schemas import Transcript, TranscriptSegment, Word

SRT_BLOCK = re.compile(
    r"^(\d+)\n"
    r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
    r"(.+)$",
    re.DOTALL,
)


def _srt_seconds(stamp: str) -> float:
    hh, mm, rest = stamp.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


def test_fake_provider_reads_real_audio_duration(sample_wav):
    assert abs(audio_duration_s(sample_wav) - 60.0) < 0.05


def test_fake_transcription_word_timestamps_are_monotonic(sample_wav):
    transcript = FakeTranscriptionProvider().transcribe(sample_wav)
    words = transcript.all_words()
    assert len(words) > 50, "60s of narration should yield many words"
    for prev, nxt in zip(words[:-1], words[1:], strict=True):
        assert prev.end <= nxt.start, (
            f"word {prev.word!r} ends at {prev.end} but {nxt.word!r} starts at {nxt.start}"
        )
        assert prev.start <= prev.end


def test_fake_transcription_stays_inside_the_audio_duration(sample_wav):
    transcript = FakeTranscriptionProvider().transcribe(sample_wav)
    words = transcript.all_words()
    assert words[0].start >= 0.0
    assert words[-1].end <= 60.0 + 1e-6, "transcript must not run past the audio"


def test_fake_transcription_is_deterministic(sample_wav):
    first = FakeTranscriptionProvider().transcribe(sample_wav)
    second = FakeTranscriptionProvider().transcribe(sample_wav)
    assert first.model_dump() == second.model_dump()


def test_silent_audio_does_not_crash(silence_wav, settings, sandbox):
    """3s of digital silence must produce a transcript, not an exception."""
    paths = ProjectPaths(sandbox / "projects" / "silent")
    paths.ensure_base()
    transcript = transcribe(silence_wav, paths, settings)
    assert transcript is not None
    assert paths.transcript_json.exists()
    assert paths.narration_srt.exists()
    # A short file still gets timings that make sense.
    for word in transcript.all_words():
        assert 0.0 <= word.start <= word.end <= 3.0 + 1e-6


def test_unreadable_audio_degrades_instead_of_raising(tmp_path):
    """A file that is not a WAV must not kill the job (§8)."""
    bogus = tmp_path / "not-audio.wav"
    bogus.write_bytes(b"this is not a wav file at all")
    transcript = FakeTranscriptionProvider().transcribe(bogus)
    assert transcript.all_words(), "should fall back to a default duration"


def test_sidecar_fixture_is_replayed_when_present(tmp_path):
    """Tests pin exact transcripts with a sidecar; it must win over synthesis."""
    audio = tmp_path / "narration.wav"
    audio.write_bytes(b"\x00" * 16)
    pinned = Transcript(
        language="en",
        duration=2.0,
        text="pinned fixture text",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="pinned fixture text",
                words=[
                    Word(word="pinned", start=0.0, end=0.6),
                    Word(word="fixture", start=0.7, end=1.3),
                    Word(word="text", start=1.4, end=2.0),
                ],
            )
        ],
    )
    (tmp_path / "narration.wav.transcript.json").write_text(
        pinned.model_dump_json(indent=2), encoding="utf-8"
    )
    got = FakeTranscriptionProvider().transcribe(audio)
    assert got.text == "pinned fixture text"
    assert len(got.all_words()) == 3


# ---------------------------------------------------------------------------
# Monotonicity repair
# ---------------------------------------------------------------------------


def test_enforce_monotonic_repairs_overlapping_words():
    """Whisper sometimes emits overlapping words; segmentation cannot cope."""
    broken = Transcript(
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=3.0,
                text="a b c",
                words=[
                    Word(word="a", start=0.0, end=1.5),
                    Word(word="b", start=1.2, end=2.0),  # starts before 'a' ended
                    Word(word="c", start=1.9, end=1.4),  # ends before it starts
                ],
            )
        ]
    )
    fixed = enforce_monotonic(broken)
    words = fixed.all_words()
    for prev, nxt in zip(words[:-1], words[1:], strict=True):
        assert prev.end <= nxt.start
    for word in words:
        assert word.start <= word.end


# ---------------------------------------------------------------------------
# SRT validity
# ---------------------------------------------------------------------------


def test_srt_is_structurally_valid(sample_wav):
    transcript = FakeTranscriptionProvider().transcribe(sample_wav)
    srt = render_srt(transcript)
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert blocks, "SRT must not be empty"

    previous_end = -1.0
    for position, block in enumerate(blocks, start=1):
        match = SRT_BLOCK.match(block.strip())
        assert match, f"block {position} is malformed:\n{block!r}"
        index, start, end, text = match.groups()
        assert int(index) == position, "SRT cue numbers must be 1-based and sequential"
        start_s, end_s = _srt_seconds(start), _srt_seconds(end)
        assert end_s > start_s, f"cue {index} ends before it starts"
        assert start_s >= previous_end - 1e-6, f"cue {index} overlaps the previous cue"
        previous_end = end_s
        assert text.strip()


def test_srt_uses_comma_not_period_for_milliseconds(sample_wav):
    srt = render_srt(FakeTranscriptionProvider().transcribe(sample_wav))
    first_stamp = srt.split("\n")[1]
    assert "," in first_stamp and "-->" in first_stamp
    assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", first_stamp), (
        "SRT requires a comma decimal separator; a period is WebVTT"
    )


def test_srt_skips_empty_cues():
    transcript = Transcript(
        segments=[
            TranscriptSegment(id=0, start=0.0, end=1.0, text="real text"),
            TranscriptSegment(id=1, start=1.0, end=2.0, text="   "),
            TranscriptSegment(id=2, start=2.0, end=3.0, text="more text"),
        ]
    )
    srt = render_srt(transcript)
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert len(blocks) == 2
    assert blocks[1].startswith("2"), "numbering must stay contiguous after a skip"


# ---------------------------------------------------------------------------
# Stage behaviour
# ---------------------------------------------------------------------------


def test_transcript_on_disk_matches_the_provider_output_exactly(sample_wav, settings, sandbox):
    """§10.2: the transcript on disk keeps the ORIGINAL words, verbatim.

    Compared field by field against the provider's own output so that any
    later stage which starts rewriting the transcript fails here.
    """
    paths = ProjectPaths(sandbox / "projects" / "orig")
    paths.ensure_base()
    written = transcribe(sample_wav, paths, settings)

    raw = FakeTranscriptionProvider().transcribe(sample_wav)
    on_disk = Transcript.model_validate_json(paths.transcript_json.read_text(encoding="utf-8"))

    assert on_disk.text == raw.text
    assert [w.word for w in on_disk.all_words()] == [w.word for w in raw.all_words()]
    assert on_disk.text == written.text
    assert paths.transcript_txt.read_text(encoding="utf-8").strip() == raw.text.strip()


def test_transcribe_skips_when_artifact_exists(sample_wav, settings, sandbox):
    paths = ProjectPaths(sandbox / "projects" / "skip")
    paths.ensure_base()
    transcribe(sample_wav, paths, settings)
    first_mtime = paths.transcript_json.stat().st_mtime_ns

    transcribe(sample_wav, paths, settings)
    assert paths.transcript_json.stat().st_mtime_ns == first_mtime, (
        "second run should have skipped rather than rewritten the transcript"
    )


def test_force_rewrites_and_keeps_a_backup(sample_wav, settings, sandbox):
    paths = ProjectPaths(sandbox / "projects" / "forced")
    paths.ensure_base()
    transcribe(sample_wav, paths, settings)
    transcribe(sample_wav, paths, settings, force=True)
    assert paths.transcript_json.with_suffix(".json.bak").exists(), (
        "§2.8 requires a .bak before overwriting user-visible data"
    )
