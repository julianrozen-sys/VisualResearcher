"""Stage 5: segmentation (CLAUDE.md §9).

Four constraints have to hold simultaneously:

* every segment 6-10s, aiming at 8s;
* boundaries at semantic joints -- sentence end > clause end > word gap >=250ms;
* **never split mid-phrase just to hit 8s**;
* contiguous and gapless.

A greedy left-to-right chop cannot do this: taking the best boundary now
regularly strands the next window with nothing but mid-phrase options. So this
is a dynamic program over word indices, minimising

    boundary_penalty  +  DURATION_WEIGHT * ((duration - target) / target)^2

Because a sentence end scores 0.0 and a mid-phrase cut scores 1.2, while the
worst in-bounds duration error contributes about 0.09, the optimiser will
always move several seconds off target to land on a sentence end. That
ordering *is* rule three, expressed as arithmetic.

Cost is O(words x window) -- about 40 candidate cuts per word -- so a
30-minute narration segments in well under a second.
"""

from __future__ import annotations

import math
import re

from ..config import Settings
from ..logging_setup import get_logger
from ..schemas import Segment, Transcript, Word

__all__ = [
    "segment_transcript",
    "boundary_penalty",
    "BoundaryKind",
    "classify_boundary",
    "DURATION_WEIGHT",
    "SILENCE_GAP_S",
]

log = get_logger("pipeline.segment")

#: How much duration error matters relative to boundary quality. Deliberately
#: small: rule three says a natural boundary beats hitting the target.
DURATION_WEIGHT = 0.35

#: A pause this long is a real breath, not just inter-word spacing (§9).
SILENCE_GAP_S = 0.250

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")
_CLAUSE_END = re.compile(r"[,;:–—][\"')\]]*$")


class BoundaryKind:
    SENTENCE = "sentence"
    CLAUSE = "clause"
    SILENCE = "silence"
    WORD = "word"


#: Penalties, ordered exactly as §9 ranks the boundary types.
_PENALTY = {
    BoundaryKind.SENTENCE: 0.00,
    BoundaryKind.CLAUSE: 0.35,
    BoundaryKind.SILENCE: 0.70,
    BoundaryKind.WORD: 1.20,
}


def classify_boundary(words: list[Word], cut: int) -> str:
    """Describe the joint between ``words[cut-1]`` and ``words[cut]``.

    ``cut`` is an exclusive end index, so it always falls *between* two words --
    a word is never sliced in half by construction.
    """
    if cut <= 0 or cut >= len(words):
        return BoundaryKind.SENTENCE  # the ends of the audio are hard boundaries
    prev, nxt = words[cut - 1], words[cut]
    token = prev.word.strip()
    if _SENTENCE_END.search(token):
        return BoundaryKind.SENTENCE
    if _CLAUSE_END.search(token):
        return BoundaryKind.CLAUSE
    if (nxt.start - prev.end) >= SILENCE_GAP_S:
        return BoundaryKind.SILENCE
    return BoundaryKind.WORD


def boundary_penalty(words: list[Word], cut: int) -> float:
    kind = classify_boundary(words, cut)
    penalty = _PENALTY[kind]
    if kind == BoundaryKind.WORD and 0 < cut < len(words):
        # Among equally mid-phrase options, a longer pause is still less bad.
        gap = max(0.0, words[cut].start - words[cut - 1].end)
        penalty -= min(0.25, gap)
    return penalty


def _cut_time(words: list[Word], cut: int) -> float:
    """The instant to cut at: the middle of the silence between two words.

    Using one value for both the end of segment N and the start of segment N+1
    is what makes the output contiguous and gapless by construction.
    """
    if cut <= 0:
        return words[0].start
    if cut >= len(words):
        return words[-1].end
    prev_end = words[cut - 1].end
    next_start = words[cut].start
    if next_start <= prev_end:
        return prev_end
    return prev_end + (next_start - prev_end) / 2.0


def _solve(
    words: list[Word], target: float, min_s: float, max_s: float, *, relax_last: bool
) -> list[int] | None:
    """Optimal cut indices, or None when no segmentation satisfies the bounds.

    Returns exclusive end indices, always ending with ``len(words)``.
    """
    n = len(words)
    inf = math.inf
    dp = [inf] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for end in range(1, n + 1):
        end_t = _cut_time(words, end)
        is_final = end == n
        lo = 1.0 if (is_final and relax_last) else min_s
        for start in range(end - 1, -1, -1):
            if dp[start] == inf:
                continue
            duration = end_t - _cut_time(words, start)
            if duration > max_s:
                break  # earlier starts only make the segment longer
            if duration < lo:
                continue
            error = (duration - target) / target
            cost = dp[start] + boundary_penalty(words, end) + DURATION_WEIGHT * error * error
            if cost < dp[end]:
                dp[end] = cost
                back[end] = start
    if dp[n] == inf:
        return None

    cuts: list[int] = []
    node = n
    while node > 0:
        cuts.append(node)
        node = back[node]
    return list(reversed(cuts))


def _narration_for(words: list[Word], start: int, end: int) -> str:
    return " ".join(w.word.strip() for w in words[start:end]).strip()


def segment_transcript(transcript: Transcript, settings: Settings) -> list[Segment]:
    """Split a transcript into contiguous 6-10s segments on semantic boundaries.

    Falls back to the transcriber's own segments when no word timings exist,
    and to a single segment for audio shorter than ``max_s``.
    """
    cfg = settings.segmentation
    words = transcript.all_words()

    if not words:
        return _segment_without_words(transcript, settings)

    total = _cut_time(words, len(words)) - _cut_time(words, 0)
    if total <= cfg.max_s:
        log.info("narration is %.1fs; emitting a single segment", total)
        return [_build(1, words, 0, len(words))]

    cuts = _solve(words, cfg.target_s, cfg.min_s, cfg.max_s, relax_last=False)
    if cuts is None:
        # A remainder that cannot be made to fit, e.g. 11s of audio: no pair of
        # in-bounds segments exists. Let the final segment run short rather
        # than stretch an earlier one past max_s.
        log.info("no strictly in-bounds segmentation; relaxing the final segment's minimum")
        cuts = _solve(words, cfg.target_s, cfg.min_s, cfg.max_s, relax_last=True)
    if cuts is None:
        log.warning("segmentation infeasible within bounds; emitting one segment")
        return [_build(1, words, 0, len(words))]

    segments: list[Segment] = []
    start = 0
    for index, end in enumerate(cuts, start=1):
        segments.append(_build(index, words, start, end))
        start = end

    _log_summary(segments, cfg.min_s, cfg.max_s, words, cuts)
    return segments


def _build(index: int, words: list[Word], start: int, end: int) -> Segment:
    return Segment(
        index=index,
        start=round(_cut_time(words, start), 3),
        end=round(_cut_time(words, end), 3),
        narration=_narration_for(words, start, end),
    )


def _segment_without_words(transcript: Transcript, settings: Settings) -> list[Segment]:
    """Fallback when the transcriber gave no word timings.

    Groups the transcriber's own segments up to ``max_s``. Boundaries are
    whatever the transcriber chose, which is worse but never wrong.
    """
    log.warning("no word timestamps available; grouping transcriber segments instead")
    cfg = settings.segmentation
    segments: list[Segment] = []
    bucket: list = []
    for seg in transcript.segments:
        if bucket and (seg.end - bucket[0].start) > cfg.max_s:
            segments.append(_from_bucket(len(segments) + 1, bucket))
            bucket = []
        bucket.append(seg)
    if bucket:
        segments.append(_from_bucket(len(segments) + 1, bucket))
    for prev, nxt in zip(segments, segments[1:], strict=False):
        nxt.start = prev.end  # keep it gapless
    return segments


def _from_bucket(index: int, bucket: list) -> Segment:
    return Segment(
        index=index,
        start=round(bucket[0].start, 3),
        end=round(bucket[-1].end, 3),
        narration=" ".join(s.text.strip() for s in bucket).strip(),
    )


def _log_summary(
    segments: list[Segment], min_s: float, max_s: float, words: list[Word], cuts: list[int]
) -> None:
    durations = [s.duration for s in segments]
    kinds: dict[str, int] = {}
    for cut in cuts[:-1]:
        kind = classify_boundary(words, cut)
        kinds[kind] = kinds.get(kind, 0) + 1
    out_of_bounds = [s.index for s in segments if not (min_s <= s.duration <= max_s)]
    log.info(
        "segmented into %d segments (%.1f-%.1fs, mean %.1fs); boundaries: %s",
        len(segments),
        min(durations),
        max(durations),
        sum(durations) / len(durations),
        ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none",
    )
    if out_of_bounds:
        log.warning("segments outside %.1f-%.1fs: %s", min_s, max_s, out_of_bounds)


def materialize_segments(
    segments: list[Segment], paths, *, width: int | None = None
) -> list[Segment]:
    """Create each segment folder and write its ``segment.json`` (§6).

    The zero-pad width is computed once from the highest index so that a
    lexicographic sort of the folder names is chronological for every project
    size, not just those under 1000 segments.
    """
    from ..utils.files import atomic_write_text, pad_width

    if width is None:
        width = pad_width(max((s.index for s in segments), default=1))
    paths.ensure_base()
    for segment in segments:
        paths.ensure_segment(segment, width=width)
        atomic_write_text(
            paths.segment_json(segment, width=width), segment.model_dump_json(indent=2)
        )
    log.info("wrote %d segment folders under %s", len(segments), paths.segments_dir)
    return segments


def write_timeline_csv(segments: list[Segment], path, *, width: int | None = None) -> None:
    """``timeline.csv`` -- the chronological index of the whole project (§6)."""
    import csv
    import io

    from ..utils.files import atomic_write_text, pad_width
    from ..utils.timefmt import timecode

    if width is None:
        width = pad_width(max((s.index for s in segments), default=1))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["segment", "folder", "start_s", "end_s", "duration_s", "timecode", "narration"]
    )
    for segment in segments:
        from ..utils.files import segment_dirname

        writer.writerow(
            [
                f"{segment.index:0{width}d}",
                segment_dirname(segment.index, segment.start, segment.end, width),
                f"{segment.start:.3f}",
                f"{segment.end:.3f}",
                f"{segment.duration:.3f}",
                timecode(segment.start),
                segment.narration,
            ]
        )
    atomic_write_text(path, buffer.getvalue())
    log.info("wrote %s", getattr(path, "name", path))
