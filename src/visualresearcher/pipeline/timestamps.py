"""Stage 12: timestamp location (CLAUDE.md §12.3).

Given a video that probably contains the moment a segment describes, find
*where*. §12.3 fixes both the order of evidence and the requirement to record
which one won:

    captions -> chapters -> description -> metadata -> text/entity similarity

The order is a confidence ranking. A caption line naming the subject is
near-direct evidence; a chapter title is the uploader's own labelling; a
timestamp in the description is usually right but sometimes stale; metadata is
weak; falling back to similarity means guessing from the title alone.

Every candidate records ``method``, ``evidence`` and ``confidence``, because a
timestamp the user cannot audit is a timestamp they have to check by hand --
which defeats the point.
"""

from __future__ import annotations

import re

from ..logging_setup import get_logger
from ..providers.video.base import VideoHit
from ..schemas import Segment, TimestampCandidate

__all__ = [
    "locate_timestamps",
    "from_captions",
    "from_chapters",
    "from_description",
    "from_metadata",
    "from_similarity",
    "METHOD_CONFIDENCE",
    "parse_description_timestamps",
]

log = get_logger("pipeline.timestamps")

#: Ceiling on the confidence each method may claim (§12.3's ordering).
METHOD_CONFIDENCE = {
    "captions": 0.92,
    "chapters": 0.80,
    "description": 0.70,
    "metadata": 0.50,
    "similarity": 0.35,
}

#: ``1:23``, ``01:23``, ``1:02:03`` at the start of a description line.
_TIMESTAMP_LINE = re.compile(
    r"^\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{2})\)?\s*[-–—:.)\]]*\s*(.+?)\s*$", re.MULTILINE
)

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2}


def _terms_for(segment: Segment) -> list[str]:
    """The words worth looking for: entities, event, location, topic."""
    terms: list[str] = []
    terms.extend(segment.entities)
    if segment.event:
        terms.append(segment.event)
    if segment.location:
        terms.append(segment.location)
    if segment.topic:
        terms.append(segment.topic)
    return [t for t in terms if t and len(t) > 2]


def _overlap(text: str, terms: list[str]) -> tuple[float, list[str]]:
    """Fraction of ``terms`` present in ``text``, plus which ones matched."""
    if not terms:
        return 0.0, []
    lowered = (text or "").lower()
    matched = [t for t in terms if t.lower() in lowered]
    if not matched:
        # Fall back to token overlap so a paraphrase still counts for something.
        term_tokens = _tokens(" ".join(terms))
        text_tokens = _tokens(text)
        shared = term_tokens & text_tokens
        if term_tokens and len(shared) >= 2:
            return len(shared) / len(term_tokens) * 0.6, sorted(shared)
        return 0.0, []
    return len(matched) / len(terms), matched


# ---------------------------------------------------------------------------
# The five methods
# ---------------------------------------------------------------------------


def from_captions(hit: VideoHit, segment: Segment) -> TimestampCandidate | None:
    """Strongest evidence: a caption line that names what the segment describes."""
    terms = _terms_for(segment)
    if not hit.captions or not terms:
        return None

    best_score = 0.0
    best_line = None
    for line in hit.captions:
        score, matched = _overlap(str(line.get("text", "")), terms)
        if score > best_score:
            best_score, best_line = score, (line, matched)
    if not best_line or best_score <= 0:
        return None

    line, matched = best_line
    start = float(line.get("start", 0.0))
    return TimestampCandidate(
        start_s=start,
        end_s=float(line.get("end", start + segment.duration)),
        method="captions",
        evidence=f"caption at {start:.1f}s matches {', '.join(matched[:3])}",
        confidence=round(METHOD_CONFIDENCE["captions"] * best_score, 3),
        url_with_t=hit.url_with_t(start),
    )


def from_chapters(hit: VideoHit, segment: Segment) -> TimestampCandidate | None:
    """The uploader's own labelling of the video's structure."""
    terms = _terms_for(segment)
    if not hit.chapters or not terms:
        return None

    best_score = 0.0
    best = None
    for chapter in hit.chapters:
        score, matched = _overlap(str(chapter.get("title", "")), terms)
        if score > best_score:
            best_score, best = score, (chapter, matched)
    if not best or best_score <= 0:
        return None

    chapter, matched = best
    start = float(chapter.get("start_time", 0.0))
    return TimestampCandidate(
        start_s=start,
        end_s=float(chapter.get("end_time", start + segment.duration)),
        method="chapters",
        evidence=f"chapter {chapter.get('title', '')!r} matches {', '.join(matched[:3])}",
        confidence=round(METHOD_CONFIDENCE["chapters"] * best_score, 3),
        url_with_t=hit.url_with_t(start),
    )


def parse_description_timestamps(description: str) -> list[tuple[float, str]]:
    """Pull ``mm:ss Label`` lines out of a description."""
    out: list[tuple[float, str]] = []
    for match in _TIMESTAMP_LINE.finditer(description or ""):
        stamp, label = match.group(1), match.group(2)
        parts = [int(p) for p in stamp.split(":")]
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + part
        if label.strip():
            out.append((float(seconds), label.strip()))
    return out


def from_description(hit: VideoHit, segment: Segment) -> TimestampCandidate | None:
    """Creator-written chapter lists, which are common and usually right."""
    terms = _terms_for(segment)
    entries = parse_description_timestamps(hit.description)
    if not entries or not terms:
        return None

    best_score = 0.0
    best = None
    for seconds, label in entries:
        score, matched = _overlap(label, terms)
        if score > best_score:
            best_score, best = score, (seconds, label, matched)
    if not best or best_score <= 0:
        return None

    seconds, label, matched = best
    return TimestampCandidate(
        start_s=seconds,
        end_s=seconds + segment.duration,
        method="description",
        evidence=f"description line {label!r} at {seconds:.0f}s matches {', '.join(matched[:3])}",
        confidence=round(METHOD_CONFIDENCE["description"] * best_score, 3),
        url_with_t=hit.url_with_t(seconds),
    )


def from_metadata(hit: VideoHit, segment: Segment) -> TimestampCandidate | None:
    """Weak evidence: a short video whose title matches is probably all relevant.

    Only fires for videos short enough that "somewhere near the start" is a
    defensible answer. For a 40-minute playthrough it is not, so it declines.
    """
    terms = _terms_for(segment)
    score, matched = _overlap(hit.title, terms)
    if score <= 0 or hit.duration <= 0 or hit.duration > 300:
        return None

    # Skip a typical intro rather than starting at zero.
    start = min(5.0, max(0.0, hit.duration * 0.05))
    return TimestampCandidate(
        start_s=start,
        end_s=min(hit.duration, start + segment.duration),
        method="metadata",
        evidence=(
            f"short video ({hit.duration:.0f}s) whose title matches {', '.join(matched[:3])}"
        ),
        confidence=round(METHOD_CONFIDENCE["metadata"] * score, 3),
        url_with_t=hit.url_with_t(start),
    )


def from_similarity(hit: VideoHit, segment: Segment) -> TimestampCandidate | None:
    """Last resort: title and description similarity, with no positional evidence.

    Deliberately low-confidence. It exists so a segment gets *a* clickable
    link rather than nothing, not so the clip downloader acts on it.
    """
    terms = _terms_for(segment)
    score, matched = _overlap(f"{hit.title} {hit.description}", terms)
    if score <= 0:
        return None
    start = 0.0
    return TimestampCandidate(
        start_s=start,
        end_s=segment.duration,
        method="similarity",
        evidence=(
            f"no positional evidence; title/description mention "
            f"{', '.join(matched[:3])} -- start of video assumed"
        ),
        confidence=round(METHOD_CONFIDENCE["similarity"] * score, 3),
        url_with_t=hit.url_with_t(start),
    )


_METHODS = (from_captions, from_chapters, from_description, from_metadata, from_similarity)


def locate_timestamps(hit: VideoHit, segment: Segment) -> list[TimestampCandidate]:
    """Every method's answer, best first.

    All five are run rather than stopping at the first hit: the review UI
    shows them side by side, and a user who disagrees with the winner needs
    the alternatives to choose from.
    """
    candidates: list[TimestampCandidate] = []
    for method in _METHODS:
        try:
            candidate = method(hit, segment)
        except Exception as exc:  # noqa: BLE001 - one method must not break the rest
            log.debug("timestamp method %s failed: %s", method.__name__, exc)
            continue
        if candidate is not None:
            # Clamp to the video, so a stale description cannot ask for a span
            # past the end of the file.
            if hit.duration:
                candidate.start_s = max(0.0, min(candidate.start_s, hit.duration - 0.1))
                candidate.end_s = max(candidate.start_s, min(candidate.end_s, hit.duration))
            candidates.append(candidate)

    candidates.sort(key=lambda c: -c.confidence)
    if candidates:
        log.debug(
            "video %s: best timestamp %.1fs via %s (%.2f)",
            hit.video_id,
            candidates[0].start_s,
            candidates[0].method,
            candidates[0].confidence,
        )
    return candidates
