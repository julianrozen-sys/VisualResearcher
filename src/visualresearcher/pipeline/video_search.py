"""Stage 11: YouTube search and classification (CLAUDE.md §12.1-12.2).

Searches with ``ytsearchN:`` -- no key, no quota -- and classifies every hit
into one of §7's four buckets **with a stated reason**.

The reason is not decoration. "EXACT_SCENE" with no justification is a claim
the user has to verify by watching the video, which is the work this tool
exists to remove. Every classification here names the evidence that produced
it.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..logging_setup import get_logger
from ..providers.base import ProviderError
from ..providers.video.base import VideoHit, VideoProvider
from ..schemas import Classification, QueryKind, Segment, YouTubeRecord

__all__ = ["search_videos", "classify_hit", "score_relevance"]

log = get_logger("pipeline.video_search")

#: Title words that promise the actual footage.
_EXACT_MARKERS = (
    "cutscene",
    "cut scene",
    "full scene",
    "scene",
    "cinematic",
    "trailer",
    "all cutscenes",
    "movie",
    "story",
    "ending",
    "finale",
    "confrontation",
)

#: Title words that mean somebody talking over the footage, or about it.
_COMMENTARY_MARKERS = (
    "reaction",
    "review",
    "explained",
    "analysis",
    "breakdown",
    "theory",
    "commentary",
    "discussion",
    "podcast",
    "top 10",
    "ranking",
    "tier list",
    "vs ",
    "versus",
    "guide",
    "tips",
    "build",
    "how to",
)

#: Title words that mean the footage is there but buried in a longer recording.
_PLAYTHROUGH_MARKERS = (
    "let's play",
    "lets play",
    "playthrough",
    "walkthrough",
    "gameplay",
    "part ",
    "episode ",
    "ep.",
    "stream",
    "vod",
    "full game",
    "longplay",
)

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2}


def score_relevance(hit: VideoHit, segment: Segment) -> tuple[float, list[str]]:
    """0.0..1.0 plus the terms that matched, from title and description."""
    terms = [t for t in (*segment.entities, segment.event, segment.location) if t and len(t) > 2]
    if not terms:
        terms = [w for w in segment.topic.split() if len(w) > 3][:4]
    if not terms:
        return 0.0, []

    haystack = f"{hit.title} {hit.description}".lower()
    matched = [t for t in terms if t.lower() in haystack]
    if matched:
        return len(matched) / len(terms), matched

    shared = _tokens(" ".join(terms)) & _tokens(hit.title)
    if shared:
        return 0.5 * len(shared) / max(1, len(_tokens(" ".join(terms)))), sorted(shared)
    return 0.0, []


def classify_hit(hit: VideoHit, segment: Segment) -> tuple[Classification, str, float]:
    """Return ``(classification, reason, relevance)`` (§12.2).

    Reads the title first, because that is what the uploader chose to call it
    and it is the single most reliable signal about what the video *is*.
    """
    relevance, matched = score_relevance(hit, segment)
    title = (hit.title or "").lower()
    matched_text = ", ".join(matched[:3]) if matched else "nothing specific"

    is_commentary = any(marker in title for marker in _COMMENTARY_MARKERS)
    is_playthrough = any(marker in title for marker in _PLAYTHROUGH_MARKERS)
    is_exact = any(marker in title for marker in _EXACT_MARKERS)
    has_chapters = bool(hit.chapters)
    is_short = 0 < hit.duration <= 600

    # Relevance is checked before the title markers. "How to bake sourdough
    # bread" trips the commentary marker "how to", and calling it commentary
    # would be a true-but-useless reason: the real problem is that nothing in
    # the segment appears in it at all.
    if relevance <= 0:
        return (
            Classification.CONTEXTUAL,
            "No term from this segment appears in the title or description.",
            relevance,
        )

    if is_commentary:
        return (
            Classification.CONTEXTUAL,
            f"Title reads as commentary rather than footage; matches {matched_text}.",
            relevance,
        )

    if is_exact and relevance >= 0.5 and is_short:
        return (
            Classification.EXACT_SCENE,
            (
                f"Title names the scene and matches {matched_text}; "
                f"{hit.duration:.0f}s long, so it is the clip rather than a compilation."
            ),
            relevance,
        )

    if is_exact and relevance >= 0.5:
        return (
            Classification.LIKELY_EXACT_SCENE,
            (
                f"Title names the scene and matches {matched_text}, but the video is "
                f"{hit.duration:.0f}s long, so the moment has to be located within it."
            ),
            relevance,
        )

    if is_playthrough:
        return (
            Classification.RELATED_FOOTAGE,
            (
                f"Playthrough footage matching {matched_text}"
                + (
                    "; it has chapters, so the moment can be located."
                    if has_chapters
                    else "; no chapters, so the timestamp is less certain."
                )
            ),
            relevance,
        )

    if relevance >= 0.5:
        return (
            Classification.LIKELY_EXACT_SCENE,
            f"Strong term match on {matched_text}, though the title does not say what it shows.",
            relevance,
        )

    return (
        Classification.RELATED_FOOTAGE,
        f"Partial match on {matched_text}; same subject area, not necessarily this moment.",
        relevance,
    )


def search_videos(
    segment: Segment,
    provider: VideoProvider,
    settings: Settings,
    *,
    limit: int = 5,
) -> tuple[list[YouTubeRecord], list[str]]:
    """Search this segment's YouTube-kind queries and classify the results.

    Returns ``(records, notes)``. A provider failure is recorded in ``notes``
    and the segment continues (§8).
    """
    from .timestamps import locate_timestamps

    queries = [q for q in segment.queries if str(q.kind) == str(QueryKind.YOUTUBE)]
    if not queries:
        # No dedicated video query: fall back to the most specific one there is.
        queries = segment.queries[:1]
    if not queries:
        return [], ["no queries to search video with"]

    notes: list[str] = []
    records: list[YouTubeRecord] = []
    seen: set[str] = set()

    for query in queries:
        try:
            hits = provider.search(query.text, limit=limit)
        except ProviderError as exc:
            message = f"video search failed for {query.text!r}: {exc}"
            log.warning("segment %03d: %s", segment.index, message)
            notes.append(message)
            continue
        except Exception as exc:  # noqa: BLE001 - never kill a segment
            message = f"video search raised for {query.text!r}: {exc}"
            log.warning("segment %03d: %s", segment.index, message)
            notes.append(message)
            continue

        for hit in hits:
            if not hit.video_id or hit.video_id in seen:
                continue
            seen.add(hit.video_id)

            classification, reason, relevance = classify_hit(hit, segment)
            candidates = locate_timestamps(hit, segment)
            best = candidates[0] if candidates else None

            records.append(
                YouTubeRecord(
                    title=hit.title,
                    # §12.7: the clickable link is stored regardless of whether
                    # a clip is ever downloaded.
                    url=best.url_with_t if best and best.url_with_t else hit.url,
                    video_id=hit.video_id,
                    channel=hit.channel,
                    duration=hit.duration,
                    thumbnail=hit.thumbnail,
                    query=query.text,
                    reason=reason,
                    relevance=round(relevance, 3),
                    classification=classification,
                    timestamp_candidates=candidates,
                )
            )

    # Best first: classification rank, then relevance, then timestamp confidence.
    order = {
        Classification.EXACT_SCENE: 0,
        Classification.LIKELY_EXACT_SCENE: 1,
        Classification.RELATED_FOOTAGE: 2,
        Classification.CONTEXTUAL: 3,
    }
    records.sort(
        key=lambda r: (
            order[r.classification],
            -r.relevance,
            -(r.best_timestamp().confidence if r.best_timestamp() else 0.0),
            r.video_id,
        )
    )
    log.info(
        "segment %03d: %d video result(s) (%s)",
        segment.index,
        len(records),
        ", ".join(
            f"{k}={sum(1 for r in records if r.classification == k)}"
            for k in Classification
            if any(r.classification == k for r in records)
        )
        or "none",
    )
    return records, notes
