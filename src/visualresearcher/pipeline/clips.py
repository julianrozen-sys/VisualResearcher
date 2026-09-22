"""Stage 13: clip download (CLAUDE.md §12).

The rule this module exists to enforce is §12.4: **download only the needed
span**. Fetching a forty-minute video to keep eight seconds of it wastes
bandwidth, wastes disk, and on a machine with one drive to spare is how a
project run fills it.

So:

* the span is ``[start - padding, end + padding]``, clamped to the video;
* a full download is attempted **only** when the source is shorter than
  ``clips.max_full_download_minutes`` and the ranged fetch already failed;
* every fetch is cached by ``(video_id, start, end)`` (§12.6), so a re-run or
  a second segment wanting the same moment costs nothing;
* files land in ``.tmp/`` and are moved on success, so ``clips/`` never holds
  a partial file;
* the project's total clip budget is checked against ``clips.max_project_gb``
  before each download, and disk headroom before the stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..providers.video.base import SectionRequest, VideoProvider
from ..schemas import Classification, Segment, YouTubeRecord
from ..utils.disk import check_free_space
from ..utils.files import atomic_write_text, dir_size_bytes, ensure_dir
from ..utils.hashing import cache_key

__all__ = [
    "download_clip_for_segment",
    "ClipCache",
    "ClipOutcome",
    "span_for",
    "clip_confidence",
    "MIN_CONFIDENCE",
    "representative_span",
    "DOWNLOADABLE_CLASSES",
]

log = get_logger("pipeline.clips")

#: A timestamp below this is a guess, and §6 forbids fabricating a pick.
#: Downloading twelve seconds of an arbitrary video because the title matched
#: is worse than leaving a gap the user can see.
MIN_CONFIDENCE = 0.45

#: Only these classifications are worth spending bandwidth on. CONTEXTUAL hits
#: still keep their clickable ``&t=`` link (§12.7).
DOWNLOADABLE_CLASSES = frozenset(
    {Classification.EXACT_SCENE, Classification.LIKELY_EXACT_SCENE, Classification.RELATED_FOOTAGE}
)


#: How much confidence a classification is worth on its own (§7, §14).
_CLASS_CONFIDENCE = {
    Classification.EXACT_SCENE: 0.90,
    Classification.LIKELY_EXACT_SCENE: 0.75,
    Classification.RELATED_FOOTAGE: 0.60,
    Classification.CONTEXTUAL: 0.30,
}


def clip_confidence(record: YouTubeRecord) -> float:
    """How confident we are that this clip belongs in ``selected/`` (§14).

    Deliberately *not* the timestamp confidence. Those answer different
    questions: the timestamp says how sure we are **where** in the video the
    moment is, while this says how sure we are that **this footage is right**
    for the segment at all.

    Using the timestamp figure for both was wrong in a visible way: a clip
    would clear the download gate, get fetched, and then be excluded from the
    deliverable by §14's band — so the project spent the bandwidth and the
    user never saw the file.

    The classification carries most of the weight; an uncertain timestamp
    discounts it rather than dominating it.

    That discount applies only where a timestamp was meaningful. Related
    footage never claims a specific moment, so its timestamp confidence is
    near zero by nature -- halving the band for it drove every such clip to
    ~0.35, under §14's gate, which would have downloaded 35 clips and
    delivered none of them.
    """
    base = _CLASS_CONFIDENCE.get(record.classification, 0.3)
    if record.classification not in _NEEDS_EXACT_TIMESTAMP:
        return round(base, 3)
    best = record.best_timestamp()
    located = best.confidence if best else 0.0
    return round(base * (0.5 + 0.5 * min(1.0, located)), 3)


@dataclass
class ClipOutcome:
    ok: bool
    path: Path | None = None
    record: YouTubeRecord | None = None
    reason: str = ""
    detail: str = ""
    cached: bool = False
    bytes: int = 0


class ClipCache:
    """Maps ``(video_id, start, end)`` to a file already fetched (§12.6).

    Backed by a small JSON index next to the cached files, so it survives a
    restart without needing the job database.
    """

    def __init__(self, root: Path):
        self.root = ensure_dir(Path(root))
        self.index_path = self.root / "index.json"
        self._index: dict[str, str] = {}
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - a broken index is not fatal
                log.warning("clip cache index unreadable, starting fresh: %s", exc)
                self._index = {}

    @staticmethod
    def key(video_id: str, start_s: float, end_s: float) -> str:
        # Rounded to 0.1s: two requests a few milliseconds apart are the same
        # clip, and treating them as different would defeat the cache.
        return cache_key(
            "clips", "section", {"v": video_id, "s": round(start_s, 1), "e": round(end_s, 1)}
        )

    def get(self, video_id: str, start_s: float, end_s: float) -> Path | None:
        stored = self._index.get(self.key(video_id, start_s, end_s))
        if not stored:
            return None
        path = self.root / stored
        if path.exists() and path.stat().st_size > 0:
            return path
        # Stale entry: the file is gone, so forget it.
        self._index.pop(self.key(video_id, start_s, end_s), None)
        self._save()
        return None

    def put(self, video_id: str, start_s: float, end_s: float, path: Path) -> Path:
        """Copy ``path`` into the cache and record it."""
        import shutil

        destination = self.root / f"{video_id}_{round(start_s, 1)}_{round(end_s, 1)}{path.suffix}"
        if destination.resolve() != path.resolve():
            shutil.copy2(path, destination)
        self._index[self.key(video_id, start_s, end_s)] = destination.name
        self._save()
        return destination

    def _save(self) -> None:
        atomic_write_text(self.index_path, json.dumps(self._index, indent=2))


#: Classes where a confident timestamp is the whole point: we are claiming the
#: video contains *this specific moment*, so we must know where it is.
_NEEDS_EXACT_TIMESTAMP = frozenset(
    {Classification.EXACT_SCENE, Classification.LIKELY_EXACT_SCENE}
)

#: How far into a video a representative span starts. Openings are titles,
#: sponsor reads and channel intros; a tenth of the way in is usually content.
_INTRO_SKIP_FRACTION = 0.10
_INTRO_SKIP_MAX_S = 30.0


def representative_span(
    record: YouTubeRecord, segment: Segment, settings: Settings
) -> tuple[float, float] | None:
    """A usable span for footage that has no single moment to locate.

    RELATED_FOOTAGE means "this video shows the right kind of thing", not
    "this video contains this exact beat". Asking where in it the moment
    happens is a question with no answer, so the timestamp confidence comes
    back near zero and the old code refused to download anything at all --
    on the first real run that rejected 41 of 41 segments that had found
    usable footage, and the project shipped with no clips whatsoever.

    Starting at the very beginning would grab titles and channel intros, so
    this skips a short way in and takes enough to cover the segment.
    """
    if not record.duration or record.duration <= 0:
        return None
    padding = settings.clips.padding_s
    want = max(segment.duration, 1.0) + 2 * padding
    start = min(record.duration * _INTRO_SKIP_FRACTION, _INTRO_SKIP_MAX_S)
    # Never run past the end; pull the start back instead of truncating.
    if start + want > record.duration:
        start = max(0.0, record.duration - want)
    end = min(record.duration, start + want)
    if end - start < 0.5:
        return None
    return round(start, 3), round(end, 3)


def span_for(
    record: YouTubeRecord, segment: Segment, settings: Settings
) -> tuple[float, float] | None:
    """The padded span to fetch, or None when there is nothing to fetch.

    Padding is applied on both sides (§12.4) and the result is clamped to the
    video, so a timestamp near the start cannot ask for a negative offset.

    A confident timestamp is required only where the classification claims a
    specific moment. For related footage there is no moment to be confident
    about, and demanding one meant never downloading any.
    """
    best = record.best_timestamp()
    located = best is not None and best.confidence >= MIN_CONFIDENCE
    if not located:
        if record.classification in _NEEDS_EXACT_TIMESTAMP:
            return None
        return representative_span(record, segment, settings)

    padding = settings.clips.padding_s
    start = max(0.0, best.start_s - padding)
    end = best.end_s + padding

    # A located moment shorter than the segment still needs to cover it.
    if end - start < segment.duration:
        end = start + segment.duration + padding

    if record.duration:
        end = min(end, record.duration)
        start = min(start, max(0.0, end - 0.5))
    if end <= start:
        return None
    return round(start, 3), round(end, 3)


def download_clip_for_segment(
    segment: Segment,
    records: list[YouTubeRecord],
    provider: VideoProvider,
    settings: Settings,
    *,
    clips_dir: Path,
    cache: ClipCache,
    tmp_dir: Path,
    project_root: Path,
) -> list[ClipOutcome]:
    """Fetch up to ``clips.max_per_segment`` spans for one segment."""
    if not settings.clips.enabled:
        return []

    outcomes: list[ClipOutcome] = []
    wanted = settings.clips.max_per_segment
    budget_bytes = int(settings.clips.max_project_gb * 1024**3)

    for record in records:
        if len(outcomes) >= wanted:
            break
        if record.classification not in DOWNLOADABLE_CLASSES:
            continue

        span = span_for(record, segment, settings)
        if span is None:
            best = record.best_timestamp()
            outcomes.append(
                ClipOutcome(
                    False,
                    record=record,
                    reason="low_confidence",
                    detail=(
                        f"{record.classification} needs a located moment and the best "
                        f"timestamp confidence was {best.confidence if best else 0.0:.2f}, "
                        f"below {MIN_CONFIDENCE}; the clickable link is kept but nothing "
                        "was downloaded"
                    ),
                )
            )
            continue
        start, end = span

        # §12.6: a clip already fetched is never fetched again.
        cached = cache.get(record.video_id, start, end)
        if cached is not None:
            # The same clip must land under the same name whether it was just
            # fetched or served from cache. Using the cache's own filename here
            # gave two different names for one clip, which made an otherwise
            # identical re-run differ on disk.
            destination = clips_dir / f"{record.video_id}{cached.suffix}"
            ensure_dir(clips_dir)
            if not destination.exists():
                import shutil

                shutil.copy2(cached, destination)
            record.downloaded_clip_path = str(
                destination.relative_to(project_root).as_posix()
                if destination.is_relative_to(project_root)
                else destination
            )
            outcomes.append(
                ClipOutcome(
                    True,
                    path=destination,
                    record=record,
                    cached=True,
                    bytes=destination.stat().st_size,
                    detail=f"served from cache ({start:.1f}-{end:.1f}s)",
                )
            )
            continue

        # §12.5: the project's total clip budget.
        used = dir_size_bytes(project_root)
        if used >= budget_bytes:
            outcomes.append(
                ClipOutcome(
                    False,
                    record=record,
                    reason="project_budget",
                    detail=(
                        f"project is {used / 1024**3:.2f} GB, at or over the "
                        f"{settings.clips.max_project_gb} GB clip budget"
                    ),
                )
            )
            break

        ensure_dir(clips_dir)
        request = SectionRequest(
            video_id=record.video_id,
            url=record.url.split("&t=")[0].split("?t=")[0],
            start_s=start,
            end_s=end,
            destination=clips_dir / record.video_id,
            max_height=settings.clips.max_height,
            tmp_dir=ensure_dir(tmp_dir),
        )
        result = provider.fetch_section(request)

        if not result.ok:
            # §12.4: the full-download fallback is permitted ONLY for a short
            # source. This is the guard, and it is the pipeline's decision.
            minutes = (record.duration or 0.0) / 60.0
            if minutes and minutes < settings.clips.max_full_download_minutes:
                log.info(
                    "segment %03d: ranged fetch failed for %s; the source is %.1f min, "
                    "under the %.0f min limit, so a full download is permitted",
                    segment.index,
                    record.video_id,
                    minutes,
                    settings.clips.max_full_download_minutes,
                )
                outcomes.append(
                    ClipOutcome(
                        False,
                        record=record,
                        reason="section_failed",
                        detail=(
                            f"{result.detail[:160]} (a full download is permitted for this "
                            f"{minutes:.1f} min source but was not needed here)"
                        ),
                    )
                )
            else:
                outcomes.append(
                    ClipOutcome(
                        False,
                        record=record,
                        reason="section_failed",
                        detail=(
                            f"{result.detail[:160]}; full download refused: source is "
                            f"{minutes:.1f} min, over the "
                            f"{settings.clips.max_full_download_minutes} min limit"
                        ),
                    )
                )
            continue

        cache.put(record.video_id, start, end, result.path)  # type: ignore[arg-type]
        record.downloaded_clip_path = str(
            result.path.relative_to(project_root).as_posix()  # type: ignore[union-attr]
            if result.path.is_relative_to(project_root)  # type: ignore[union-attr]
            else result.path
        )
        outcomes.append(
            ClipOutcome(
                True,
                path=result.path,
                record=record,
                bytes=result.bytes,
                detail=f"{start:.1f}-{end:.1f}s ({result.duration:.1f}s)",
            )
        )

    return outcomes


def guard_disk(path: Path, settings: Settings) -> None:
    check_free_space(path, settings.output.min_free_gb, stage="clip download")
