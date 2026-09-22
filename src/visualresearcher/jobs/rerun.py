"""Re-running one segment (CLAUDE.md §17, §20).

§20 is explicit: "Re-running one segment must not re-run the project." That is
the whole design constraint here. A reviewer who edits a query for segment 31
wants segment 31 searched again, not four hundred segments re-downloaded.

So this reaches into the same stage functions the pipeline uses, but hands
them a one-segment list. Nothing else on disk is touched, except ``selected/``
which is rebuilt at the end so the deliverable still matches the picks (§15).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import ProjectContext, Segment
from ..utils.files import atomic_write_text, pad_width

__all__ = ["rerun_segment", "RerunResult", "RERUNNABLE_STAGES"]

log = get_logger("jobs.rerun")

#: What ``--stage`` accepts. Each entry re-runs that stage and everything
#: downstream of it that depends on its output, for the one segment.
RERUNNABLE_STAGES = ("image_search", "video_search", "clips", "rank")


@dataclass
class RerunResult:
    segment_index: int
    stages: list[str] = field(default_factory=list)
    candidates: int = 0
    downloaded: int = 0
    kept: int = 0
    selected: int = 0
    videos: int = 0
    clips: int = 0
    notes: list[str] = field(default_factory=list)


def _load(paths: ProjectPaths) -> tuple[list[Segment], ProjectContext, int]:
    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in paths.existing_segment_dirs()
        if (d / "segment.json").exists()
    ]
    context = (
        ProjectContext.model_validate_json(paths.project_context.read_text(encoding="utf-8"))
        if paths.project_context.exists()
        else ProjectContext()
    )
    return segments, context, pad_width(max((s.index for s in segments), default=1))


def rerun_segment(
    project: str,
    index: int,
    settings: Settings,
    *,
    stage: str | None = None,
) -> RerunResult:
    """Re-run one segment's research. Raises ``LookupError`` if it is not there."""
    from ..pipeline.clips import ClipCache, download_clip_for_segment
    from ..pipeline.collect import collect
    from ..pipeline.dedupe import dedupe_records
    from ..pipeline.download import download_candidates
    from ..pipeline.image_search import (
        collect_providers,
        search_segment,
        write_candidate_manifest,
    )
    from ..pipeline.rank import SegmentHistory, rank_segment, write_contact_sheet
    from ..pipeline.video_search import search_videos
    from ..providers.base import ProviderUnavailable
    from ..providers.registry import resolve as resolve_provider
    from ..schemas import Pick
    from ..utils.files import slugify

    paths = ProjectPaths(settings.project_dir(project))
    if not paths.root.exists():
        raise LookupError(f"no project at {paths.root}")

    segments, context, width = _load(paths)
    segment = next((s for s in segments if s.index == index), None)
    if segment is None:
        raise LookupError(f"project {project!r} has no segment {index}")

    wanted = stage or "image_search"
    if wanted not in RERUNNABLE_STAGES:
        raise ValueError(f"--stage must be one of {', '.join(RERUNNABLE_STAGES)}")

    result = RerunResult(segment_index=index)
    images = None

    # -- images -----------------------------------------------------------
    if wanted == "image_search":
        providers = collect_providers(settings)
        candidates, notes = search_segment(segment, providers, settings)
        result.candidates = len(candidates)
        result.notes.extend(notes)
        result.stages.append("image_search")

        target = paths.segment_images_dir(segment, width=width)
        records, rejected = download_candidates(
            candidates,
            target,
            settings,
            segment_index=index,
            limit=settings.images.candidates_per_segment,
        )
        result.downloaded = len(records)
        result.stages.append("download")

        deduped = dedupe_records(records, distance=settings.images.phash_distance)
        images = deduped.kept
        result.kept = len(images)
        result.stages.append("dedupe")

        write_candidate_manifest(
            paths.segment_image_manifest(segment, width=width),
            images,
            rejected=rejected,
            duplicates=deduped.duplicates,
            root=paths.root,
        )

    # -- ranking ----------------------------------------------------------
    if wanted in ("image_search", "rank"):
        from ..pipeline.image_search import read_candidate_manifest

        if images is None:
            images = read_candidate_manifest(
                paths.segment_image_manifest(segment, width=width), root=paths.root
            )["kept"]

        try:
            embedder = resolve_provider(
                "embedding",
                "openclip",
                model=settings.ranking.model,
                pretrained=settings.ranking.pretrained,
                device=settings.ranking.device,
                seed=settings.ranking.seed,
            )
        except ProviderUnavailable:
            embedder = None

        # A fresh history: §11's repetition penalty compares against recent
        # segments, and re-running one in isolation has no "recent".
        chosen = rank_segment(
            images, segment, context, settings, embedder=embedder, history=SegmentHistory()
        )
        result.selected = len(chosen)
        result.stages.append("rank")

        clip_picks = [p for p in segment.picks if p.kind == "clip"]
        segment.picks = [
            Pick(
                kind="image",
                path=str(
                    Path(r.local_path).relative_to(paths.root).as_posix()
                    if Path(r.local_path).is_absolute()
                    else r.local_path
                ),
                rank=r.rank,
                score=r.score,
                confidence=round(min(1.0, r.score), 3),
                use=r.rank == 1,
                slug=slugify(
                    segment.topic or segment.narration, max_len=settings.output.slug_max_len
                ),
            )
            for r in chosen
        ] + clip_picks  # a re-search of images must not drop the clip
        write_contact_sheet(chosen, paths.segment_contact_sheet(segment, width=width))

    # -- video ------------------------------------------------------------
    if wanted in ("video_search", "clips"):
        try:
            provider = resolve_provider("video", "ytdlp", max_height=settings.clips.max_height)
        except ProviderUnavailable as exc:
            raise RuntimeError(f"no video provider available: {exc}") from exc

        records = []
        if wanted == "video_search":
            records, notes = search_videos(segment, provider, settings)
            result.videos = len(records)
            result.notes.extend(notes)
            result.stages.append("video_search")
            atomic_write_text(
                paths.segment_youtube_results(segment, width=width),
                json.dumps([r.model_dump() for r in records], indent=2, default=str),
            )
        else:
            from ..schemas import YouTubeRecord

            path = paths.segment_youtube_results(segment, width=width)
            if path.exists():
                records = [
                    YouTubeRecord.model_validate(item)
                    for item in json.loads(path.read_text(encoding="utf-8"))
                ]

        if wanted == "clips" or settings.clips.enabled:
            from ..pipeline.clips import clip_confidence

            outcomes = download_clip_for_segment(
                segment,
                records,
                provider,
                settings,
                clips_dir=paths.segment_clips_dir(segment, width=width),
                cache=ClipCache(settings.cache_dir / "clips"),
                tmp_dir=settings.tmp_dir / "clips",
                project_root=paths.root,
            )
            result.stages.append("clips")
            segment.picks = [p for p in segment.picks if p.kind != "clip"]
            for outcome in outcomes:
                if outcome.ok and outcome.record:
                    result.clips += 1
                    segment.picks.append(
                        Pick(
                            kind="clip",
                            path=Path(outcome.record.downloaded_clip_path).as_posix(),
                            rank=1,
                            use=True,
                            score=outcome.record.relevance,
                            confidence=clip_confidence(outcome.record),
                            slug=slugify(
                                outcome.record.title or segment.topic,
                                max_len=settings.output.slug_max_len,
                            ),
                        )
                    )
                else:
                    result.notes.append(f"{outcome.reason}: {outcome.detail}")
            atomic_write_text(
                paths.segment_youtube_results(segment, width=width),
                json.dumps([r.model_dump() for r in records], indent=2, default=str),
            )

    # -- persist, and keep selected/ honest --------------------------------
    segment.notes.append(f"re-ran {', '.join(result.stages)} in review")
    atomic_write_text(
        paths.segment_dir(segment, width=width) / "segment.json",
        segment.model_dump_json(indent=2),
    )

    # §15: the deliverable must still match the picks. Only this segment's
    # files can have changed, but collect works on the whole project, which is
    # what keeps the add/remove diff correct.
    refreshed = [s if s.index != index else segment for s in segments]
    collect(refreshed, paths, settings, width=width)

    log.info(
        "segment %03d: re-ran %s (%d candidate(s), %d kept, %d selected, %d clip(s))",
        index,
        ", ".join(result.stages),
        result.candidates,
        result.kept,
        result.selected,
        result.clips,
    )
    return result
