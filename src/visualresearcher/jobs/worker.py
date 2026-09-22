"""The pipeline runner (CLAUDE.md §8).

One function walks ``STAGE_ORDER``, skipping stages that already have a
successful checkpoint unless ``--force`` was given. Stages that are not built
yet announce themselves and are skipped rather than faked -- a stage never
writes a placeholder to look finished (§6).

Ordinary stage failures mark the job FAILED with the error preserved; a
*provider* failure inside a stage is the stage's own business to degrade
around (§8).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import EntityCorrection, ImageRecord, ProjectContext, Segment, Transcript
from ..utils.disk import DiskSpaceError
from ..utils.files import pad_width
from .queue import (
    completed_stages,
    heartbeat,
    record_checkpoint,
    set_state,
)
from .states import STAGE_ORDER, STAGE_STATE, JobState, Stage

__all__ = ["RunContext", "run_pipeline", "StageResult", "IMPLEMENTED_STAGES"]

log = get_logger("jobs.worker")


@dataclass
class RunContext:
    """Everything a stage needs, and everything stages hand to each other."""

    job_id: str
    project: str
    project_root: Path
    source_path: Path
    settings: Settings
    db_path: Path
    force: bool = False
    no_clips: bool = False

    paths: ProjectPaths = field(init=False)
    transcript: Transcript | None = None
    segments: list[Segment] = field(default_factory=list)
    context: ProjectContext | None = None
    corrections: list[EntityCorrection] = field(default_factory=list)
    packs: list = field(default_factory=list)
    #: segment index -> search hits, downloaded records, and rejections.
    candidates: dict[int, list] = field(default_factory=dict)
    images: dict[int, list[ImageRecord]] = field(default_factory=dict)
    videos: dict[int, list] = field(default_factory=dict)
    collect_result: object | None = None
    #: Shared across concurrently running jobs (§19). None means solo.
    limits: object | None = None
    rejections: dict[int, list] = field(default_factory=dict)
    width: int = 3
    notes: list[str] = field(default_factory=list)

    #: When set, `_load_segments` yields only these indices. This is how
    #: segment pipelining reuses the batch stages unchanged: a stage that
    #: loops over "every segment" is handed a list of one. Two separate
    #: implementations of a stage would be free to disagree; this cannot.
    only_segments: set[int] | None = None
    #: Expensive state that must outlive a single segment.
    #:
    #: `embedder` holds the CLIP weights -- rebuilding it per segment would
    #: reload the model 95 times. `history` is the repetition window, which is
    #: causal: it only looks back, so feeding it segment by segment in
    #: chronological order gives byte-identical scores to the batch run.
    embedder: object | None = None
    history: object | None = None
    video_provider: object | None = None
    clip_cache: object | None = None
    shapes_seen: set[str] = field(default_factory=set)
    #: Segments whose files are already in selected/ (§21 progress).
    delivered: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.paths = ProjectPaths(self.project_root)

    @property
    def audio_path(self) -> Path:
        return self.paths.narration_wav


@dataclass
class StageResult:
    stage: Stage
    ok: bool
    skipped: bool = False
    detail: str = ""
    artifact: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _stage_ingest(ctx: RunContext) -> StageResult:
    from ..pipeline.ingest import ingest

    result = ingest(
        ctx.source_path,
        ctx.project_root,
        ctx.settings,
        move=False,
        force=ctx.force,
    )
    return StageResult(
        Stage.INGEST,
        ok=True,
        detail=f"{result.duration_s:.1f}s of audio",
        artifact=str(result.audio),
    )


def _stage_transcribe(ctx: RunContext) -> StageResult:
    from ..pipeline.transcribe import transcribe

    # §19: transcription is CPU-heavy and does not parallelise without a
    # GPU, so it waits for the one global slot.
    with _cpu_slot(ctx, "transcribe"):
        ctx.transcript = transcribe(ctx.audio_path, ctx.paths, ctx.settings, force=ctx.force)
    return StageResult(
        Stage.TRANSCRIBE,
        ok=True,
        detail=(
            f"{len(ctx.transcript.segments)} transcript segments, "
            f"{len(ctx.transcript.all_words())} words, "
            f"provider={ctx.transcript.provider}"
        ),
        artifact=str(ctx.paths.transcript_json),
    )


def _load_transcript(ctx: RunContext) -> Transcript:
    if ctx.transcript is None:
        if not ctx.paths.transcript_json.exists():
            raise FileNotFoundError(
                f"no transcript at {ctx.paths.transcript_json}; run the transcribe stage first"
            )
        ctx.transcript = Transcript.model_validate_json(
            ctx.paths.transcript_json.read_text(encoding="utf-8")
        )
    return ctx.transcript


def _stage_segment(ctx: RunContext) -> StageResult:
    from ..pipeline.segment import materialize_segments, segment_transcript, write_timeline_csv

    transcript = _load_transcript(ctx)
    segments = segment_transcript(transcript, ctx.settings)
    ctx.width = pad_width(max((s.index for s in segments), default=1))
    materialize_segments(segments, ctx.paths, width=ctx.width)
    write_timeline_csv(segments, ctx.paths.timeline_csv, width=ctx.width)
    ctx.segments = segments

    durations = [s.duration for s in segments]
    return StageResult(
        Stage.SEGMENT,
        ok=True,
        detail=(
            f"{len(segments)} segments, {min(durations):.1f}-{max(durations):.1f}s"
            if durations
            else "0 segments"
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _cpu_slot(ctx: RunContext, label: str):
    """The shared CPU semaphore, or a no-op when running solo (§19)."""
    from contextlib import nullcontext

    limits = ctx.limits
    if limits is None:
        return nullcontext()
    return limits.compute.slot(f"{ctx.project}:{label}")


def _guard_disk(ctx: RunContext, stage: str) -> None:
    """Disk headroom, counting every other running job's reservation (§19)."""
    if ctx.limits is not None:
        ctx.limits.disk.check(ctx.project_root, job_id=ctx.job_id, stage=stage)
    else:
        from ..utils.disk import check_free_space

        check_free_space(ctx.project_root, ctx.settings.output.min_free_gb, stage=stage)


def _llm(ctx: RunContext):
    """The project's LLM provider, or None when none is usable.

    ``resolve`` already substitutes the fake under ``VR_OFFLINE=1`` and falls
    back to it when a real provider is unavailable, so this returns None only
    if nothing at all is registered.
    """
    from ..providers.base import ProviderUnavailable
    from ..providers.registry import resolve as resolve_provider

    try:
        return resolve_provider("llm", "openai")
    except ProviderUnavailable as exc:
        log.warning("no LLM provider available: %s", exc)
        return None


def _active_packs(ctx: RunContext) -> list:
    from ..pipeline.entities import load_domain_packs, select_packs

    if ctx.packs:
        return ctx.packs
    transcript = _load_transcript(ctx)
    ctx.packs = select_packs(transcript.text, load_domain_packs(ctx.settings.domain_packs_dir))
    return ctx.packs


def _write_context(ctx: RunContext, context: ProjectContext) -> None:
    from ..utils.files import atomic_write_text

    ctx.context = context
    atomic_write_text(ctx.paths.project_context, context.model_dump_json(indent=2))


def _stage_context(ctx: RunContext) -> StageResult:
    """Build ``project_context.json``.

    Written here without corrections; the entities stage adds them and rewrites
    the file. Every stage leaves its artifact complete on disk, which is what
    makes each one independently re-runnable (§8).
    """
    from ..pipeline.context import build_context

    transcript = _load_transcript(ctx)
    context = build_context(transcript, _active_packs(ctx), llm=_llm(ctx))
    _write_context(ctx, context)

    return StageResult(
        Stage.CONTEXT,
        ok=True,
        detail=(
            f"subject={context.subject!r}, "
            f"{len(context.characters)} character(s), "
            f"{len(ctx.packs)} pack(s) active"
        ),
        artifact=str(ctx.paths.project_context),
    )


def _stage_entities(ctx: RunContext) -> StageResult:
    """Resolve mishearings and record them in ``project_context.json`` (§10)."""
    from ..pipeline.entities import (
        EntityResolver,
        load_overrides,
        write_overrides_template,
    )

    transcript = _load_transcript(ctx)
    if write_overrides_template(ctx.paths.entity_overrides):
        log.info("wrote %s for you to edit", ctx.paths.entity_overrides.name)
    overrides = load_overrides(ctx.paths.entity_overrides)

    resolver = EntityResolver(_active_packs(ctx), overrides=overrides, llm=_llm(ctx))
    ctx.corrections = resolver.resolve(transcript.text)

    context = _load_context(ctx)
    context.entity_corrections = list(ctx.corrections)
    for correction in ctx.corrections:
        if correction.applied and correction.resolved not in context.characters:
            context.characters.append(correction.resolved)
    _write_context(ctx, context)

    applied = sum(1 for c in ctx.corrections if c.applied)
    return StageResult(
        Stage.ENTITIES,
        ok=True,
        detail=(
            f"{len(ctx.corrections)} correction(s), {applied} applied at "
            f">=0.75, {len(ctx.packs)} pack(s) active"
        ),
        artifact=str(ctx.paths.project_context),
    )


def _stage_queries(ctx: RunContext) -> StageResult:
    """Per-segment analysis and query generation, then rewrite every segment.json."""
    from ..pipeline.context import analyze_segments
    from ..pipeline.queries import generate_queries
    from ..pipeline.segment import materialize_segments

    segments = _load_segments(ctx)
    context = _load_context(ctx)
    corrections = _load_corrections(ctx)

    analyze_segments(segments, context, llm=_llm(ctx), corrections=corrections)
    generate_queries(segments, context, corrections, ctx.settings)
    materialize_segments(segments, ctx.paths, width=ctx.width)
    ctx.segments = segments

    counts = [len(s.queries) for s in segments] or [0]
    kinds = {str(q.kind) for s in segments for q in s.queries}
    return StageResult(
        Stage.QUERIES,
        ok=True,
        detail=(
            f"{sum(counts)} queries across {len(segments)} segment(s), "
            f"min {min(counts)}/segment, {len(kinds)} distinct kind(s)"
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _load_corrections(ctx: RunContext) -> list[EntityCorrection]:
    """Corrections from memory, or read back out of ``project_context.json``.

    Resuming at ``queries`` must not silently lose the corrections computed by
    an earlier process -- that would generate queries against the misheard
    spellings and quietly return the wrong images.
    """
    if ctx.corrections:
        return ctx.corrections
    ctx.corrections = list(_load_context(ctx).entity_corrections)
    if ctx.corrections:
        log.info("recovered %d correction(s) from project_context.json", len(ctx.corrections))
    return ctx.corrections


def _load_segments(ctx: RunContext) -> list[Segment]:
    """Segments for this stage, narrowed to `ctx.only_segments` when set."""
    segments = _all_segments(ctx)
    if ctx.only_segments is None:
        return segments
    return [s for s in segments if s.index in ctx.only_segments]


def _all_segments(ctx: RunContext) -> list[Segment]:
    """Every segment, from memory or read back off disk when resuming."""
    if ctx.segments:
        return ctx.segments
    dirs = ctx.paths.existing_segment_dirs()
    if not dirs:
        raise FileNotFoundError(
            f"no segments under {ctx.paths.segments_dir}; run the segment stage first"
        )
    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in dirs
        if (d / "segment.json").exists()
    ]
    ctx.width = pad_width(max((s.index for s in segments), default=1))
    ctx.segments = segments
    return segments


def _load_context(ctx: RunContext) -> ProjectContext:
    if ctx.context is not None:
        return ctx.context
    if ctx.paths.project_context.exists():
        ctx.context = ProjectContext.model_validate_json(
            ctx.paths.project_context.read_text(encoding="utf-8")
        )
    else:
        ctx.context = ProjectContext()
    return ctx.context


# ---------------------------------------------------------------------------
# One segment's worth of each per-segment stage.
#
# Extracted so the batch stages and segment pipelining run *the same* code.
# Two copies of "what one segment needs" would drift, and the drift would show
# up as a delivered folder that differs depending on which mode produced it.
# ---------------------------------------------------------------------------


def _one_image_search(ctx: RunContext, segment, providers) -> bool:
    """Search one segment. True when it came back degraded."""
    from ..pipeline.image_search import search_segment

    candidates, notes = search_segment(segment, providers, ctx.settings)
    ctx.candidates[segment.index] = candidates
    if notes:
        segment.notes.extend(notes)
        segment.status = "degraded"
        return True
    return False


def _one_download(ctx: RunContext, segment) -> None:
    """Download and validate one segment's candidates."""
    from ..pipeline.download import download_candidates

    target = ctx.paths.segment_images_dir(segment, width=ctx.width)
    records, rejected = download_candidates(
        ctx.candidates.get(segment.index, []),
        target,
        ctx.settings,
        segment_index=segment.index,
        # §11: ~30 candidates per segment. Without this cap a provider
        # returning generously would blow the budget and the disk.
        limit=ctx.settings.images.candidates_per_segment,
    )
    ctx.images[segment.index] = records
    ctx.rejections[segment.index] = rejected


def _one_dedupe(ctx: RunContext, segment) -> tuple[int, int]:
    """Collapse one segment's duplicates. Returns (kept, duplicates)."""
    from ..pipeline.dedupe import dedupe_records
    from ..pipeline.image_search import write_candidate_manifest

    records = ctx.images.get(segment.index, [])
    result = dedupe_records(records, distance=ctx.settings.images.phash_distance)
    ctx.images[segment.index] = result.kept
    write_candidate_manifest(
        ctx.paths.segment_image_manifest(segment, width=ctx.width),
        result.kept,
        rejected=ctx.rejections.get(segment.index, []),
        duplicates=result.duplicates,
        root=ctx.paths.root,
    )
    return len(result.kept), len(result.duplicates)


def _stage_image_search(ctx: RunContext) -> StageResult:
    """Search every segment's queries and stash the candidates for download."""
    from ..pipeline.image_search import collect_providers

    segments = _load_segments(ctx)
    providers = collect_providers(ctx.settings)
    if not providers:
        raise RuntimeError(
            "no image providers are available; set VR_OFFLINE=1 to use the fake, "
            "or check `visualresearch doctor`"
        )

    ctx.candidates = {}
    degraded = sum(_one_image_search(ctx, s, providers) for s in segments)

    totals = [len(v) for v in ctx.candidates.values()] or [0]
    return StageResult(
        Stage.IMAGE_SEARCH,
        ok=True,
        detail=(
            f"{sum(totals)} candidate(s), {min(totals)}-{max(totals)} per segment"
            + (f", {degraded} segment(s) degraded" if degraded else "")
        ),
    )


def _stage_download(ctx: RunContext) -> StageResult:
    """Download and validate candidates (§11). Rejections are kept, not dropped."""
    from ..pipeline.image_search import collect_providers, search_segment

    _guard_disk(ctx, "image download")
    segments = _load_segments(ctx)
    if not ctx.candidates:
        # Resuming straight into download: redo the (cheap, cached) search.
        providers = collect_providers(ctx.settings)
        ctx.candidates = {s.index: search_segment(s, providers, ctx.settings)[0] for s in segments}

    ctx.images = {}
    ctx.rejections = {}
    for segment in segments:
        _one_download(ctx, segment)

    kept = sum(len(v) for v in ctx.images.values())
    rejected_total = sum(len(v) for v in ctx.rejections.values())
    per_segment = [len(v) for v in ctx.images.values()] or [0]
    return StageResult(
        Stage.DOWNLOAD,
        ok=True,
        detail=(
            f"{kept} image(s) downloaded, {rejected_total} rejected, "
            f"{min(per_segment)}-{max(per_segment)} per segment"
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _stage_dedupe(ctx: RunContext) -> StageResult:
    """Collapse duplicates and write each segment's manifest (§11)."""

    segments = _load_segments(ctx)
    total_kept = 0
    total_dupes = 0

    for segment in segments:
        kept, dupes = _one_dedupe(ctx, segment)
        total_kept += kept
        total_dupes += dupes

    per_segment = [len(v) for v in ctx.images.values()] or [0]
    thin = [i for i, v in ctx.images.items() if len(v) < 15]
    if thin:
        log.warning(
            "%d segment(s) have fewer than 15 deduplicated candidates: %s",
            len(thin),
            sorted(thin)[:10],
        )
    return StageResult(
        Stage.DEDUPE,
        ok=True,
        detail=(
            f"{total_kept} kept, {total_dupes} duplicate(s) recorded, "
            f"{min(per_segment)}-{max(per_segment)} per segment"
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _stage_rank(ctx: RunContext) -> StageResult:
    """Score, select a diverse top-N, write the contact sheets (§11)."""
    from ..pipeline.rank import SegmentHistory
    from ..pipeline.segment import materialize_segments
    from ..providers.base import ProviderUnavailable
    from ..providers.registry import resolve as resolve_provider

    segments = _load_segments(ctx)
    context = _load_context(ctx)

    if ctx.embedder is None:
        try:
            ctx.embedder = resolve_provider(
                "embedding",
                "openclip",
                model=ctx.settings.ranking.model,
                pretrained=ctx.settings.ranking.pretrained,
                device=ctx.settings.ranking.device,
                seed=ctx.settings.ranking.seed,
            )
        except ProviderUnavailable as exc:
            log.warning("no embedding provider: %s", exc)
            ctx.embedder = _NO_EMBEDDER
    embedder = None if ctx.embedder is _NO_EMBEDDER else ctx.embedder

    if ctx.history is None:
        ctx.history = SegmentHistory()
    history = ctx.history
    total_selected = 0
    shapes_seen = ctx.shapes_seen

    # §19: CLIP ranking is the other CPU-heavy stage. try/finally rather than
    # a bare acquire, so an exception mid-ranking cannot leak the one global
    # slot and wedge every other job behind it.
    cpu = _cpu_slot(ctx, "rank")
    cpu.__enter__()
    try:
        _rank_all(ctx, segments, context, embedder, history, shapes_seen)
    finally:
        cpu.__exit__(None, None, None)

    materialize_segments(segments, ctx.paths, width=ctx.width)
    total_selected = sum(len(s.picks) for s in segments)

    per_segment = [len(s.picks) for s in segments] or [0]
    return StageResult(
        Stage.RANK,
        ok=True,
        detail=(
            f"{total_selected} selected, {min(per_segment)}-{max(per_segment)} per segment, "
            f"{len(shapes_seen)} frame shape(s), "
            f"embedder={embedder.name if embedder else 'none'}"
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _rank_all(ctx, segments, context, embedder, history, shapes_seen) -> None:
    """Rank every segment. Split out so the CPU slot can be held in a try block."""

    for segment in segments:
        _one_rank(ctx, segment, context, embedder, history, shapes_seen)


def _one_rank(ctx, segment, context, embedder, history, shapes_seen) -> None:
    """Rank one segment. Shared by the batch stage and segment pipelining."""
    from ..pipeline.image_search import read_candidate_manifest, write_candidate_manifest
    from ..pipeline.rank import _shape_bucket, rank_segment, write_contact_sheet
    from ..schemas import Pick
    from ..utils.files import slugify

    records = ctx.images.get(segment.index)
    if records is None:
        # Resuming into rank: read the manifest back rather than re-downloading.
        manifest = read_candidate_manifest(
            ctx.paths.segment_image_manifest(segment, width=ctx.width),
            root=ctx.paths.root,
        )
        records = manifest["kept"]
        ctx.images[segment.index] = records

    chosen = rank_segment(
        records, segment, context, ctx.settings, embedder=embedder, history=history
    )

    segment.picks = [
        Pick(
            kind="image",
            path=str(
                Path(r.local_path).relative_to(ctx.paths.root).as_posix()
                if Path(r.local_path).is_absolute()
                else r.local_path
            ),
            rank=r.rank,
            score=r.score,
            confidence=round(min(1.0, r.score), 3),
            # The top-ranked image is the default pick for the
            # deliverable; the rest stay available for the review UI to
            # promote with USE THIS (§15).
            use=r.rank == 1,
            slug=slugify(
                segment.topic or segment.narration, max_len=ctx.settings.output.slug_max_len
            ),
        )
        for r in chosen
    ]

    shapes_seen.update(_shape_bucket(r) for r in chosen)

    write_contact_sheet(chosen, ctx.paths.segment_contact_sheet(segment, width=ctx.width))
    write_candidate_manifest(
        ctx.paths.segment_image_manifest(segment, width=ctx.width),
        records,
        rejected=ctx.rejections.get(segment.index, []),
        duplicates=[],
        root=ctx.paths.root,
    )


#: Sentinel: "we tried to build one and could not", so a failed lookup is
#: cached too. Without it, an unavailable provider is retried 95 times.
_NO_EMBEDDER = object()


def _video_provider(ctx: RunContext):
    from ..providers.base import ProviderUnavailable
    from ..providers.registry import resolve as resolve_provider

    if ctx.video_provider is not None:
        return None if ctx.video_provider is _NO_EMBEDDER else ctx.video_provider
    try:
        ctx.video_provider = resolve_provider(
            "video", "ytdlp", max_height=ctx.settings.clips.max_height
        )
    except ProviderUnavailable as exc:
        log.warning("no video provider available: %s", exc)
        ctx.video_provider = _NO_EMBEDDER
        return None
    return ctx.video_provider


def _stage_video_search(ctx: RunContext) -> StageResult:
    """Search YouTube and classify the hits, with reasons (§12.1-12.2).

    Timestamp location runs inside this stage rather than after it, because
    the caption and chapter data it needs arrives with the search result and
    holding it until the next stage would mean fetching it twice.
    """
    from ..pipeline.video_search import search_videos
    from ..utils.files import atomic_write_text

    segments = _load_segments(ctx)
    provider = _video_provider(ctx)
    if provider is None:
        raise RuntimeError("no video provider is available; install yt-dlp or set VR_OFFLINE=1")

    ctx.videos = {}
    total = 0
    exact = 0
    for segment in segments:
        records, notes = search_videos(segment, provider, ctx.settings)
        ctx.videos[segment.index] = records
        total += len(records)
        exact += sum(1 for r in records if str(r.classification).startswith("EXACT"))
        if notes:
            segment.notes.extend(notes)
            segment.status = "degraded"

        results_path = ctx.paths.segment_youtube_results(segment, width=ctx.width)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            results_path,
            json.dumps([r.model_dump() for r in records], indent=2, default=str),
        )

    return StageResult(
        Stage.VIDEO_SEARCH,
        ok=True,
        detail=f"{total} result(s) across {len(segments)} segment(s), {exact} EXACT_SCENE",
        artifact=str(ctx.paths.segments_dir),
    )


def _stage_timestamps(ctx: RunContext) -> StageResult:
    """Report on the timestamps located during the search stage (§12.3)."""
    segments = _load_segments(ctx)
    _load_videos(ctx, segments)

    by_method: dict[str, int] = {}
    with_timestamp = 0
    for records in ctx.videos.values():
        for record in records:
            best = record.best_timestamp()
            if best:
                with_timestamp += 1
                by_method[best.method] = by_method.get(best.method, 0) + 1

    return StageResult(
        Stage.TIMESTAMPS,
        ok=True,
        detail=(
            f"{with_timestamp} result(s) with a located timestamp"
            + (
                " (" + ", ".join(f"{k}={v}" for k, v in sorted(by_method.items())) + ")"
                if by_method
                else ""
            )
        ),
    )


def _stage_clips(ctx: RunContext) -> StageResult:
    """Download only the needed span of each chosen video (§12.4)."""
    from ..pipeline.clips import (
        ClipCache,
        clip_confidence,
        download_clip_for_segment,
        guard_disk,
    )
    from ..pipeline.segment import materialize_segments
    from ..schemas import Pick
    from ..utils.files import atomic_write_text, slugify

    segments = _load_segments(ctx)

    if ctx.no_clips or not ctx.settings.clips.enabled:
        reason = "--no-clips" if ctx.no_clips else "clips.enabled is false"
        log.info("clip download skipped: %s", reason)
        return StageResult(Stage.CLIPS, ok=True, skipped=True, detail=f"skipped ({reason})")

    _load_videos(ctx, segments)
    guard_disk(ctx.project_root, ctx.settings)

    provider = _video_provider(ctx)
    if provider is None:
        raise RuntimeError("no video provider is available for clip download")

    if ctx.clip_cache is None:
        ctx.clip_cache = ClipCache(ctx.settings.cache_dir / "clips")
    cache = ctx.clip_cache
    downloaded = 0
    cached = 0
    total_bytes = 0
    gaps: list[str] = []

    for segment in segments:
        outcomes = download_clip_for_segment(
            segment,
            ctx.videos.get(segment.index, []),
            provider,
            ctx.settings,
            clips_dir=ctx.paths.segment_clips_dir(segment, width=ctx.width),
            cache=cache,
            tmp_dir=ctx.settings.tmp_dir / "clips",
            project_root=ctx.paths.root,
        )
        if not outcomes:
            # No candidate was even eligible. An empty clips/ folder with no
            # explanation is the thing a reviewer cannot act on, so say why
            # here; §6 turns these into the gap list in shotlist.md.
            records = ctx.videos.get(segment.index, [])
            if not records:
                why = "no video results at all for this segment"
            else:
                why = (
                    f"all {len(records)} video result(s) were classified CONTEXTUAL, "
                    "so none was worth downloading"
                )
            segment.notes.append(f"no clip: {why}")
            gaps.append(f"{segment.index:03d}")

        for outcome in outcomes:
            if outcome.ok and outcome.path:
                downloaded += 1
                cached += 1 if outcome.cached else 0
                total_bytes += outcome.bytes
                segment.picks.append(
                    Pick(
                        kind="clip",
                        path=Path(outcome.record.downloaded_clip_path).as_posix()
                        if outcome.record
                        else str(outcome.path),
                        rank=1,
                        use=True,
                        score=outcome.record.relevance if outcome.record else 0.0,
                        # §14's band, not the timestamp confidence -- see
                        # clip_confidence() for why they are different.
                        confidence=(clip_confidence(outcome.record) if outcome.record else 0.0),
                        slug=slugify(
                            (outcome.record.title if outcome.record else segment.topic)
                            or segment.topic,
                            max_len=ctx.settings.output.slug_max_len,
                        ),
                    )
                )
            else:
                note = f"clip not downloaded ({outcome.reason}): {outcome.detail}"
                segment.notes.append(note)
                gaps.append(f"{segment.index:03d}")

        # Persist the updated records: downloaded_clip_path lives on them.
        results_path = ctx.paths.segment_youtube_results(segment, width=ctx.width)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            results_path,
            json.dumps(
                [r.model_dump() for r in ctx.videos.get(segment.index, [])],
                indent=2,
                default=str,
            ),
        )

    materialize_segments(segments, ctx.paths, width=ctx.width)
    return StageResult(
        Stage.CLIPS,
        ok=True,
        detail=(
            f"{downloaded} clip(s) ({cached} from cache), "
            f"{total_bytes / 1024**2:.1f} MB"
            + (f", {len(set(gaps))} segment(s) without one" if gaps else "")
        ),
        artifact=str(ctx.paths.segments_dir),
    )


def _load_videos(ctx: RunContext, segments: list[Segment]) -> None:
    """Read ``youtube/results.json`` back when resuming into a later stage."""
    if ctx.videos:
        return
    from ..schemas import YouTubeRecord

    ctx.videos = {}
    for segment in segments:
        path = ctx.paths.segment_youtube_results(segment, width=ctx.width)
        if not path.exists():
            ctx.videos[segment.index] = []
            continue
        ctx.videos[segment.index] = [
            YouTubeRecord.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]


def _stage_collect(ctx: RunContext) -> StageResult:
    """Build selected/ -- the deliverable -- and the flat browse folder (§6, §14).

    `all_candidates/` is written here rather than behind a command, because a
    folder you have to know to ask for is a folder nobody has. Every run gets
    one: it is how you swap a pick without opening 170 segment folders.
    """
    from ..pipeline.all_candidates import dump_all_candidates
    from ..pipeline.collect import collect

    segments = _load_segments(ctx)
    result = collect(segments, ctx.paths, ctx.settings, width=ctx.width)
    ctx.collect_result = result
    dump_all_candidates(_all_segments(ctx), ctx.paths, ctx.settings, width=ctx.width)

    return StageResult(
        Stage.COLLECT,
        ok=True,
        detail=(
            f"{result.total} file(s) in selected/ "
            f"({len(result.written)} new, {len(result.removed)} removed), "
            f"{len(result.gaps)} gap(s), {len(result.flagged)} flagged"
        ),
        artifact=str(ctx.paths.selected_dir),
    )


def _stage_report(ctx: RunContext) -> StageResult:
    """sources.csv, shotlist.md and research_report.md (§6, §13)."""
    from ..pipeline.collect import CollectResult, collect
    from ..pipeline.report import write_reports

    segments = _load_segments(ctx)
    context = _load_context(ctx)

    result = ctx.collect_result
    if not isinstance(result, CollectResult):
        # Resuming straight into report: recompute the plan from disk rather
        # than reporting on a state we did not observe.
        result = collect(segments, ctx.paths, ctx.settings, width=ctx.width)

    from .queue import stage_timings

    rows = write_reports(
        result,
        segments,
        context,
        ctx.paths,
        ctx.settings,
        width=ctx.width,
        timings=stage_timings(ctx.db_path, ctx.job_id),
    )
    return StageResult(
        Stage.REPORT,
        ok=True,
        detail=(
            f"sources.csv {rows} row(s), shotlist.md with {len(result.gaps)} gap(s), "
            "research_report.md"
        ),
        artifact=str(ctx.paths.sources_csv),
    )


#: Stages with a real implementation. Anything not listed is skipped with a
#: clear message naming the phase that will build it.
IMPLEMENTED_STAGES: dict[Stage, Callable[[RunContext], StageResult]] = {
    Stage.INGEST: _stage_ingest,
    Stage.TRANSCRIBE: _stage_transcribe,
    Stage.CONTEXT: _stage_context,
    Stage.ENTITIES: _stage_entities,
    Stage.SEGMENT: _stage_segment,
    Stage.QUERIES: _stage_queries,
    Stage.IMAGE_SEARCH: _stage_image_search,
    Stage.DOWNLOAD: _stage_download,
    Stage.DEDUPE: _stage_dedupe,
    Stage.RANK: _stage_rank,
    Stage.VIDEO_SEARCH: _stage_video_search,
    Stage.TIMESTAMPS: _stage_timestamps,
    Stage.CLIPS: _stage_clips,
    Stage.COLLECT: _stage_collect,
    Stage.REPORT: _stage_report,
}

#: Which phase (CLAUDE.md §22) will implement each remaining stage.
_PLANNED: dict[Stage, str] = {}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


#: How often the background ticker writes a heartbeat while a stage runs.
#:
#: `heartbeat()` used to be called only between stages, which made the beat
#: period equal to the *stage* duration. `image_search` over a 20-minute
#: narration runs longer than `STALE_AFTER`, so a perfectly healthy job looked
#: abandoned -- and the fix for that must not be "raise the threshold", because
#: then a genuinely dead job stays invisible for an hour. Beat on a clock
#: instead, and staleness means what it says.
HEARTBEAT_EVERY_S = 30.0


class _Heartbeat:
    """Writes a heartbeat on a timer for as long as the job is running."""

    def __init__(self, db_path, job_id: str):
        self.db_path = db_path
        self.job_id = job_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{self.job_id}", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY_S):
            try:
                heartbeat(self.db_path, self.job_id)
            except Exception as exc:  # noqa: BLE001
                # A heartbeat is bookkeeping. Never let it kill a running job.
                log.debug("heartbeat failed for %s: %s", self.job_id, exc)

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Segment pipelining (§21)
#
# The batch order runs every segment through a stage before any segment sees
# the next one, so `selected/` stays empty until the very end and a 40-minute
# run shows nothing until minute 39. Pipelining walks the same stages in the
# same order, but one segment at a time, delivering each as it finishes.
#
# It is deliberately *not* a second implementation: each stage function is
# called exactly as the batch runner calls it, with `ctx.only_segments`
# narrowing what it sees. Total API calls and total work are unchanged -- this
# reorders, it does not reduce.
# ---------------------------------------------------------------------------

#: Stages that operate on one segment at a time, in order.
PER_SEGMENT_STAGES: tuple[Stage, ...] = (
    Stage.IMAGE_SEARCH,
    Stage.DOWNLOAD,
    Stage.DEDUPE,
    Stage.RANK,
    Stage.VIDEO_SEARCH,
    Stage.CLIPS,
)

#: Everything before the per-segment run: they need the whole narration.
PROLOGUE_STAGES: tuple[Stage, ...] = (
    Stage.INGEST,
    Stage.TRANSCRIBE,
    Stage.CONTEXT,
    Stage.ENTITIES,
    Stage.SEGMENT,
    Stage.QUERIES,
)

#: Everything after: whole-project summaries over the finished segments.
EPILOGUE_STAGES: tuple[Stage, ...] = (Stage.TIMESTAMPS, Stage.COLLECT, Stage.REPORT)


def deliver_segment(ctx: RunContext) -> tuple[int, int]:
    """Publish everything finished so far into selected/ + sources.csv.

    Returns ``(files, rows)``, which §21 requires to be equal.

    `collect` is already idempotent and diff-based, and segments that have not
    been ranked yet simply contribute no picks, so calling it with the whole
    segment list after each segment adds exactly the newly-finished files and
    touches nothing else. `write_sources_csv` then derives its rows from that
    same CollectResult, which is what keeps the two counts equal by
    construction rather than by coincidence.
    """
    from ..pipeline.all_candidates import dump_all_candidates
    from ..pipeline.collect import collect
    from ..pipeline.report import write_sources_csv

    segments = _all_segments(ctx)
    result = collect(segments, ctx.paths, ctx.settings, width=ctx.width)
    ctx.collect_result = result
    rows = write_sources_csv(result, segments, ctx.paths, width=ctx.width)
    # The flat browse folder fills alongside selected/, for the same reason
    # selected/ fills per segment rather than at the end: on a 170-segment run
    # nobody wants to wait an hour before they can look at the alternatives.
    dump_all_candidates(segments, ctx.paths, ctx.settings, width=ctx.width)
    files = sum(1 for f in ctx.paths.selected_dir.iterdir() if f.is_file())
    if files != rows:
        # §21 is an invariant, not an aspiration. If it ever breaks, the run
        # says so loudly rather than shipping a folder whose credits are wrong.
        log.error(
            "§21 violated: selected/ holds %d file(s) but sources.csv has %d row(s)",
            files,
            rows,
        )
    return files, rows


def segment_is_complete(ctx: RunContext, segment: Segment) -> bool:
    """Has this segment already been searched, ranked and delivered?

    Read off disk rather than from a checkpoint. Stage checkpoints are recorded
    per *stage*, and a pipelined run touches each stage 170 times, so they
    cannot say which segments are finished. The files can: the image manifest
    is written once download and dedupe have run, and picks appear once ranking
    has. A segment killed mid-download leaves no manifest and is redone whole,
    which is what makes resume safe rather than merely fast.

    A segment that genuinely found nothing still counts as complete -- it has a
    manifest with no ranked records -- or it would be retried on every resume
    forever.
    """
    manifest = ctx.paths.segment_image_manifest(segment, width=ctx.width)
    if not manifest.exists():
        return False
    if segment.picks:
        return True
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not any((r.get("rank") or 0) >= 1 for r in payload.get("kept", []) or [])


def run_segment_pipelined(
    ctx: RunContext,
    *,
    on_stage: Callable[[StageResult], None] | None = None,
    on_segment: Callable[[int, int, int], None] | None = None,
    resume: bool = True,
) -> list[StageResult]:
    """Run the pipeline delivering one segment at a time, in narration order.

    ``resume`` skips whatever is already finished: checkpointed prologue stages
    and segments that already have their files. It is on by default because a
    long run is exactly the kind that gets interrupted, and redoing 20 segments
    of live API calls to reach segment 21 is the expensive kind of correct.
    """
    from .queue import set_state

    results: list[StageResult] = []
    resume = resume and not ctx.force
    done = set(completed_stages(ctx.db_path, ctx.job_id)) if resume else set()

    with _Heartbeat(ctx.db_path, ctx.job_id):
        for stage in PROLOGUE_STAGES:
            result = _run_one_stage(ctx, stage, on_stage, done=done)
            results.append(result)
            if not result.ok:
                return results

        segments = _all_segments(ctx)
        total = len(segments)
        set_state(ctx.db_path, ctx.job_id, JobState.SEARCHING_IMAGES, segments_total=total)

        # Chronological, so selected/ fills front-to-back and the repetition
        # window sees the same history the batch run would have given it.
        skipped = 0
        for position, segment in enumerate(sorted(segments, key=lambda s: s.index), start=1):
            if resume and segment_is_complete(ctx, segment):
                skipped += 1
                ctx.delivered.add(segment.index)
                continue
            if skipped:
                log.info(
                    "resuming at segment %03d - %d segment(s) already done, left alone",
                    segment.index,
                    skipped,
                )
                skipped = 0

            ctx.only_segments = {segment.index}
            try:
                for stage in PER_SEGMENT_STAGES:
                    result = _run_one_stage(ctx, stage, on_stage, quiet=True)
                    if not result.ok:
                        results.append(result)
                        return results
            finally:
                ctx.only_segments = None

            files, rows = deliver_segment(ctx)
            ctx.delivered.add(segment.index)
            log.info(
                "segment %03d delivered - %d of %d segment(s) done, "
                "selected/ %d file(s), sources.csv %d row(s)",
                segment.index,
                position,
                total,
                files,
                rows,
            )
            set_state(
                ctx.db_path,
                ctx.job_id,
                JobState.SEARCHING_IMAGES,
                segments_done=position,
                segments_total=total,
            )
            if on_segment is not None:
                on_segment(segment.index, position, total)

        for stage in EPILOGUE_STAGES:
            result = _run_one_stage(ctx, stage, on_stage)
            results.append(result)
            if not result.ok:
                return results

    return results


def run_pipeline(
    ctx: RunContext,
    *,
    stages: list[Stage] | None = None,
    on_stage: Callable[[StageResult], None] | None = None,
) -> list[StageResult]:
    """Run the pipeline for one job. Returns a result per attempted stage.

    Raises nothing for stage failure: the job is marked FAILED and the error is
    stored. The caller decides what to print.
    """
    wanted = list(stages) if stages else list(STAGE_ORDER)
    done = set() if ctx.force else completed_stages(ctx.db_path, ctx.job_id)
    results: list[StageResult] = []

    with _Heartbeat(ctx.db_path, ctx.job_id):
        return _run_stages(ctx, wanted, done, results, on_stage)


def _run_one_stage(ctx, stage, on_stage=None, *, quiet: bool = False, done=()) -> StageResult:
    """Execute one stage with its state update, timing and error handling.

    Shared by the batch runner and segment pipelining so the two cannot drift
    in how a failure is recorded. ``quiet`` suppresses the per-stage start/done
    lines: in pipelined mode they would fire 6 x 95 times and bury the
    per-segment delivery lines that are the actual progress signal.
    """
    impl = IMPLEMENTED_STAGES.get(stage)
    if impl is None:
        phase = _PLANNED.get(stage, "a later phase")
        if not quiet:
            log.info("stage %-13s skipped - not implemented yet (%s)", str(stage), phase)
        return StageResult(stage, ok=True, skipped=True, detail=f"not implemented ({phase})")

    if str(stage) in done:
        # Resuming. Without this a resumed pipelined run re-transcribed the
        # whole narration despite `transcribe` being checkpointed, because the
        # skip lived only in the batch runner's loop.
        if not quiet:
            log.info("stage %-13s skipped - already checkpointed (use --force to redo)", str(stage))
        return StageResult(stage, ok=True, skipped=True, detail="checkpointed")

    state = STAGE_STATE.get(stage, JobState.QUEUED)
    set_state(ctx.db_path, ctx.job_id, state, stage=str(stage))
    if not quiet:
        log.info("stage %-13s started", str(stage))
    started = time.perf_counter()

    try:
        result = impl(ctx)
    except DiskSpaceError as exc:
        log.error("stage %s aborted: %s", str(stage), exc)
        set_state(ctx.db_path, ctx.job_id, JobState.FAILED, stage=str(stage), error=str(exc))
        return StageResult(stage, ok=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - the runner is the top-level guard
        elapsed = time.perf_counter() - started
        log.exception("stage %s failed after %.1fs", str(stage), elapsed)
        record_checkpoint(
            ctx.db_path, ctx.job_id, stage, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )
        set_state(
            ctx.db_path,
            ctx.job_id,
            JobState.FAILED,
            stage=str(stage),
            error=f"{type(exc).__name__}: {exc}",
        )
        return StageResult(stage, ok=False, detail=f"{type(exc).__name__}: {exc}")

    result.duration_s = time.perf_counter() - started
    if not quiet:
        log.info("stage %-13s done in %.1fs - %s", str(stage), result.duration_s, result.detail)
        record_checkpoint(
            ctx.db_path,
            ctx.job_id,
            stage,
            ok=True,
            artifact=result.artifact,
            detail=result.detail,
            duration_s=result.duration_s,
        )
    if on_stage:
        on_stage(result)
    return result


def _run_stages(ctx, wanted, done, results, on_stage):
    for stage in wanted:
        heartbeat(ctx.db_path, ctx.job_id)
        impl = IMPLEMENTED_STAGES.get(stage)

        if impl is None:
            phase = _PLANNED.get(stage, "a later phase")
            log.info("stage %-13s skipped - not implemented yet (%s)", str(stage), phase)
            results.append(
                StageResult(stage, ok=True, skipped=True, detail=f"not implemented ({phase})")
            )
            continue

        if str(stage) in done:
            log.info("stage %-13s skipped - already checkpointed (use --force to redo)", str(stage))
            results.append(StageResult(stage, ok=True, skipped=True, detail="checkpointed"))
            continue

        state = STAGE_STATE.get(stage, JobState.QUEUED)
        set_state(ctx.db_path, ctx.job_id, state, stage=str(stage))
        log.info("stage %-13s started", str(stage))
        started = time.perf_counter()

        try:
            result = impl(ctx)
        except DiskSpaceError as exc:
            # Never a code bug; always the user's disk. Say which drive (§3).
            log.error("stage %s aborted: %s", str(stage), exc)
            set_state(ctx.db_path, ctx.job_id, JobState.FAILED, stage=str(stage), error=str(exc))
            results.append(StageResult(stage, ok=False, detail=str(exc)))
            return results
        except Exception as exc:  # noqa: BLE001 - the runner is the top-level guard
            elapsed = time.perf_counter() - started
            log.exception("stage %s failed after %.1fs", str(stage), elapsed)
            record_checkpoint(
                ctx.db_path, ctx.job_id, stage, ok=False, detail=f"{type(exc).__name__}: {exc}"
            )
            set_state(
                ctx.db_path,
                ctx.job_id,
                JobState.FAILED,
                stage=str(stage),
                error=f"{type(exc).__name__}: {exc}",
            )
            results.append(StageResult(stage, ok=False, detail=f"{type(exc).__name__}: {exc}"))
            return results

        result.duration_s = time.perf_counter() - started
        log.info("stage %-13s done in %.1fs - %s", str(stage), result.duration_s, result.detail)
        record_checkpoint(
            ctx.db_path,
            ctx.job_id,
            stage,
            ok=True,
            artifact=result.artifact,
            detail=result.detail,
            duration_s=result.duration_s,
        )
        results.append(result)
        if on_stage:
            on_stage(result)

    _finalize(ctx, results)
    return results


def _finalize(ctx: RunContext, results: list[StageResult]) -> None:
    """Set the terminal state (§14).

    A job that produced a gap ends REVIEW_REQUIRED, not COMPLETE. §14 is
    explicit about that, and it is the right default: a gap means a segment
    has nothing on screen, which the user must see before the project is
    treated as finished.
    """
    from ..pipeline.collect import CollectResult
    from ..utils.files import dir_size_bytes

    ran = [r for r in results if not r.skipped]
    unbuilt = [r for r in results if r.skipped and "not implemented" in r.detail]

    result = ctx.collect_result
    gaps = len(result.gaps) if isinstance(result, CollectResult) else 0
    delivered = result.total if isinstance(result, CollectResult) else 0

    fields = {
        "segments_total": len(ctx.segments),
        "gaps": gaps,
        "bytes_used": dir_size_bytes(ctx.project_root),
    }

    if unbuilt:
        set_state(
            ctx.db_path,
            ctx.job_id,
            JobState.REVIEW_REQUIRED,
            stage=str(results[-1].stage) if results else "",
            **fields,
        )
        log.info(
            "job %s: %d stages ran, %d still to be built - state REVIEW_REQUIRED",
            ctx.job_id,
            len(ran),
            len(unbuilt),
        )
        return

    if gaps:
        set_state(ctx.db_path, ctx.job_id, JobState.REVIEW_REQUIRED, **fields)
        log.info(
            "job %s REVIEW_REQUIRED: %d file(s) delivered, %d segment(s) have a gap",
            ctx.job_id,
            delivered,
            gaps,
        )
        return

    set_state(ctx.db_path, ctx.job_id, JobState.COMPLETE, **fields)
    log.info("job %s COMPLETE: %d file(s) in selected/", ctx.job_id, delivered)


def worker_id() -> str:
    return f"{os.environ.get('COMPUTERNAME', 'host')}-{os.getpid()}"
