"""Stage 15: the reports (CLAUDE.md §6, §13).

Three files, each answering a different question:

* ``sources.csv`` — **the credits list for the video description.** §13 is
  emphatic: one row per file in ``selected/``, every asset gets a row, no
  silent assets. A licence that is not known is written as ``unknown`` rather
  than guessed at, because a wrong attribution is worse than an absent one.
* ``shotlist.md`` — what to put on screen, segment by segment, and **the
  gaps**. A gap is the whole point: §6 forbids fabricating a placeholder, so
  the only honest way to report "nothing was found here" is to say so
  somewhere the user will look.
* ``research_report.md`` — what the run actually did. Entity corrections with
  their reasons, provider coverage, where confidence came from, what degraded.

All three are derived from the same plan the collect stage used, so the row
count in ``sources.csv`` and the file count in ``selected/`` cannot drift.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import ImageRecord, ProjectContext, Segment
from ..utils.files import atomic_write_text, dir_size_bytes
from ..utils.timefmt import timecode
from .collect import CollectResult, SelectedFile
from .context import BATCH_SIZE

__all__ = [
    "write_sources_csv",
    "write_shotlist",
    "write_research_report",
    "write_reports",
    "SOURCES_COLUMNS",
]

log = get_logger("pipeline.report")

#: §13's columns, in §13's order.
SOURCES_COLUMNS = [
    "segment",
    "timecode",
    "file",
    "type",
    "source_url",
    "source_page",
    "channel_or_creator",
    "timestamp",
    "license",
    "license_url",
    "query",
]


@dataclass
class _Provenance:
    """Where one selected file came from, gathered from the segment artifacts."""

    source_url: str = ""
    source_page: str = ""
    channel_or_creator: str = ""
    timestamp: str = ""
    license: str = "unknown"
    license_url: str = ""
    query: str = ""


def _image_index(paths: ProjectPaths, segment: Segment, width: int) -> dict[str, ImageRecord]:
    from .image_search import read_candidate_manifest

    manifest = read_candidate_manifest(
        paths.segment_image_manifest(segment, width=width), root=paths.root
    )
    index: dict[str, ImageRecord] = {}
    for record in manifest["kept"] + manifest["duplicates"]:
        index[Path(record.local_path).name] = record
    return index


def _video_index(paths: ProjectPaths, segment: Segment, width: int) -> dict[str, dict]:
    path = paths.segment_youtube_results(segment, width=width)
    if not path.exists():
        return {}
    index: dict[str, dict] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        clip = item.get("downloaded_clip_path") or ""
        if clip:
            index[Path(clip).name] = item
    return index


def _provenance(
    entry: SelectedFile,
    images: dict[str, ImageRecord],
    videos: dict[str, dict],
) -> _Provenance:
    """Trace one selected file back to the record that produced it.

    Falls back to a clearly-marked unknown rather than inventing anything:
    §13 says to mark a licence unknown rather than guess, and the same
    reasoning applies to every other column.
    """
    name = entry.source.name

    if entry.kind == "clip":
        record = videos.get(name)
        if record:
            best = (record.get("timestamp_candidates") or [None])[0]
            return _Provenance(
                source_url=record.get("url", ""),
                source_page=record.get("url", "").split("&t=")[0],
                channel_or_creator=record.get("channel", ""),
                timestamp=timecode(best["start_s"]) if best else "",
                # YouTube gives no machine-readable licence; §13 says say so.
                license="unknown",
                license_url="",
                query=record.get("query", ""),
            )
        return _Provenance(timestamp="", license="unknown")

    record = images.get(name)
    if record:
        return _Provenance(
            source_url=record.image_url,
            source_page=record.source_page,
            channel_or_creator=record.creator,
            timestamp="",
            license=record.license or "unknown",
            license_url=record.license_url,
            query=record.query,
        )
    return _Provenance(license="unknown")


def write_sources_csv(
    result: CollectResult,
    segments: list[Segment],
    paths: ProjectPaths,
    *,
    width: int = 3,
) -> int:
    """One row per file in ``selected/`` (§13). Returns the row count.

    The rows are built from the collect result, not from a directory listing,
    so "every asset gets a row" holds by construction rather than by luck.
    """
    by_index = {s.index: s for s in segments}
    image_cache: dict[int, dict[str, ImageRecord]] = {}
    video_cache: dict[int, dict[str, dict]] = {}

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SOURCES_COLUMNS, lineterminator="\n")
    writer.writeheader()

    entries = sorted(
        result.written + result.unchanged, key=lambda e: (e.segment_index, e.pick_index)
    )
    for entry in entries:
        segment = by_index.get(entry.segment_index)
        if segment is None:
            continue
        if entry.segment_index not in image_cache:
            image_cache[entry.segment_index] = _image_index(paths, segment, width)
            video_cache[entry.segment_index] = _video_index(paths, segment, width)

        provenance = _provenance(
            entry, image_cache[entry.segment_index], video_cache[entry.segment_index]
        )
        writer.writerow(
            {
                "segment": f"{entry.segment_index:0{width}d}",
                "timecode": timecode(entry.segment_start),
                "file": entry.name,
                "type": entry.kind,
                "source_url": provenance.source_url,
                "source_page": provenance.source_page,
                "channel_or_creator": provenance.channel_or_creator,
                "timestamp": provenance.timestamp,
                "license": provenance.license or "unknown",
                "license_url": provenance.license_url,
                "query": provenance.query,
            }
        )

    atomic_write_text(paths.sources_csv, buffer.getvalue())
    log.info("wrote %s with %d row(s)", paths.sources_csv.name, len(entries))
    return len(entries)


def write_shotlist(
    result: CollectResult,
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int = 3,
) -> None:
    """``shotlist.md`` — what goes on screen, and where the gaps are (§6)."""
    by_segment: dict[int, list[SelectedFile]] = {}
    for entry in result.written + result.unchanged:
        by_segment.setdefault(entry.segment_index, []).append(entry)
    for entries in by_segment.values():
        entries.sort(key=lambda e: e.pick_index)

    gaps = {s.index for s in result.gaps}
    lines: list[str] = [
        f"# Shot list — {paths.root.name}",
        "",
        f"{len(segments)} segments · {result.total} file(s) in `selected/` · "
        f"**{len(gaps)} gap(s)**",
        "",
    ]

    if gaps:
        lines += [
            "## Gaps",
            "",
            "Nothing confident enough was found for these segments, so "
            "`selected/` holds no file for them. Nothing was fabricated to "
            "fill the space (CLAUDE.md §6).",
            "",
            "| Segment | Timecode | Narration | Why |",
            "| --- | --- | --- | --- |",
        ]
        for segment in result.gaps:
            why = next(
                (n for n in reversed(segment.notes) if "clip" in n.lower() or "no " in n.lower()),
                "no pick reached the confidence threshold",
            )
            lines.append(
                f"| {segment.index:0{width}d} | {timecode(segment.start)} | "
                f"{_cell(segment.narration, 70)} | {_cell(why, 80)} |"
            )
        lines.append("")

    if result.flagged:
        lines += [
            "## Flagged — medium confidence",
            "",
            f"{len(result.flagged)} file(s) were copied but are worth a look "
            f"(§14: {settings.confidence.medium:.2f}–{settings.confidence.high:.2f}).",
            "",
        ]
        for entry in sorted(result.flagged, key=lambda e: (e.segment_index, e.pick_index)):
            lines.append(f"- `{entry.name}` — {entry.confidence:.2f}")
        lines.append("")

    lines += ["## Shots", ""]
    for segment in segments:
        marker = " — **GAP**" if segment.index in gaps else ""
        lines += [
            f"### {segment.index:0{width}d} · {timecode(segment.start)}–"
            f"{timecode(segment.end)}{marker}",
            "",
            f"> {segment.narration}",
            "",
        ]
        if segment.interpretation:
            lines += [f"*{segment.interpretation}*", ""]
        details = []
        if segment.entities:
            details.append(f"**entities** {', '.join(segment.entities)}")
        if segment.location:
            details.append(f"**location** {segment.location}")
        details.append(f"**intent** {segment.visual_intent}")
        lines += ["  \n".join(details), ""]

        for entry in by_segment.get(segment.index, []):
            lines.append(
                f"- `{entry.name}` · {entry.kind} · confidence {entry.confidence:.2f} "
                f"({entry.band})"
            )
        if segment.index in gaps:
            lines.append("- _nothing selected_")
        lines.append("")

    atomic_write_text(paths.shotlist_md, "\n".join(lines))
    log.info("wrote %s (%d gap(s))", paths.shotlist_md.name, len(gaps))


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def _estimate_llm_calls(segment_count: int) -> int:
    """§20: one context call plus one per batch of segments."""
    import math

    return 1 + math.ceil(segment_count / BATCH_SIZE) if segment_count else 1


def _cell(text: str, limit: int) -> str:
    """Escape a value for a markdown table cell."""
    flat = " ".join((text or "").split()).replace("|", "\\|")
    return flat[:limit] + ("…" if len(flat) > limit else "")


def write_research_report(
    result: CollectResult,
    segments: list[Segment],
    context: ProjectContext,
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int = 3,
    timings: list[tuple[str, float, str]] | None = None,
) -> None:
    """``research_report.md`` — what the run did, and how much to trust it."""
    degraded = [s for s in segments if s.status == "degraded"]
    with_clips = [s for s in segments if any(p.kind == "clip" and p.use for p in s.picks)]
    applied = [c for c in context.entity_corrections if c.applied]
    unapplied = [c for c in context.entity_corrections if not c.applied]

    lines: list[str] = [
        f"# Research report — {paths.root.name}",
        "",
        "## Subject",
        "",
        f"- **Subject** {context.subject or '_not identified_'}",
        f"- **Franchise** {context.franchise or '—'}",
        f"- **Era** {context.era or '—'}",
        f"- **Search tag** `{context.search_tag or '—'}`",
        f"- **Domain packs** {', '.join(context.domain_packs) or 'none active'}",
        f"- **Analysed by** {context.provider}",
        "",
        "## Outcome",
        "",
        f"- {len(segments)} segments",
        f"- {result.total} file(s) in `selected/`",
        f"- {len(result.gaps)} gap(s)",
        f"- {len(result.flagged)} file(s) flagged medium-confidence",
        f"- {len(with_clips)} segment(s) with a video clip",
        f"- {len(degraded)} segment(s) degraded during the run",
        "",
    ]

    if context.entity_corrections:
        lines += [
            "## Entity resolution",
            "",
            "The transcript on disk keeps the original words; these corrections "
            "were applied to **search queries only** (CLAUDE.md §10.2).",
            "",
            "| Original | Resolved | Confidence | Method | Applied | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for correction in context.entity_corrections:
            lines.append(
                f"| {correction.original} | {correction.resolved} | "
                f"{correction.confidence:.2f} | {correction.method} | "
                f"{'yes' if correction.applied else 'no'} | "
                f"{_cell(correction.reason, 70)} |"
            )
        lines += [""]
        if unapplied:
            lines += [
                f"{len(unapplied)} correction(s) scored below "
                f"{0.75:.2f} and were **not** applied; both spellings were "
                "searched and ranking decided (§10.3).",
                "",
            ]
        else:
            lines += [f"All {len(applied)} correction(s) met the 0.75 threshold.", ""]

    if context.characters or context.places or context.terminology:
        lines += ["## Identified", ""]
        for label, values in (
            ("Characters", context.characters),
            ("People", context.people),
            ("Places", context.places),
            ("Events", context.events),
            ("Terminology", context.terminology),
        ):
            if values:
                lines.append(f"- **{label}** {', '.join(values[:20])}")
        lines.append("")

    if degraded:
        lines += [
            "## Degraded segments",
            "",
            "A provider failed for these; the segment continued with what it had (§8).",
            "",
        ]
        for segment in degraded:
            note = segment.notes[-1] if segment.notes else "no detail recorded"
            lines.append(f"- **{segment.index:0{width}d}** — {_cell(note, 100)}")
        lines.append("")

    if result.missing_sources or result.foreign:
        lines += ["## Needs attention", ""]
        for missing in result.missing_sources:
            lines.append(f"- source file missing when collecting: `{missing}`")
        for foreign in result.foreign:
            lines.append(
                f"- `selected/{foreign}` was not created by this tool, so it was left alone"
            )
        lines.append("")

    if timings:
        total = sum(seconds for _, seconds, _ in timings)
        lines += [
            "## Timing",
            "",
            f"Total pipeline time: **{_duration(total)}**"
            + (
                f" for {len(segments)} segments ({total / len(segments):.1f}s per segment)"
                if segments
                else ""
            ),
            "",
            "| Stage | Time | Share | Detail |",
            "| --- | ---: | ---: | --- |",
        ]
        for stage, seconds, detail in timings:
            share = (seconds / total * 100) if total else 0.0
            lines.append(f"| {stage} | {_duration(seconds)} | {share:.0f}% | {_cell(detail, 70)} |")
        lines += ["", f"Project on disk: **{dir_size_bytes(paths.root) / 1024**2:.0f} MB**", ""]

        llm_calls = _estimate_llm_calls(len(segments))
        lines += [
            "### Cost",
            "",
            "Nothing in this run cost money: image search (ddgs, Wikimedia) and "
            "video search (yt-dlp) need no key, and transcription runs locally.",
            "",
            f"The only paid component is the LLM, and §20's batching keeps it "
            f"small: **{llm_calls} call(s)** for {len(segments)} segments "
            f"(1 for the project context, then {BATCH_SIZE} segments per call). "
            "Without batching that would be "
            f"{len(segments) + 1} calls.",
            "",
        ]

    lines += [
        "## How to read the confidence bands",
        "",
        f"- **high** ≥ {settings.confidence.high:.2f} — copied automatically",
        f"- **medium** {settings.confidence.medium:.2f}–{settings.confidence.high:.2f} "
        "— copied and flagged above",
        f"- **low** < {settings.confidence.medium:.2f} — nothing copied; listed as a gap",
        "",
        "A pick you chose yourself in the review UI is copied regardless of its "
        "band: the bands govern what the pipeline does on its own.",
        "",
    ]

    atomic_write_text(paths.research_report, "\n".join(lines))
    log.info("wrote %s", paths.research_report.name)


def write_reports(
    result: CollectResult,
    segments: list[Segment],
    context: ProjectContext,
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int = 3,
    timings: list[tuple[str, float, str]] | None = None,
) -> int:
    """Write all three reports. Returns the ``sources.csv`` row count."""
    rows = write_sources_csv(result, segments, paths, width=width)
    write_shotlist(result, segments, paths, settings, width=width)
    write_research_report(result, segments, context, paths, settings, width=width, timings=timings)

    # §21: sources.csv row count == selected/ file count. Checked here rather
    # than only in a test, because a mismatch means an asset shipped with no
    # credit -- which is the one thing §13 is written to prevent.
    on_disk = len([p for p in paths.selected_dir.iterdir() if p.is_file()])
    if rows != on_disk:
        log.warning(
            "sources.csv has %d row(s) but selected/ holds %d file(s); "
            "some asset may be missing its credit",
            rows,
            on_disk,
        )
    return rows
