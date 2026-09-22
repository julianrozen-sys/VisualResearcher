"""Stage 14: build ``selected/`` (CLAUDE.md §6, §14, §15).

``selected/`` **is the product.** Everything else in the project folder is
working material; this is the folder the user drags into CapCut. So its rules
are the strictest in the constitution:

* flat, no subfolders;
* ``{segment:03d}_{pick:d}_{slug}.{ext}``, zero-padded so a plain name sort is
  chronological order;
* **copies**, never moves or symlinks -- the originals stay in ``segments/``;
* no confident pick means **no file**, listed as a gap. Never a placeholder;
* changing a pick rewrites the folder exactly: adds and removes, no orphans
  (§15).

The last two pull against §2.8's "never destroy user data", so removal is
deliberately narrow. This module keeps its own manifest of the files *it*
created, and will only ever delete one of those. A file the user put in
``selected/`` by hand is left alone and reported, because guessing that an
unrecognised file is rubbish is exactly the kind of destruction §2.8 forbids.

The manifest lives at the project root rather than inside ``selected/``, so
the deliverable folder stays clean.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import Pick, Segment
from ..utils.disk import check_free_space
from ..utils.files import atomic_write_text, ensure_dir, pad_width, selected_filename, slugify

__all__ = [
    "collect",
    "CollectResult",
    "SelectedFile",
    "confidence_band",
    "plan_selected",
    "MANIFEST_NAME",
]

log = get_logger("pipeline.collect")

MANIFEST_NAME = ".selected_manifest.json"


def confidence_band(score: float, settings: Settings) -> str:
    """``high`` / ``medium`` / ``low`` per §14."""
    if score >= settings.confidence.high:
        return "high"
    if score >= settings.confidence.medium:
        return "medium"
    return "low"


@dataclass
class SelectedFile:
    """One file that belongs in ``selected/``."""

    name: str
    source: Path
    segment_index: int
    pick_index: int
    kind: str
    confidence: float
    band: str
    slug: str
    segment_start: float = 0.0
    segment_end: float = 0.0


@dataclass
class CollectResult:
    written: list[SelectedFile] = field(default_factory=list)
    unchanged: list[SelectedFile] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    #: Segments with nothing confident enough to include (§6, §14).
    gaps: list[Segment] = field(default_factory=list)
    #: Files in selected/ this module did not create. Never deleted.
    foreign: list[str] = field(default_factory=list)
    flagged: list[SelectedFile] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.written) + len(self.unchanged)


def _pick_confidence(pick: Pick, segment: Segment) -> float:
    """A pick's confidence, falling back to the segment's own."""
    return pick.confidence or segment.confidence or 0.0


def plan_selected(
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int | None = None,
) -> tuple[list[SelectedFile], list[Segment]]:
    """Work out exactly which files ``selected/`` should hold.

    Pure: it reads the segments and decides. Nothing is written here, which is
    what makes the add/remove diff in :func:`collect` straightforward to get
    right.
    """
    if width is None:
        width = pad_width(max((s.index for s in segments), default=1))

    planned: list[SelectedFile] = []
    gaps: list[Segment] = []

    for segment in segments:
        usable = [p for p in segment.picks if p.use]
        # §14: below the medium threshold nothing is copied *automatically*.
        # A pick a human chose in the review UI is not automatic, so it is
        # included regardless of band -- otherwise clicking USE THIS would
        # mark the thumbnail used while no file appeared, and the page and the
        # folder would disagree, which is exactly what §15 forbids.
        confident = [
            p
            for p in usable
            if p.user_set or _pick_confidence(p, segment) >= settings.confidence.medium
        ]
        if not confident:
            gaps.append(segment)
            continue

        # Clips first: a moving shot is the better default for a segment that
        # has one, and the pick index follows the order in the folder.
        confident.sort(key=lambda p: (0 if p.kind == "clip" else 1, p.rank, p.path))

        for position, pick in enumerate(confident, start=1):
            source = Path(pick.path)
            if not source.is_absolute():
                source = paths.root / source
            confidence = _pick_confidence(pick, segment)
            slug = pick.slug or slugify(
                segment.topic or segment.narration, max_len=settings.output.slug_max_len
            )
            planned.append(
                SelectedFile(
                    name=selected_filename(segment.index, position, slug, source.suffix, width),
                    source=source,
                    segment_index=segment.index,
                    pick_index=position,
                    kind=pick.kind,
                    confidence=confidence,
                    band=confidence_band(confidence, settings),
                    slug=slug,
                    segment_start=segment.start,
                    segment_end=segment.end,
                )
            )
    return planned, gaps


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(n) for n in data.get("files", [])]
    except Exception as exc:  # noqa: BLE001 - a broken manifest is not fatal
        log.warning("selected manifest unreadable, treating as empty: %s", exc)
        return []


def collect(
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int | None = None,
) -> CollectResult:
    """Make ``selected/`` match the current picks exactly.

    Adds what is missing, removes what this module previously created and no
    longer wants, and leaves anything it did not create alone.
    """
    result = CollectResult()
    ensure_dir(paths.selected_dir)
    check_free_space(paths.selected_dir, settings.output.min_free_gb, stage="collect")

    planned, gaps = plan_selected(segments, paths, settings, width=width)
    result.gaps = gaps

    wanted = {f.name: f for f in planned}
    previously_ours = set(_read_manifest(paths.root / MANIFEST_NAME))
    on_disk = {p.name for p in paths.selected_dir.iterdir() if p.is_file()}

    # -- add --------------------------------------------------------------
    for name, entry in wanted.items():
        destination = paths.selected_dir / name
        if not entry.source.exists():
            log.warning(
                "segment %03d: pick %s is missing from disk; not copying",
                entry.segment_index,
                entry.source,
            )
            result.missing_sources.append(str(entry.source))
            continue

        if destination.exists() and destination.stat().st_size == entry.source.stat().st_size:
            result.unchanged.append(entry)
        else:
            # §6: copies, not moves or symlinks. The original stays put.
            shutil.copy2(entry.source, destination)
            result.written.append(entry)

        if entry.band == "medium":
            result.flagged.append(entry)

    # -- remove (§15: no orphans) -------------------------------------------
    for name in sorted(on_disk - set(wanted)):
        if name in previously_ours:
            (paths.selected_dir / name).unlink()
            result.removed.append(name)
            log.info("removed %s from selected/ (no longer picked)", name)
        else:
            # Not ours. §2.8 -- never destroy user data.
            result.foreign.append(name)
            log.warning("leaving %s in selected/: this module did not create it", name)

    # -- record what is ours now -------------------------------------------
    current = sorted(name for name in wanted if (paths.selected_dir / name).exists())
    atomic_write_text(
        paths.root / MANIFEST_NAME,
        json.dumps(
            {
                "files": current,
                "note": (
                    "Written by the collect stage. It lists the files in selected/ "
                    "that VisualResearcher created, so a later run can remove the "
                    "ones you no longer want without touching anything you added "
                    "yourself. Safe to delete; you will just lose that protection."
                ),
            },
            indent=2,
        ),
    )

    log.info(
        "selected/: %d file(s) (%d new, %d unchanged, %d removed), "
        "%d gap(s), %d flagged medium-confidence",
        result.total,
        len(result.written),
        len(result.unchanged),
        len(result.removed),
        len(result.gaps),
        len(result.flagged),
    )
    if result.foreign:
        log.info("left %d file(s) in selected/ that were not ours", len(result.foreign))
    return result
