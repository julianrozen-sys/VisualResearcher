"""``all_candidates/`` — every ranked candidate for a project, in one flat folder.

``selected/`` holds the single pick per segment the pipeline is confident about.
This is the view for changing your mind about one: every image that survived
ranking, laid out so you can scan the alternatives for a segment and swap one
in without opening ``segments/NNN_.../images/`` folder by folder.

Three things define it, and each has a specific way of going wrong:

**It is the *ranked* set, not the searched pool.** Ranking keeps
``images.keep_per_segment`` images and gives them ranks 1..N; everything else
that survived download and dedupe stays in the manifest with ``rank: 0``. Those
unranked leftovers are not candidates in any meaningful sense -- nothing judged
them -- so including them would triple the folder with images the ranker never
endorsed, and numbering them would invent a ranking that does not exist.

**Alphabetical order must equal narration order.** See
:func:`~..utils.files.candidate_filename` for the three fields that carry that
guarantee and how each one breaks.

**The SELECTED marker must never go stale.** It is derived from
:func:`~.collect.plan_selected` -- the same function that decides ``selected/``
itself -- so the two cannot disagree about which image is current. A marker
computed independently would be a second opinion, and second opinions drift.
The marker is a *suffix* for the same reason the timecode is not first: a
leading marker would sort every chosen file into one clump at the top of the
folder and destroy the chronological ordering that is the point of it.

Everything here copies. The originals under ``segments/`` are never moved or
deleted, and neither is anything a person put in the folder themselves (§2.8).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..schemas import Segment
from ..utils.files import (
    atomic_write_text,
    candidate_filename,
    ensure_dir,
    pad_width,
    rank_pad_width,
    slugify,
)

__all__ = [
    "AllCandidate",
    "AllCandidatesResult",
    "dump_all_candidates",
    "plan_all_candidates",
    "DIR_NAME",
    "MANIFEST_NAME",
    "SELECTED_MARKER",
]

log = get_logger("pipeline.all_candidates")

#: Per project, never shared. `projects/<name>/all_candidates/`.
DIR_NAME = "all_candidates"

#: Our own bookkeeping, so a re-run can clean up after itself without ever
#: deleting a file a person dropped in.
MANIFEST_NAME = ".all_candidates_manifest.json"

#: Suffix, not prefix -- a leading marker would clump every chosen file at the
#: top of the folder and destroy the chronological ordering.
SELECTED_MARKER = "_SELECTED"


@dataclass
class AllCandidate:
    name: str
    source: Path
    segment_index: int
    rank: int
    score: float
    provider: str
    query: str
    selected: bool = False


@dataclass
class AllCandidatesResult:
    written: list[AllCandidate] = field(default_factory=list)
    unchanged: list[AllCandidate] = field(default_factory=list)
    remarked: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    foreign: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)
    segments_covered: int = 0

    @property
    def total(self) -> int:
        return len(self.written) + len(self.unchanged)

    @property
    def selected_count(self) -> int:
        return sum(1 for c in self.written + self.unchanged if c.selected)


def _selected_sources(
    segments: list[Segment], paths: ProjectPaths, settings: Settings, *, width: int
) -> set[str]:
    """Absolute paths of the originals that `selected/` currently holds.

    Derived from the collect planner rather than from ``pick.use`` alone: a
    pick can be marked used and still not reach ``selected/`` when it sits
    below the confidence gate, and the marker has to describe the folder as it
    actually is.
    """
    from .collect import plan_selected

    planned, _gaps = plan_selected(segments, paths, settings, width=width)
    out: set[str] = set()
    for chosen in planned:
        source = Path(chosen.source)
        if not source.is_absolute():
            source = paths.root / source
        out.add(str(source.resolve()).lower())
    return out


def plan_all_candidates(
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int | None = None,
) -> list[AllCandidate]:
    """Decide the whole folder, in sorted order. Pure: nothing is written.

    Returned in the order the files should sort, so a caller can assert
    ``sorted(names) == names`` rather than trusting the format by eye.
    """
    from .image_search import read_candidate_manifest

    if width is None:
        width = pad_width(max((s.index for s in segments), default=1))
    chosen = _selected_sources(segments, paths, settings, width=width)

    planned: list[AllCandidate] = []
    for segment in sorted(segments, key=lambda s: s.index):
        manifest = read_candidate_manifest(
            paths.segment_image_manifest(segment, width=width), root=paths.root
        )
        # rank >= 1 is the ranked keep set. Everything else in `kept` survived
        # download and dedupe but was never ranked, and carries rank 0.
        ranked = [r for r in manifest.get("kept", []) if r.local_path and (r.rank or 0) >= 1]
        if not ranked:
            continue
        ranked.sort(key=lambda r: r.rank)
        rank_width = rank_pad_width(max(r.rank for r in ranked))

        for record in ranked:
            source = Path(record.local_path)
            if not source.is_absolute():
                source = paths.root / source
            is_selected = str(source.resolve()).lower() in chosen
            # The query says *why* this candidate turned up, which is the most
            # useful thing to read when scanning for a better pick.
            label = record.query or segment.topic or segment.narration
            name = candidate_filename(
                segment.index,
                segment.start,
                record.rank,
                slugify(label, max_len=settings.output.slug_max_len),
                source.suffix,
                width,
                rank_width,
            )
            if is_selected:
                stem, _, ext = name.rpartition(".")
                name = f"{stem}{SELECTED_MARKER}.{ext}"
            planned.append(
                AllCandidate(
                    name=name,
                    source=source,
                    segment_index=segment.index,
                    rank=record.rank,
                    score=record.score or 0.0,
                    provider=record.provider or "",
                    query=record.query or "",
                    selected=is_selected,
                )
            )
    return planned


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [str(n) for n in json.loads(path.read_text(encoding="utf-8")).get("files", [])]
    except Exception as exc:  # noqa: BLE001 - a broken manifest is not fatal
        log.warning("all_candidates manifest unreadable, treating as empty: %s", exc)
        return []


def _unmarked(name: str) -> str:
    """The same filename without its SELECTED marker."""
    stem, _, ext = name.rpartition(".")
    if stem.endswith(SELECTED_MARKER):
        return f"{stem[: -len(SELECTED_MARKER)]}.{ext}"
    return name


def dump_all_candidates(
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int | None = None,
) -> AllCandidatesResult:
    """Make ``all_candidates/`` hold every ranked candidate, marker current."""
    result = AllCandidatesResult()
    target = paths.root / DIR_NAME
    ensure_dir(target)

    planned = plan_all_candidates(segments, paths, settings, width=width)
    result.segments_covered = len({c.segment_index for c in planned})
    wanted = {c.name: c for c in planned}
    previously_ours = set(_read_manifest(target / MANIFEST_NAME))
    on_disk = {p.name for p in target.iterdir() if p.is_file() and p.name != MANIFEST_NAME}

    # A pick moving is a *rename*, not a re-copy: the bytes did not change, only
    # which file carries the marker. Renaming keeps a folder of 800 images from
    # being rewritten every time somebody swaps one pick in the review UI.
    by_unmarked = {_unmarked(n): n for n in on_disk}
    for name in wanted:
        if name in on_disk:
            continue
        existing = by_unmarked.get(_unmarked(name))
        if existing and existing in previously_ours and existing not in wanted:
            (target / existing).rename(target / name)
            on_disk.discard(existing)
            on_disk.add(name)
            previously_ours.discard(existing)
            result.remarked.append(name)

    for name, entry in wanted.items():
        destination = target / name
        if not entry.source.exists():
            result.missing_sources.append(str(entry.source))
            continue
        if destination.exists() and destination.stat().st_size == entry.source.stat().st_size:
            result.unchanged.append(entry)
        else:
            # §6: copies, never moves. The segment folder keeps its original.
            shutil.copy2(entry.source, destination)
            result.written.append(entry)

    for name in sorted(on_disk - set(wanted)):
        if name in previously_ours:
            (target / name).unlink()
            result.removed.append(name)
        else:
            result.foreign.append(name)
            log.warning("leaving %s in %s/: this module did not create it", name, DIR_NAME)

    current = sorted(name for name in wanted if (target / name).exists())
    atomic_write_text(
        target / MANIFEST_NAME,
        json.dumps(
            {
                "files": current,
                "note": (
                    "Written by the all_candidates dump. It lists the files here that "
                    "VisualResearcher created, so a later run can remove or re-mark the "
                    "ones it no longer wants without touching anything you put here "
                    "yourself. Safe to delete; you just lose that protection."
                ),
            },
            indent=2,
        ),
    )

    log.info(
        "%s/: %d file(s) across %d segment(s), %d marked SELECTED "
        "(%d new, %d unchanged, %d re-marked, %d removed)",
        DIR_NAME,
        result.total,
        result.segments_covered,
        result.selected_count,
        len(result.written),
        len(result.unchanged),
        len(result.remarked),
        len(result.removed),
    )
    return result


def refresh_if_present(
    segments: list[Segment],
    paths: ProjectPaths,
    settings: Settings,
    *,
    width: int | None = None,
) -> AllCandidatesResult | None:
    """Update the folder, but only for a project that already has one.

    Called wherever ``selected/`` is rebuilt, so the SELECTED marker cannot go
    stale after a re-run or a swap in the review UI. It stays opt-in: a project
    that never asked for the dump does not silently grow an 800-file copy of
    itself the first time somebody re-collects.
    """
    if not (paths.root / DIR_NAME).is_dir():
        return None
    return dump_all_candidates(segments, paths, settings, width=width)
