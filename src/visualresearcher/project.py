"""The output folder contract (CLAUDE.md §6).

§5's module list does not name this file, but every stage needs the same answer
to "where does this go", and duplicating those joins across fifteen stages is
how ``selected/`` ends up with orphans. One place, one contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import Segment
from .utils.files import ensure_dir, segment_dirname

__all__ = ["ProjectPaths"]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for one project. Construct with ``ProjectPaths(root)``."""

    root: Path

    # -- top level -------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def narration_wav(self) -> Path:
        return self.input_dir / "narration.wav"

    @property
    def transcript_json(self) -> Path:
        return self.input_dir / "transcript.json"

    @property
    def transcript_txt(self) -> Path:
        return self.input_dir / "transcript.txt"

    @property
    def narration_srt(self) -> Path:
        return self.input_dir / "narration.srt"

    @property
    def project_context(self) -> Path:
        return self.root / "project_context.json"

    @property
    def entity_overrides(self) -> Path:
        return self.root / "entity_overrides.yaml"

    @property
    def timeline_csv(self) -> Path:
        return self.root / "timeline.csv"

    @property
    def shotlist_md(self) -> Path:
        return self.root / "shotlist.md"

    @property
    def sources_csv(self) -> Path:
        return self.root / "sources.csv"

    @property
    def research_report(self) -> Path:
        return self.root / "research_report.md"

    @property
    def job_log(self) -> Path:
        return self.root / "job.log"

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def selected_dir(self) -> Path:
        """THE DELIVERABLE. Flat, no subfolders (§6)."""
        return self.root / "selected"

    # -- per segment -----------------------------------------------------
    def segment_dir(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segments_dir / segment_dirname(segment.index, segment.start, segment.end, width)

    def segment_json(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_dir(segment, width=width) / "segment.json"

    def segment_images_dir(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_dir(segment, width=width) / "images"

    def segment_image_manifest(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_images_dir(segment, width=width) / "manifest.json"

    def segment_clips_dir(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_dir(segment, width=width) / "clips"

    def segment_youtube_results(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_dir(segment, width=width) / "youtube" / "results.json"

    def segment_contact_sheet(self, segment: Segment, *, width: int = 3) -> Path:
        return self.segment_dir(segment, width=width) / "contact_sheet.jpg"

    # -- creation --------------------------------------------------------
    def ensure_base(self) -> None:
        for path in (self.input_dir, self.segments_dir, self.selected_dir):
            ensure_dir(path)

    def ensure_segment(self, segment: Segment, *, width: int = 3) -> Path:
        base = self.segment_dir(segment, width=width)
        for path in (
            base,
            self.segment_images_dir(segment, width=width),
            self.segment_clips_dir(segment, width=width),
            base / "youtube",
        ):
            ensure_dir(path)
        return base

    def existing_segment_dirs(self) -> list[Path]:
        if not self.segments_dir.exists():
            return []
        return sorted(p for p in self.segments_dir.iterdir() if p.is_dir())
