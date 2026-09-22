"""Video search and sectioned download interface (CLAUDE.md §12).

Two capabilities, deliberately on one interface: the thing that can find a
video is the thing that knows how to fetch part of it, and splitting them
would mean threading video ids and format selectors through the pipeline.

The pipeline still owns every decision -- which hit is relevant, where the
timestamp is, whether a full download is permitted. This layer only does I/O.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..base import Provider

__all__ = ["VideoHit", "SectionRequest", "SectionResult", "VideoProvider"]


@dataclass
class VideoHit:
    """One search result, with whatever metadata the source exposed.

    ``captions``, ``chapters`` and ``description`` are what the timestamp
    locator works from (§12.3), so they are fetched here rather than in a
    second pass.
    """

    video_id: str = ""
    url: str = ""
    title: str = ""
    channel: str = ""
    duration: float = 0.0
    thumbnail: str = ""
    description: str = ""
    query: str = ""
    view_count: int = 0
    upload_date: str = ""
    #: ``[{"start": float, "end": float, "text": str}]``
    captions: list[dict] = field(default_factory=list)
    #: ``[{"start_time": float, "end_time": float, "title": str}]``
    chapters: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def url_with_t(self, seconds: float) -> str:
        """The clickable ``&t=`` link §12.7 requires, stored regardless of download."""
        base = self.url or (
            f"https://www.youtube.com/watch?v={self.video_id}" if self.video_id else ""
        )
        if not base:
            return ""
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}t={int(max(0, seconds))}s"


@dataclass
class SectionRequest:
    """A request to fetch one span of one video."""

    video_id: str
    url: str
    start_s: float
    end_s: float
    destination: Path
    max_height: int = 1080
    #: Staging directory. The file is moved into place only on success (§12.6).
    tmp_dir: Path | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class SectionResult:
    ok: bool
    path: Path | None = None
    bytes: int = 0
    duration: float = 0.0
    reason: str = ""
    detail: str = ""
    #: True when the span came from a full download rather than a ranged fetch.
    full_download: bool = False


class VideoProvider(Provider):
    kind = "video"

    @abstractmethod
    def search(self, query: str, *, limit: int = 5, **kwargs) -> list[VideoHit]:
        """Search for videos. An empty list is a valid answer, not an error."""

    @abstractmethod
    def fetch_section(self, request: SectionRequest) -> SectionResult:
        """Download just ``[start_s, end_s]`` of one video.

        Must never download the whole file to trim it afterwards -- that is
        the behaviour §12.4 exists to prevent. A provider that cannot do a
        ranged fetch should return ``ok=False`` and let the pipeline decide
        whether the full-download fallback is permitted.
        """

    def probe(self, url: str) -> VideoHit | None:
        """Metadata for one known URL. Optional; the default is "unknown"."""
        return None
