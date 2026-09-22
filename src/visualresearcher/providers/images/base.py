"""Image search provider interface (CLAUDE.md §11).

A provider's only job is to turn a query into candidate URLs plus whatever
attribution the source gave us. It does not download, validate, hash, score or
decide anything -- all of that belongs to the pipeline (§2.7).

Attribution is collected here rather than later because it is only available
here: once you have a bare image URL, the licence and the page it came from
are gone, and §13 requires a row in `sources.csv` for every asset.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..base import Provider

__all__ = ["ImageCandidate", "ImageSearchProvider"]


@dataclass
class ImageCandidate:
    """One search hit, before anything has been downloaded."""

    image_url: str
    source_page: str = ""
    title: str = ""
    #: Reported by the source; treated as a hint only, never trusted (§11).
    width: int = 0
    height: int = 0
    creator: str = ""
    license: str = "unknown"
    license_url: str = ""
    provider: str = ""
    query: str = ""
    query_kind: str = ""
    thumbnail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.source_page or self.image_url).netloc.lower()
        except ValueError:
            return ""

    def dedupe_key(self) -> str:
        """Cheap pre-download identity, so the same URL is not fetched twice."""
        return self.image_url.split("?")[0].lower()


class ImageSearchProvider(Provider):
    kind = "images"

    @abstractmethod
    def search(self, query: str, *, limit: int = 30, **kwargs) -> list[ImageCandidate]:
        """Return up to ``limit`` candidates for ``query``.

        Must not raise for an empty result -- an empty list is a valid answer.
        Raising is reserved for transport failures, and even then the caller
        continues with the other providers (§11: one failing provider never
        stops a segment).
        """
