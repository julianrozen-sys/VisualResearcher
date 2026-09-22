"""DuckDuckGo image search via the ``ddgs`` package (CLAUDE.md §4 -- no key).

``ddgs`` is imported lazily so ``doctor`` can report it as missing without the
process failing, and so ``VR_OFFLINE=1`` never needs it.

DuckDuckGo gives no licence information. That is recorded honestly as
``unknown`` rather than guessed at -- §13 says to mark it unknown rather than
guess, because this feeds the credits list in the video description.
"""

from __future__ import annotations

import importlib.util

from ...logging_setup import get_logger
from ..base import Availability, ProviderError
from .base import ImageCandidate, ImageSearchProvider

__all__ = ["DdgsImageProvider"]

log = get_logger("providers.images.ddgs")


class DdgsImageProvider(ImageSearchProvider):
    name = "ddgs"

    def __init__(
        self,
        *,
        min_width: int = 800,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timeout: float = 20.0,
        **_ignored,
    ):
        self.min_width = min_width
        self.region = region
        self.safesearch = safesearch
        self.timeout = timeout

    def availability(self) -> Availability:
        if importlib.util.find_spec("ddgs") is None:
            return Availability.unavailable(
                "the ddgs package is not installed", "uv pip install ddgs"
            )
        return Availability.available("no key required")

    def search(self, query: str, *, limit: int = 30, **kwargs) -> list[ImageCandidate]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - guarded by availability()
            raise ProviderError(f"ddgs is not installed: {exc}") from exc

        try:
            with DDGS(timeout=self.timeout) as client:
                results = list(
                    client.images(
                        query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=limit,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - ddgs raises a variety of types
            raise ProviderError(f"ddgs search failed: {exc}") from exc

        out: list[ImageCandidate] = []
        for item in results:
            width = int(item.get("width") or 0)
            if width and width < self.min_width:
                continue
            out.append(
                ImageCandidate(
                    image_url=item.get("image", "") or "",
                    source_page=item.get("url", "") or "",
                    title=item.get("title", "") or "",
                    width=width,
                    height=int(item.get("height") or 0),
                    creator="",
                    # DuckDuckGo reports no licence; §13 says say so, don't guess.
                    license="unknown",
                    license_url="",
                    provider=self.name,
                    query=query,
                    thumbnail=item.get("thumbnail", "") or "",
                    extra={"source": item.get("source", "")},
                )
            )
        log.debug("ddgs %r -> %d result(s)", query, len(out))
        return out
