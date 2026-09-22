"""Wikimedia Commons image search (CLAUDE.md §4 -- no key required).

Uses the Commons MediaWiki API rather than scraping (§13: "prefer APIs over
HTML scraping"), and asks for ``extmetadata`` so licence and author come back
with the result. Commons is the one source that reliably tells us both, which
makes it the most valuable provider for `sources.csv`.

Rate limiting is honoured three ways: a real User-Agent (the Wikimedia API
policy asks for identification, not for a key), the shared per-host gate that
also paces image downloads, and a retry when Commons answers 429 anyway. The
last two were added after a live run: a burst of segment searches earned a
``429 Too Many Requests`` while the downloader was politely spacing its own
requests to the very same host.
"""

from __future__ import annotations

import html
import re
import time

import httpx

from ... import __version__
from ...logging_setup import get_logger
from ...utils.politeness import SHARED as POLITENESS
from ..base import Availability, ProviderError
from .base import ImageCandidate, ImageSearchProvider

__all__ = ["WikimediaImageProvider"]

log = get_logger("providers.images.wikimedia")

API = "https://commons.wikimedia.org/w/api.php"

#: Wikimedia answers a burst of searches with 429. Seen for real on a
#: 12-minute narration: the fourth segment's search was refused while the
#: downloader was already pacing itself against the same host. Searches go
#: through the same per-host gate now, and a 429 is retried rather than
#: costing the segment a provider.
RETRY_ON_429 = 3
RETRY_BACKOFF_S = 2.0

# A deliberate judgement call, recorded because it looks like an oversight.
#
# Commons' robots.txt carries `Disallow: /w/`, which covers this endpoint, so
# `POLITENESS.allowed(API)` is False and we do not consult it here. We do
# consult it for every image download, and those hosts (upload.wikimedia.org,
# and the CDNs ddgs returns) allow us.
#
# The reasoning: robots.txt is the Robots Exclusion Protocol, addressed to
# crawlers that discover content by following links, and `Disallow: /w/` is
# there to keep them off expensive dynamic pages. The Action API is a
# documented, publicly supported interface with its own policy, which asks for
# a descriptive User-Agent and a sane request rate rather than a key -- both of
# which we now do. §13 also says "prefer APIs over HTML scraping", and the
# alternative to this call is scraping Commons' HTML, which is worse for
# Wikimedia on every axis.
#
# If you disagree, the fix is to drop this provider, NOT to start scraping.
USER_AGENT = (
    f"VisualResearcher/{__version__} (offline research tool; contact via the repository owner)"
)

_TAGS = re.compile(r"<[^>]+>")


def _plain(value: str) -> str:
    """Commons returns small HTML fragments in extmetadata."""
    return html.unescape(_TAGS.sub("", value or "")).strip()


class WikimediaImageProvider(ImageSearchProvider):
    name = "wikimedia"

    def __init__(self, *, timeout: float = 20.0, min_width: int = 800, **_ignored):
        self.timeout = timeout
        self.min_width = min_width

    def availability(self) -> Availability:
        return Availability.available("public API, no key required")

    def _request(self, params: dict) -> dict:
        """One API call, rate-limited and retried on 429 (§13)."""
        last: Exception | None = None
        for attempt in range(RETRY_ON_429 + 1):
            # Shared per-host gate, so searches and downloads pace together.
            POLITENESS.before_request(API)
            try:
                response = httpx.get(
                    API,
                    params=params,
                    timeout=self.timeout,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"wikimedia search failed: {exc}") from exc

            if response.status_code == 429:
                # Honour Retry-After when the server sets it; it knows better
                # than our backoff does.
                header = response.headers.get("retry-after", "")
                try:
                    wait = float(header)
                except ValueError:
                    wait = RETRY_BACKOFF_S * (2**attempt)
                last = ProviderError("wikimedia returned 429 Too Many Requests")
                # A 429 is the host telling us our rate is wrong, so widen the
                # gate for the rest of the run. Without this we retry into the
                # same wall: the first live run spent 10.6 minutes waiting out
                # 14 separate refusals over 32 segments.
                factor = POLITENESS.rate_limited(API)
                if attempt < RETRY_ON_429:
                    # Name the status code. The old message said only
                    # "rate-limited", so every log filter watching for "429"
                    # -- including the one watching this run -- saw nothing.
                    log.warning(
                        "wikimedia HTTP 429 Too Many Requests; waiting %.1fs "
                        "before retry %d/%d (host interval now %.0fx)",
                        wait,
                        attempt + 1,
                        RETRY_ON_429,
                        factor,
                    )
                    time.sleep(wait)
                    continue
                raise last

            try:
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                raise ProviderError(f"wikimedia search failed: {exc}") from exc
            except ValueError as exc:
                raise ProviderError(f"wikimedia returned non-JSON: {exc}") from exc

        raise ProviderError(f"wikimedia search failed: {last}")

    def search(self, query: str, *, limit: int = 30, **kwargs) -> list[ImageCandidate]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": "6",  # File:
            "gsrlimit": str(min(limit, 50)),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "1280",
        }
        body = self._request(params)

        pages = (body.get("query") or {}).get("pages") or {}
        out: list[ImageCandidate] = []
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            meta = info.get("extmetadata") or {}

            width = int(info.get("width") or 0)
            if width and width < self.min_width:
                continue

            licence = _plain((meta.get("LicenseShortName") or {}).get("value", "")) or "unknown"
            out.append(
                ImageCandidate(
                    image_url=info.get("url", ""),
                    source_page=info.get("descriptionurl", "")
                    or f"https://commons.wikimedia.org/wiki/{page.get('title', '')}",
                    title=str(page.get("title", "")).removeprefix("File:"),
                    width=width,
                    height=int(info.get("height") or 0),
                    creator=_plain((meta.get("Artist") or {}).get("value", "")),
                    license=licence,
                    license_url=_plain((meta.get("LicenseUrl") or {}).get("value", "")),
                    provider=self.name,
                    query=query,
                    thumbnail=info.get("thumburl", ""),
                    extra={"mime": info.get("mime", "")},
                )
            )
        log.debug("wikimedia %r -> %d result(s)", query, len(out))
        return out
