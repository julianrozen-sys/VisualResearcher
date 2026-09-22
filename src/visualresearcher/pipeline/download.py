"""Stage 8: image download and validation (CLAUDE.md §11).

Every rule here exists because a remote server is not trustworthy:

* **15 MB maximum**, enforced *while streaming*, not after. Checking
  ``Content-Length`` alone lets a server lie and fill the disk;
* **content-type allowlist**, with the real bytes checked afterwards, because a
  content-type header is just a claim;
* **Pillow-decodable**, verified by actually decoding;
* **SVG rejected** outright -- it is script-capable markup, not a raster image;
* **filenames sanitised and path traversal blocked** -- the remote filename is
  never trusted, and never used;
* files are written to ``.tmp/`` and moved into place on success, so a partial
  download is never visible in ``images/``.

Disk headroom is checked before the stage starts (§3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .. import __version__
from ..config import Settings
from ..logging_setup import get_logger
from ..providers.images.base import ImageCandidate
from ..schemas import ImageRecord
from ..utils.disk import check_free_space
from ..utils.files import atomic_move, ensure_dir, is_within, sanitize_filename
from ..utils.hashing import sha256_bytes
from ..utils.politeness import SHARED as POLITENESS_DEFAULT

__all__ = [
    "download_candidate",
    "download_candidates",
    "validate_image_bytes",
    "RejectReason",
    "DownloadOutcome",
    "ALLOWED_CONTENT_TYPES",
    "ALLOWED_EXTENSIONS",
    "POLITENESS",
]

log = get_logger("pipeline.download")

USER_AGENT = f"VisualResearcher/{__version__} (research tool)"

#: §13: respect robots.txt and rate limits. The shared instance, so a
#: download and a provider search count against the same per-host budget.
POLITENESS = POLITENESS_DEFAULT

#: Raster formats only. SVG is deliberately absent (§11).
ALLOWED_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }
)

ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"})

#: Pillow format name -> the extension we save under.
_FORMAT_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "MPO": ".jpg",
}


class RejectReason:
    TOO_LARGE = "too_large"
    DISALLOWED = "robots_disallowed"
    TOO_SMALL = "too_small"
    BAD_CONTENT_TYPE = "bad_content_type"
    UNDECODABLE = "undecodable"
    SVG = "svg"
    EMPTY = "empty"
    TRANSPORT = "transport"
    PATH_ESCAPE = "path_escape"
    DUPLICATE_URL = "duplicate_url"


@dataclass
class DownloadOutcome:
    """What happened to one candidate. A rejection is never silent (§11)."""

    ok: bool
    record: ImageRecord | None = None
    reason: str = ""
    detail: str = ""
    candidate: ImageCandidate | None = None


def _looks_like_svg(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith(b"<?xml") and b"<svg" in data[:2048].lower() or head.startswith(b"<svg")


def validate_image_bytes(
    data: bytes, settings: Settings
) -> tuple[bool, str, str, tuple[int, int], str]:
    """Check downloaded bytes against every §11 rule.

    Returns ``(ok, reason, detail, (width, height), extension)``.
    """
    if not data:
        return False, RejectReason.EMPTY, "zero bytes", (0, 0), ""

    if len(data) > settings.images.max_bytes:
        return (
            False,
            RejectReason.TOO_LARGE,
            f"{len(data)} bytes exceeds the {settings.images.max_bytes} byte limit",
            (0, 0),
            "",
        )

    if _looks_like_svg(data):
        return False, RejectReason.SVG, "SVG is not accepted", (0, 0), ""

    try:
        import io

        from PIL import Image

        # verify() checks structure but leaves the object unusable, so the
        # image is opened twice: once to verify, once to read the size.
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            fmt = (image.format or "").upper()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a rejection
        return False, RejectReason.UNDECODABLE, f"Pillow could not decode it: {exc}", (0, 0), ""

    if fmt == "SVG":
        return False, RejectReason.SVG, "SVG is not accepted", (width, height), ""

    extension = _FORMAT_EXT.get(fmt, "")
    if not extension:
        return (
            False,
            RejectReason.BAD_CONTENT_TYPE,
            f"format {fmt or 'unknown'} is not in the allowlist",
            (width, height),
            "",
        )

    if width < settings.images.min_width:
        return (
            False,
            RejectReason.TOO_SMALL,
            f"{width}px wide, below the {settings.images.min_width}px minimum",
            (width, height),
            extension,
        )

    return True, "", "", (width, height), extension


def _fetch(url: str, *, max_bytes: int, timeout: float) -> tuple[bytes, str]:
    """Read a URL into memory, stopping the moment it exceeds ``max_bytes``.

    Supports ``file://`` so the offline fake exercises this same path.
    """
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path.lstrip("/")))
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"{size} bytes exceeds the {max_bytes} byte limit")
        return path.read_bytes(), ""

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {parsed.scheme!r}")

    with httpx.stream(
        "GET",
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        content_type = (
            (response.headers.get("content-type", "") or "").split(";")[0].strip().lower()
        )

        # A declared length over the limit is refused before reading a byte.
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise ValueError(f"declared {declared} bytes, over the {max_bytes} byte limit")

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes(64 * 1024):
            total += len(chunk)
            # Enforced while streaming: a lying Content-Length must not win.
            if total > max_bytes:
                raise ValueError(f"stream exceeded the {max_bytes} byte limit")
            chunks.append(chunk)
        return b"".join(chunks), content_type


def download_candidate(
    candidate: ImageCandidate,
    target_dir: Path,
    settings: Settings,
    *,
    index: int,
    segment_index: int,
    tmp_dir: Path | None = None,
    timeout: float = 30.0,
) -> DownloadOutcome:
    """Download and validate one candidate.

    The saved filename is built from the segment and a counter -- **never** from
    the remote filename or URL (§11).
    """
    max_bytes = settings.images.max_bytes

    # A content-type claim is checked early as a cheap filter, but the real
    # decision is made on the bytes.
    declared_ext = Path(urlparse(candidate.image_url).path).suffix.lower()
    if declared_ext == ".svg":
        return DownloadOutcome(
            False, reason=RejectReason.SVG, detail="URL ends in .svg", candidate=candidate
        )

    # §13: respect robots.txt and rate limits. Skipped for file:// URLs,
    # so the offline fake is neither consulted nor throttled.
    if not POLITENESS.allowed(candidate.image_url):
        return DownloadOutcome(
            False,
            reason=RejectReason.DISALLOWED,
            detail="robots.txt disallows fetching this URL",
            candidate=candidate,
        )
    POLITENESS.before_request(candidate.image_url)

    try:
        data, content_type = _fetch(candidate.image_url, max_bytes=max_bytes, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - one bad URL never stops a segment
        return DownloadOutcome(
            False, reason=RejectReason.TRANSPORT, detail=str(exc), candidate=candidate
        )

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return DownloadOutcome(
            False,
            reason=RejectReason.BAD_CONTENT_TYPE,
            detail=f"server said {content_type!r}",
            candidate=candidate,
        )

    ok, reason, detail, (width, height), extension = validate_image_bytes(data, settings)
    if not ok:
        return DownloadOutcome(False, reason=reason, detail=detail, candidate=candidate)

    # Our own name, derived from our own counters. The remote name is discarded.
    filename = sanitize_filename(f"{segment_index:03d}_{index:02d}{extension}")
    destination = target_dir / filename
    if not is_within(target_dir, destination):
        return DownloadOutcome(
            False,
            reason=RejectReason.PATH_ESCAPE,
            detail=f"refusing to write outside {target_dir}",
            candidate=candidate,
        )

    # Write to .tmp/ and move on success, so images/ never holds a partial file.
    staging = ensure_dir(tmp_dir or settings.tmp_dir) / f"{filename}.{id(candidate):x}.part"
    staging.write_bytes(data)
    ensure_dir(target_dir)
    atomic_move(staging, destination)

    record = ImageRecord(
        local_path=str(destination),
        image_url=candidate.image_url,
        source_page=candidate.source_page,
        domain=candidate.domain,
        provider=candidate.provider,
        query=candidate.query,
        query_kind=candidate.query_kind,
        width=width,
        height=height,
        bytes=len(data),
        sha256=sha256_bytes(data),
        segment_index=segment_index,
        creator=candidate.creator,
        license=candidate.license or "unknown",
        license_url=candidate.license_url,
        status="downloaded",
        notes=str(candidate.extra.get("note", "")),
    )
    return DownloadOutcome(True, record=record, candidate=candidate)


def download_candidates(
    candidates: list[ImageCandidate],
    target_dir: Path,
    settings: Settings,
    *,
    segment_index: int,
    limit: int | None = None,
    rate_limit_s: float = 0.0,
) -> tuple[list[ImageRecord], list[DownloadOutcome]]:
    """Download a segment's candidates. Returns ``(records, rejections)``.

    Rejections are returned, not swallowed: §11 requires discards to be
    recorded rather than dropped silently.
    """
    check_free_space(target_dir, settings.output.min_free_gb, stage="image download")

    records: list[ImageRecord] = []
    rejected: list[DownloadOutcome] = []
    seen_urls: set[str] = set()
    counter = 0

    for candidate in candidates:
        if limit is not None and len(records) >= limit:
            break

        key = candidate.dedupe_key()
        if key in seen_urls:
            rejected.append(
                DownloadOutcome(
                    False,
                    reason=RejectReason.DUPLICATE_URL,
                    detail="the same URL was already fetched for this segment",
                    candidate=candidate,
                )
            )
            continue
        seen_urls.add(key)

        outcome = download_candidate(
            candidate,
            target_dir,
            settings,
            index=counter,
            segment_index=segment_index,
        )
        counter += 1
        if outcome.ok and outcome.record:
            records.append(outcome.record)
        else:
            rejected.append(outcome)
            log.debug(
                "rejected %s (%s): %s", candidate.image_url[:80], outcome.reason, outcome.detail
            )

        if rate_limit_s:
            time.sleep(rate_limit_s)

    by_reason: dict[str, int] = {}
    for outcome in rejected:
        by_reason[outcome.reason] = by_reason.get(outcome.reason, 0) + 1
    log.info(
        "segment %03d: downloaded %d, rejected %d%s",
        segment_index,
        len(records),
        len(rejected),
        " (" + ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())) + ")"
        if rejected
        else "",
    )
    return records, rejected
