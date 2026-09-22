"""yt-dlp video search and sectioned download (CLAUDE.md §12).

Driven as a subprocess rather than through the Python API. Two reasons: the
command in §12.4 is specified literally, and this way the exact arguments used
are loggable, testable and reproducible by hand when a download misbehaves.

The section download is the whole point of this module. Fetching a 40-minute
video to keep eight seconds of it is the failure mode §12 is written to
prevent, so the full-download fallback is not implemented here at all -- the
pipeline decides whether it is permitted and calls back in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ...logging_setup import get_logger
from ...utils.files import atomic_move, ensure_dir
from ...utils.timefmt import yt_dlp_stamp
from ..base import Availability, ProviderError
from .base import SectionRequest, SectionResult, VideoHit, VideoProvider

__all__ = ["YtDlpVideoProvider", "build_section_command", "build_search_command"]

log = get_logger("providers.video.ytdlp")

#: §12.4's format selector, verbatim apart from the configurable height.
FORMAT_TEMPLATE = "bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}]"

SEARCH_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 600


def build_search_command(query: str, limit: int) -> list[str]:
    """``ytsearchN:query`` -- no key, no quota (§12.1)."""
    return [
        "yt-dlp",
        f"ytsearch{limit}:{query}",
        "--dump-single-json",
        "--no-warnings",
        "--no-playlist",
        "--skip-download",
        "--ignore-errors",
    ]


def build_section_command(request: SectionRequest, output_template: str) -> list[str]:
    """The §12.4 command.

    §12.4 writes the range as ``*00:01:42:13-00:01:43:16``. yt-dlp's
    ``--download-sections`` takes ``*START-END`` where the times are
    ``HH:MM:SS`` with an optional decimal fraction, so the sub-second part is
    written with a dot. That is the same instant, in the syntax the tool
    actually parses.
    """
    section = f"*{yt_dlp_stamp(request.start_s)}-{yt_dlp_stamp(request.end_s)}"
    return [
        "yt-dlp",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-f",
        FORMAT_TEMPLATE.format(h=request.max_height),
        "-o",
        output_template,
        "--no-warnings",
        "--no-playlist",
        "--no-part",
        request.url,
    ]


class YtDlpVideoProvider(VideoProvider):
    name = "ytdlp"

    def __init__(self, *, max_height: int = 1080, rate_limit: str = "", **_ignored):
        self.max_height = max_height
        self.rate_limit = rate_limit

    def availability(self) -> Availability:
        found = shutil.which("yt-dlp")
        if not found:
            return Availability.unavailable(
                "yt-dlp is not on PATH", "uv pip install yt-dlp", "or: winget install yt-dlp"
            )
        if not shutil.which("ffmpeg"):
            return Availability.unavailable(
                "ffmpeg is not on PATH; sectioned downloads need it",
                "install ffmpeg and put it on PATH",
            )
        return Availability.available(f"yt-dlp at {found}")

    # -- search ------------------------------------------------------------

    def search(self, query: str, *, limit: int = 5, **kwargs) -> list[VideoHit]:
        command = build_search_command(query, limit)
        log.debug("yt-dlp search: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=SEARCH_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise ProviderError(f"yt-dlp is not installed: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"yt-dlp search timed out after {SEARCH_TIMEOUT}s") from exc

        if completed.returncode != 0 and not completed.stdout.strip():
            raise ProviderError(
                f"yt-dlp search failed ({completed.returncode}): "
                f"{(completed.stderr or '').strip()[:300]}"
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"yt-dlp returned unparseable JSON: {exc}") from exc

        entries = payload.get("entries") or []
        return [self._to_hit(entry, query) for entry in entries if entry]

    def _to_hit(self, entry: dict, query: str) -> VideoHit:
        video_id = str(entry.get("id", ""))
        return VideoHit(
            video_id=video_id,
            url=entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
            title=entry.get("title", "") or "",
            channel=entry.get("channel") or entry.get("uploader", "") or "",
            duration=float(entry.get("duration") or 0.0),
            thumbnail=entry.get("thumbnail", "") or "",
            description=entry.get("description", "") or "",
            query=query,
            view_count=int(entry.get("view_count") or 0),
            upload_date=str(entry.get("upload_date") or ""),
            chapters=list(entry.get("chapters") or []),
            extra={"live": bool(entry.get("is_live"))},
        )

    # -- sectioned download -------------------------------------------------

    def fetch_section(self, request: SectionRequest) -> SectionResult:
        """Fetch just the requested span (§12.4).

        Downloads into ``.tmp/`` and moves into place on success, so a partial
        file is never visible in ``clips/`` (§12.6).
        """
        if request.duration <= 0:
            return SectionResult(False, reason="bad_range", detail="the span has no duration")

        staging = ensure_dir(Path(request.tmp_dir or request.destination.parent / ".staging"))
        template = str(staging / f"{request.video_id}.%(ext)s")
        command = build_section_command(request, template)
        if self.rate_limit:
            command.extend(["--limit-rate", self.rate_limit])

        log.info(
            "yt-dlp: fetching %.2f-%.2fs of %s", request.start_s, request.end_s, request.video_id
        )
        log.debug("yt-dlp command: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            return SectionResult(False, reason="not_installed", detail=str(exc))
        except subprocess.TimeoutExpired:
            return SectionResult(False, reason="timeout", detail=f"exceeded {DOWNLOAD_TIMEOUT}s")

        produced = sorted(staging.glob(f"{request.video_id}.*"))
        if completed.returncode != 0 or not produced:
            for leftover in produced:
                leftover.unlink(missing_ok=True)
            return SectionResult(
                False,
                reason="download_failed",
                detail=(completed.stderr or completed.stdout or "").strip()[:300],
            )

        source = produced[0]
        destination = request.destination.with_suffix(source.suffix)
        atomic_move(source, destination)
        for leftover in produced[1:]:
            leftover.unlink(missing_ok=True)

        return SectionResult(
            True,
            path=destination,
            bytes=destination.stat().st_size,
            duration=request.duration,
        )

    # -- probe --------------------------------------------------------------

    def probe(self, url: str) -> VideoHit | None:
        try:
            completed = subprocess.run(
                ["yt-dlp", url, "--dump-single-json", "--skip-download", "--no-warnings"],
                capture_output=True,
                text=True,
                timeout=SEARCH_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode != 0:
                return None
            return self._to_hit(json.loads(completed.stdout), query="")
        except Exception as exc:  # noqa: BLE001
            log.debug("probe failed for %s: %s", url, exc)
            return None
