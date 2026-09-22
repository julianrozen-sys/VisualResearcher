"""Fixture-backed video search and sectioned download (CLAUDE.md §2.9).

Like the image fake, this one produces **real files**: it synthesises source
videos with ffmpeg and then trims them with ffmpeg, so an offline run actually
exercises the span arithmetic and leaves a playable mp4 of the right duration.

That matters because §21's clip test is "``--download-sections`` range matches
the located timestamp". A fake that returned a fabricated path would make that
test vacuous; trimming a real file means the assertion can be made against the
duration ffprobe reports.

Each synthetic video carries captions, chapters and a description that mention
its subject at a known time, so the timestamp locator has all four of its
evidence sources to work from (§12.3).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

from ...logging_setup import get_logger
from ...utils.files import atomic_move, ensure_dir
from ..base import Availability
from .base import SectionRequest, SectionResult, VideoHit, VideoProvider

__all__ = ["FakeVideoProvider"]

log = get_logger("providers.video.fake")

#: Length of each synthetic source video, in seconds.
SOURCE_DURATION = 180.0

_CHANNELS = [
    "Old Republic Archives",
    "SWTOR Cutscenes HD",
    "Galactic Lore",
    "Sith Warrior Playthrough",
    "Star Wars Story Time",
]


def _seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


class FakeVideoProvider(VideoProvider):
    name = "fake"
    is_fake = True

    def __init__(self, *, cache_dir: Path | None = None, max_height: int = 1080, **_ignored):
        if cache_dir is None:
            from ...config import install_root

            cache_dir = install_root() / ".cache" / "fake_videos"
        self.cache_dir = Path(cache_dir)
        self.max_height = max_height

    def availability(self) -> Availability:
        if not shutil.which("ffmpeg"):
            return Availability.unavailable(
                "ffmpeg is not on PATH; the fake needs it to synthesise videos",
                "install ffmpeg and put it on PATH",
            )
        return Availability.available("synthesises real videos locally", offline_safe=True)

    # -- source synthesis ---------------------------------------------------

    def _source_path(self, video_id: str) -> Path:
        """Create (once) a synthetic source video for this id."""
        path = self.cache_dir / f"{video_id}.mp4"
        if path.exists():
            return path
        ensure_dir(self.cache_dir)
        staging = path.with_suffix(".partial.mp4")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=640x360:rate=15:duration={SOURCE_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(staging),
        ]
        log.debug("synthesising source video %s", video_id)
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not staging.exists():
            staging.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg could not synthesise a source video: {result.stderr[:300]}")
        atomic_move(staging, path)
        return path

    # -- search -------------------------------------------------------------

    def search(self, query: str, *, limit: int = 5, **kwargs) -> list[VideoHit]:
        seed = _seed_for(query)
        rng = random.Random(seed)
        subject = " ".join(query.split()[:4]) or "the scene"

        hits: list[VideoHit] = []
        for i in range(limit):
            video_id = hashlib.sha256(f"{query}:{i}".encode()).hexdigest()[:11]
            # A known moment this video "contains", placed deterministically.
            moment = round(20.0 + (seed % 100) + i * 13.0, 1)
            moment = min(moment, SOURCE_DURATION - 20.0)

            hits.append(
                VideoHit(
                    video_id=video_id,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    # The first hit reads as an exact-scene match; later ones
                    # get progressively vaguer, so classification has a real
                    # spread to work with rather than one uniform case.
                    title=self._title_for(i, subject),
                    channel=_CHANNELS[rng.randrange(len(_CHANNELS))],
                    duration=SOURCE_DURATION,
                    thumbnail=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                    description=self._description_for(i, subject, moment),
                    query=query,
                    view_count=rng.randrange(1_000, 900_000),
                    upload_date="20190314",
                    captions=self._captions_for(subject, moment),
                    chapters=self._chapters_for(subject, moment),
                    extra={"synthetic": True, "planted_moment": moment},
                )
            )
        return hits

    def _title_for(self, index: int, subject: str) -> str:
        return [
            f"{subject} - full cutscene [1080p]",
            f"{subject} scene",
            f"Let's Play - {subject} and more",
            f"Everything about {subject} explained",
            f"{subject} - reaction and commentary",
        ][index % 5]

    def _description_for(self, index: int, subject: str, moment: float) -> str:
        minutes, seconds = divmod(int(moment), 60)
        if index == 0:
            return (
                f"The complete {subject} sequence.\n\n"
                f"{minutes}:{seconds:02d} {subject}\n"
                f"Recorded at 1080p60."
            )
        if index == 2:
            return f"Full playthrough.\n\n0:00 Intro\n{minutes}:{seconds:02d} {subject}\n"
        return f"A video about {subject}."

    def _captions_for(self, subject: str, moment: float) -> list[dict]:
        lines = []
        cursor = 0.0
        while cursor < SOURCE_DURATION:
            if abs(cursor - moment) < 4.0:
                text = f"and here is {subject}, exactly as it happened"
            else:
                text = "generic narration filler for this part of the video"
            lines.append({"start": round(cursor, 2), "end": round(cursor + 4.0, 2), "text": text})
            cursor += 4.0
        return lines

    def _chapters_for(self, subject: str, moment: float) -> list[dict]:
        return [
            {"start_time": 0.0, "end_time": moment - 10.0, "title": "Introduction"},
            {"start_time": moment - 10.0, "end_time": moment + 30.0, "title": subject},
            {"start_time": moment + 30.0, "end_time": SOURCE_DURATION, "title": "Aftermath"},
        ]

    # -- sectioned download -------------------------------------------------

    def fetch_section(self, request: SectionRequest) -> SectionResult:
        """Trim the synthetic source with ffmpeg -- a real ranged extraction."""
        if request.duration <= 0:
            return SectionResult(False, reason="bad_range", detail="the span has no duration")
        if request.start_s >= SOURCE_DURATION:
            return SectionResult(
                False,
                reason="download_failed",
                detail=f"start {request.start_s}s is past the end of the source",
            )

        try:
            source = self._source_path(request.video_id)
        except Exception as exc:  # noqa: BLE001
            return SectionResult(False, reason="download_failed", detail=str(exc))

        staging = ensure_dir(Path(request.tmp_dir or request.destination.parent / ".staging"))
        temp = staging / f"{request.video_id}.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{request.start_s:.3f}",
            "-to",
            f"{request.end_s:.3f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(temp),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not temp.exists():
            temp.unlink(missing_ok=True)
            return SectionResult(
                False, reason="download_failed", detail=(result.stderr or "")[:300]
            )

        destination = request.destination.with_suffix(".mp4")
        atomic_move(temp, destination)
        return SectionResult(
            True,
            path=destination,
            bytes=destination.stat().st_size,
            duration=request.duration,
        )

    def probe(self, url: str) -> VideoHit | None:
        video_id = url.rsplit("=", 1)[-1]
        return VideoHit(video_id=video_id, url=url, duration=SOURCE_DURATION)


def probe_duration(path: Path) -> float:
    """Actual duration of a media file, via ffprobe.

    Used by the tests to check that a trimmed clip really is as long as the
    located span says it should be.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return 0.0
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 0.0
