"""Stage 1: ingest (CLAUDE.md §8).

Takes the narration file the user pointed at and establishes the project
folder. Two rules govern this stage:

* never destroy user data (§2.8) — the source is copied, not moved, when it
  came from the CLI; only the watcher moves (§18.5), and it moves a file it
  was handed on purpose;
* nothing starts without disk headroom (§3).
"""

from __future__ import annotations

import shutil
import wave
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..utils.disk import check_free_space
from ..utils.files import backup_file, slugify
from ..utils.timefmt import timecode

__all__ = ["IngestResult", "ingest", "derive_project_name", "probe_audio"]

log = get_logger("pipeline.ingest")

#: Extensions we will accept as narration input.
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


class IngestResult:
    def __init__(self, paths: ProjectPaths, audio: Path, duration_s: float, moved: bool):
        self.paths = paths
        self.audio = audio
        self.duration_s = duration_s
        self.moved = moved

    def __repr__(self) -> str:  # pragma: no cover
        return f"<IngestResult project={self.paths.root.name!r} duration={self.duration_s:.1f}s>"


def derive_project_name(source: Path, projects_dir: Path, *, slug_max_len: int = 40) -> str:
    """Slugified stem, with a timestamp suffix on collision (§18.4)."""
    base = slugify(Path(source).stem, max_len=slug_max_len)
    candidate = projects_dir / base
    if not candidate.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}"


def probe_audio(path: Path) -> tuple[float, dict]:
    """Duration and basic WAV properties, without shelling out to ffprobe.

    Returns ``(0.0, {})`` for formats ``wave`` cannot open; a downstream
    transcriber will still handle them, we just cannot report on them here.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            info = {
                "channels": handle.getnchannels(),
                "sample_rate": rate,
                "sample_width_bytes": handle.getsampwidth(),
                "frames": frames,
            }
            return (frames / float(rate) if rate else 0.0), info
    except Exception as exc:  # noqa: BLE001
        log.debug("probe_audio: %s is not a readable wav (%s)", path.name, exc)
        return 0.0, {}


def ingest(
    source: Path,
    project_root: Path,
    settings: Settings,
    *,
    move: bool = False,
    force: bool = False,
) -> IngestResult:
    """Create the project tree and place the narration at ``input/narration.wav``.

    Args:
        source: the user's narration file.
        project_root: ``projects/<name>``.
        settings: for the disk floor.
        move: move rather than copy. Only the watcher sets this (§18.5).
        force: re-ingest over an existing narration, keeping a ``.bak``.

    Raises:
        FileNotFoundError: no such source.
        ValueError: unsupported extension.
        DiskSpaceError: below ``output.min_free_gb``.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"narration file not found: {source}")
    if source.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(
            f"unsupported audio format {source.suffix!r}; expected one of "
            f"{', '.join(sorted(AUDIO_SUFFIXES))}"
        )

    usage = check_free_space(project_root, settings.output.min_free_gb, stage="ingest")
    log.info("disk %s: %.1f GB free", usage.drive, usage.free_gb)

    paths = ProjectPaths(project_root)
    paths.ensure_base()
    target = paths.narration_wav

    if target.exists() and not force:
        duration, _ = probe_audio(target)
        log.info("narration already ingested at %s (%s)", target, timecode(duration))
        return IngestResult(paths, target, duration, moved=False)

    if target.exists():
        backup = backup_file(target)
        log.info("force: backed up existing narration to %s", backup.name if backup else "-")

    if move:
        shutil.move(str(source), str(target))
        log.info("moved %s -> %s", source.name, target)
    else:
        shutil.copy2(source, target)
        log.info("copied %s -> %s", source.name, target)

    duration, info = probe_audio(target)
    if duration:
        log.info(
            "narration: %s (%d Hz, %d ch)",
            timecode(duration),
            info.get("sample_rate", 0),
            info.get("channels", 0),
        )
    else:
        log.warning("could not determine duration of %s", target.name)
    return IngestResult(paths, target, duration, moved=move)
