"""The folder watcher (CLAUDE.md §18) — the automation trigger.

VoiceCleaner drops a `.wav` into a folder. No HTTP call, no JSON, no
cooperation from its side at all. That constraint drives every decision here:

* **the file may still be being written.** A drop is not an event, it is the
  *start* of one, so nothing is touched until its size has been unchanged for
  ``watch.stable_secs``. Reading a half-written WAV produces a transcript of
  half a narration, which is worse than waiting;
* **the same file must never be processed twice**, across restarts. The ledger
  is a unique index in SQLite, claimed *before* any work begins, so a crash
  mid-ingest cannot cause a re-run;
* **the loop must not die.** One unreadable file, one permissions error, one
  malformed name — logged, source left untouched, keep watching. A watcher
  that exits on the first bad file is worse than no watcher, because the user
  believes it is running.

``watchdog`` is used when available and polling is the fallback (§18.1). The
polling path is not dead code: it runs on network shares where inotify-style
events are unreliable, which is exactly where a drop folder tends to live.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.files import slugify

__all__ = ["FolderWatcher", "matches_prefix", "derive_project_name", "WatchEvent"]

log = get_logger("jobs.watcher")

#: How often the stability guard re-checks a file's size (§18.3).
STABILITY_POLL_S = 1.0


@dataclass
class WatchEvent:
    """One accepted drop, after it passed every guard."""

    source: Path
    project: str
    job_id: str = ""
    size: int = 0


def matches_prefix(name: str, prefix: str) -> bool:
    """§18.2: audio files starting with the prefix, case-insensitively.

    Every format :mod:`~..pipeline.ingest` can read is accepted, not just
    ``.wav``. Restricting the drop folder to one extension meant dropping an
    mp3 did nothing at all -- no error, no project, no log line -- while the
    same file passed straight through `visualresearch run`.
    """
    from ..pipeline.ingest import AUDIO_SUFFIXES

    lowered = name.lower()
    if not any(lowered.endswith(suffix) for suffix in AUDIO_SUFFIXES):
        return False
    return lowered.startswith(prefix.lower()) if prefix else True


def derive_project_name(source: Path, projects_dir: Path, *, slug_max_len: int = 40) -> str:
    """§18.4: slugified stem, timestamp suffix on collision."""
    base = slugify(source.stem, max_len=slug_max_len)
    if not (projects_dir / base).exists():
        return base
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class FolderWatcher:
    """Watches a folder and enqueues jobs for the files it accepts.

    Enqueue-only by design: it never runs the pipeline itself. The worker does
    that, on the same queue the CLI uses, so a watched job and a hand-started
    job take an identical path downstream (§18.5).
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Path,
        *,
        on_accept: Callable[[WatchEvent], None] | None = None,
        use_watchdog: bool = True,
    ) -> None:
        self.settings = settings
        self.db_path = db_path
        self.on_accept = on_accept
        self.use_watchdog = use_watchdog
        self.watch_dir = Path(settings.watch.dir)
        self._stop = threading.Event()
        self._observer = None
        self._thread: threading.Thread | None = None
        #: Files seen but not yet stable, so a log line is not repeated forever.
        self._pending: set[str] = set()
        self.accepted: list[WatchEvent] = []

    # -- guards ------------------------------------------------------------

    def is_stable(self, path: Path, *, timeout: float | None = None) -> bool:
        """§18.3: True once the file's size has stopped changing.

        VoiceCleaner may still be writing. Polls every second and requires the
        size to hold for ``watch.stable_secs`` before accepting.
        """
        required = self.settings.watch.stable_secs
        deadline = time.monotonic() + (timeout if timeout is not None else required * 10 + 5)
        last_size = -1
        steady_since = 0.0

        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                size = path.stat().st_size
            except OSError as exc:
                # Moved or deleted while we watched: not our file any more.
                log.debug("stability check: %s is unreadable (%s)", path.name, exc)
                return False

            now = time.monotonic()
            if size != last_size or size == 0:
                last_size, steady_since = size, now
            elif now - steady_since >= required:
                return True
            self._stop.wait(STABILITY_POLL_S)

        log.warning("%s never settled; leaving it for the next pass", path.name)
        return False

    # -- one file ----------------------------------------------------------

    def handle(self, path: Path) -> WatchEvent | None:
        """Consider one file. Returns the event when it was accepted.

        Every failure path here leaves the source file untouched (§18.8), so a
        file this pass could not take is simply reconsidered next pass.
        """
        from .queue import claim_filename, discard_filename, enqueue, release_filename

        try:
            if not matches_prefix(path.name, self.settings.watch.filename_prefix):
                return None
            if not path.is_file():
                return None

            if path.name.lower() in self._pending:
                return None
            self._pending.add(path.name.lower())
            try:
                if not self.is_stable(path):
                    return None
            finally:
                self._pending.discard(path.name.lower())

            size = path.stat().st_size
            if size == 0:
                log.warning("ignoring %s: zero bytes", path.name)
                return None

            # §18.6: claim BEFORE any work, so a crash mid-ingest cannot
            # cause a second run. The unique index is what makes this safe.
            if not claim_filename(self.db_path, path.name, source_path=path, size=size):
                log.debug("%s was already processed; ignoring", path.name)
                return None

            project = derive_project_name(
                path,
                self.settings.projects_dir,
                slug_max_len=self.settings.output.slug_max_len,
            )
            destination = self.settings.project_dir(project) / "input" / "narration.wav"

            try:
                # §18.5: MOVE, so the drop folder does not fill up and a
                # restarted watcher does not see the same file again.
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._move(path, destination)
            except Exception as exc:  # noqa: BLE001
                # The move failed and no work was started, so the claim is
                # discarded outright rather than marked failed. Marking it
                # failed would leave the row in place and blacklist the file
                # forever over a transient error.
                log.error("could not move %s: %s", path.name, exc)
                discard_filename(self.db_path, path.name)
                return None

            job = enqueue(
                self.db_path,
                project=project,
                source_path=destination,
                origin="watcher",
                origin_filename=path.name,
            )
            release_filename(
                self.db_path, path.name, job_id=job.id, project=project, status="queued"
            )

            event = WatchEvent(source=path, project=project, job_id=job.id, size=size)
            self.accepted.append(event)
            log.info(
                "accepted %s -> project %r (job %s, %.1f MB)",
                path.name,
                project,
                job.id,
                size / 1_048_576,
            )
            if self.on_accept:
                try:
                    self.on_accept(event)
                except Exception as exc:  # noqa: BLE001 - a callback must not kill the loop
                    log.error("watcher callback failed: %s", exc)
            return event

        except Exception as exc:  # noqa: BLE001 - §18.8: never crash the loop
            log.exception("watcher: skipping %s after an unexpected error: %s", path, exc)
            return None

    @staticmethod
    def _move(source: Path, destination: Path) -> None:
        import shutil

        try:
            os.replace(source, destination)
        except OSError:
            # Different volume: copy then remove, and only remove once the
            # copy is verifiably the same size.
            shutil.copy2(source, destination)
            if destination.stat().st_size != source.stat().st_size:
                destination.unlink(missing_ok=True)
                raise
            source.unlink()

    # -- sweeps ------------------------------------------------------------

    def scan_once(self) -> list[WatchEvent]:
        """Consider everything currently in the folder."""
        if not self.watch_dir.exists():
            log.warning("watch directory does not exist: %s", self.watch_dir)
            return []
        out: list[WatchEvent] = []
        try:
            entries = sorted(self.watch_dir.iterdir())
        except OSError as exc:
            log.error("cannot list %s: %s", self.watch_dir, exc)
            return []
        for entry in entries:
            event = self.handle(entry)
            if event is not None:
                out.append(event)
        return out

    # -- running -----------------------------------------------------------

    def run(self, *, once: bool = False) -> None:
        """Block, watching until :meth:`stop` is called."""
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "watching %s for %s*.wav (stable for %.0fs before processing)",
            self.watch_dir,
            self.settings.watch.filename_prefix,
            self.settings.watch.stable_secs,
        )

        # An existing backlog is processed first: files dropped while the
        # watcher was down still need picking up.
        self.scan_once()
        if once:
            return

        if self.use_watchdog and self._start_watchdog():
            while not self._stop.is_set():
                # A periodic sweep alongside the event stream. Events get
                # missed on network shares, and a missed drop is invisible --
                # the user just waits for a project that never appears.
                self._stop.wait(self.settings.watch.poll_interval_s)
                if not self._stop.is_set():
                    self.scan_once()
            self._stop_watchdog()
            return

        log.info(
            "watchdog unavailable; polling every %.0fs",
            self.settings.watch.poll_interval_s,
        )
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.settings.watch.poll_interval_s)

    def _start_watchdog(self) -> bool:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as exc:
            log.info("watchdog is not installed (%s); falling back to polling", exc)
            return False

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):  # pragma: no cover - driven by the OS
                if not event.is_directory:
                    watcher.handle(Path(event.src_path))

            def on_moved(self, event):  # pragma: no cover
                dest = getattr(event, "dest_path", None)
                if dest and not event.is_directory:
                    watcher.handle(Path(dest))

        try:
            self._observer = Observer()
            self._observer.schedule(_Handler(), str(self.watch_dir), recursive=False)
            self._observer.start()
            log.info("watchdog observer started")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("could not start watchdog (%s); falling back to polling", exc)
            self._observer = None
            return False

    def _stop_watchdog(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as exc:  # noqa: BLE001
                log.debug("observer shutdown: %s", exc)
            self._observer = None

    def start_background(self) -> threading.Thread:
        """Run in a daemon thread. This is how ``serve`` hosts it (§18.7)."""
        self._thread = threading.Thread(target=self.run, name="vr-watcher", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        self._stop_watchdog()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
