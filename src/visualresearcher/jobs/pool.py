"""The worker pool (CLAUDE.md §19).

Runs up to ``worker.max_concurrent`` jobs at once, each in its own thread,
all sharing the limits in :mod:`.limits`.

Threads rather than processes, deliberately. The work here is dominated by
subprocesses (ffmpeg, yt-dlp), network I/O, and torch — all of which release
the GIL — while the parts that do not are already serialised behind the
compute semaphore. Processes would buy nothing and cost a second SQLite
connection per job plus the model loaded twice.

Each job writes only inside its own project folder, so two jobs cannot write
into each other's output (§21). That is a property of the layout rather than
of a lock: paths are derived from the project name, and the name is unique.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from .limits import Limits, get_limits
from .queue import claim_next, get_job, set_state
from .states import JobState
from .worker import RunContext, StageResult, run_pipeline, worker_id

__all__ = ["WorkerPool", "JobOutcome", "run_one_job"]

log = get_logger("jobs.pool")

#: How long an idle worker waits before asking the queue again.
IDLE_SLEEP_S = 2.0

#: Rough per-job disk reservation, used by the cross-job guard (§19). Deliberately
#: generous: under-reserving is what lets two jobs fill a drive together.
DEFAULT_RESERVE_GB = 2.0


@dataclass
class JobOutcome:
    job_id: str
    project: str
    state: str = ""
    results: list[StageResult] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and all(r.ok for r in self.results)


def run_one_job(
    job,
    settings: Settings,
    db_path: Path,
    *,
    limits: Limits | None = None,
    on_finish: Callable[[JobOutcome], None] | None = None,
) -> JobOutcome:
    """Run one claimed job to completion. Never raises."""
    limits = limits or get_limits(settings)
    started = time.perf_counter()
    project_root = settings.project_dir(job.project)

    from ..logging_setup import add_job_log

    handler = add_job_log(project_root / "job.log")
    limits.disk.reserve(job.id, DEFAULT_RESERVE_GB)

    outcome = JobOutcome(job_id=job.id, project=job.project)
    try:
        ctx = RunContext(
            job_id=job.id,
            project=job.project,
            project_root=project_root,
            source_path=Path(job.source_path),
            settings=settings,
            db_path=db_path,
            force=job.force,
            no_clips=job.no_clips,
            limits=limits,
        )
        outcome.results = run_pipeline(ctx)
        outcome.state = str(get_job(db_path, job.id).state)
    except Exception as exc:  # noqa: BLE001 - a pool worker must never die
        log.exception("job %s crashed", job.id)
        outcome.error = f"{type(exc).__name__}: {exc}"
        try:
            set_state(db_path, job.id, JobState.FAILED, error=outcome.error)
            outcome.state = str(JobState.FAILED)
        except Exception:  # noqa: BLE001
            pass
    finally:
        limits.disk.release(job.id)
        outcome.duration_s = time.perf_counter() - started
        logger = log.getChild("run")
        logger.parent.removeHandler(handler) if handler in logger.parent.handlers else None
        handler.close()

    if on_finish:
        try:
            on_finish(outcome)
        except Exception as exc:  # noqa: BLE001
            log.error("job callback failed: %s", exc)
    return outcome


class WorkerPool:
    """Claims and runs queued jobs, ``worker.max_concurrent`` at a time."""

    def __init__(
        self,
        settings: Settings,
        db_path: Path,
        *,
        on_finish: Callable[[JobOutcome], None] | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = db_path
        self.on_finish = on_finish
        self.limits = get_limits(settings, refresh=True)
        self.max_concurrent = max(1, settings.worker.max_concurrent)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active: dict[str, str] = {}
        self._lock = threading.Lock()
        self.completed: list[JobOutcome] = []

    @property
    def active(self) -> dict[str, str]:
        with self._lock:
            return dict(self._active)

    def _worker_loop(self, index: int) -> None:
        name = f"{worker_id()}-{index}"
        log.debug("worker %s started", name)
        while not self._stop.is_set():
            try:
                job = claim_next(self.db_path, worker_id=name)
            except Exception as exc:  # noqa: BLE001 - a locked DB must not kill the worker
                log.error("worker %s could not claim: %s", name, exc)
                self._stop.wait(IDLE_SLEEP_S)
                continue

            if job is None:
                self._stop.wait(IDLE_SLEEP_S)
                continue

            with self._lock:
                self._active[job.id] = job.project
            log.info("worker %s running job %s (%s)", name, job.id, job.project)
            try:
                outcome = run_one_job(
                    job,
                    self.settings,
                    self.db_path,
                    limits=self.limits,
                    on_finish=self.on_finish,
                )
                with self._lock:
                    self.completed.append(outcome)
            finally:
                with self._lock:
                    self._active.pop(job.id, None)
        log.debug("worker %s stopped", name)

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(
                target=self._worker_loop, args=(i,), name=f"vr-worker-{i}", daemon=True
            )
            for i in range(self.max_concurrent)
        ]
        for thread in self._threads:
            thread.start()
        log.info(
            "worker pool started: %d concurrent job(s), %d CPU-heavy slot(s)",
            self.max_concurrent,
            self.limits.compute.max_concurrent,
        )

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            log.info("interrupted; finishing the current jobs")
        finally:
            self.stop()

    def drain(self, timeout: float = 600.0) -> list[JobOutcome]:
        """Run until the queue is empty and nothing is in flight. For tests."""
        from .queue import list_jobs

        self.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            queued = [j for j in list_jobs(self.db_path, limit=500) if j.state == JobState.QUEUED]
            if not queued and not self.active:
                break
            time.sleep(0.2)
        self.stop()
        return list(self.completed)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=30)
        self._threads = []
