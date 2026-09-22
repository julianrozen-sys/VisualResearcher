"""Shared limits across concurrently running jobs (CLAUDE.md §19).

Three of §19's rules are about resources that are shared *between* jobs, which
means a per-job limit is the wrong shape for all three:

* **CPU-heavy stages** (transcription, CLIP ranking) obey one global
  semaphore. Without a GPU they do not parallelise; running two at once on the
  same cores makes both slower and neither finishes sooner.
* **Network stages share a total download budget**, not a per-job one. §19 is
  explicit about why: three jobs each politely rate-limiting themselves still
  triple the request rate a remote host sees, and it is the host's view that
  gets you blocked.
* **The disk guard sums across jobs.** Two jobs that each individually fit on
  the remaining space can still fill the drive together.

These are process-wide, which matches the deployment: one `serve` or one
`worker` process runs the jobs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.disk import DiskSpaceError, usage_for

__all__ = ["ComputeLimiter", "DownloadBudget", "DiskArbiter", "Limits", "get_limits"]

log = get_logger("jobs.limits")


class ComputeLimiter:
    """One global semaphore for CPU-heavy stages (§19)."""

    def __init__(self, max_concurrent: int = 1) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def acquire(self, label: str = "", timeout: float | None = None) -> bool:
        waited = time.monotonic()
        acquired = (
            self._semaphore.acquire(timeout=timeout) if timeout else self._semaphore.acquire()
        )
        if not acquired:
            return False
        with self._lock:
            self._active += 1
        delay = time.monotonic() - waited
        if delay > 0.5:
            log.info("%s waited %.1fs for a CPU slot", label or "stage", delay)
        return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
        try:
            self._semaphore.release()
        except ValueError:  # pragma: no cover - released more than acquired
            log.warning("compute limiter released without a matching acquire")

    class _Slot:
        def __init__(self, limiter: ComputeLimiter, label: str):
            self.limiter, self.label = limiter, label

        def __enter__(self):
            self.limiter.acquire(self.label)
            return self

        def __exit__(self, *exc):
            self.limiter.release()
            return False

    def slot(self, label: str = "") -> _Slot:
        """``with limits.compute.slot("rank"): ...``"""
        return ComputeLimiter._Slot(self, label)


@dataclass
class DownloadBudget:
    """A total download allowance shared by every running job (§19).

    Both a byte budget and a request rate: a remote host cares about how often
    you knock, not about how you divided the knocking between your own jobs.
    """

    max_bytes: int = 0
    min_interval_s: float = 0.0
    _spent: int = field(default=0, init=False)
    _last_request: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> int:
        if not self.max_bytes:
            return 1 << 62
        with self._lock:
            return max(0, self.max_bytes - self._spent)

    def can_spend(self, size: int) -> bool:
        return not self.max_bytes or self.remaining >= size

    def spend(self, size: int) -> None:
        with self._lock:
            self._spent += max(0, size)

    def throttle(self) -> float:
        """Block until the shared minimum interval has passed. Returns the wait."""
        if self.min_interval_s <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._last_request + self.min_interval_s - now)
            self._last_request = now + wait
        if wait:
            time.sleep(wait)
        return wait

    def reset(self) -> None:
        with self._lock:
            self._spent = 0
            self._last_request = 0.0


class DiskArbiter:
    """Checks free space against the **sum** of what active jobs expect (§19).

    A per-job check passes twice and still fills the drive. Each job registers
    what it expects to need; the guard subtracts every active reservation
    before deciding.
    """

    def __init__(self, min_free_gb: float = 5.0) -> None:
        self.min_free_gb = min_free_gb
        self._reserved: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def reserved_gb(self) -> float:
        with self._lock:
            return sum(self._reserved.values())

    def reserve(self, job_id: str, gb: float) -> None:
        with self._lock:
            self._reserved[job_id] = max(0.0, gb)

    def release(self, job_id: str) -> None:
        with self._lock:
            self._reserved.pop(job_id, None)

    def check(self, path: Path, *, job_id: str = "", stage: str = "stage") -> float:
        """Raise ``DiskSpaceError`` when this job plus the others would not fit.

        Returns the headroom in GB once every other active job's reservation
        is accounted for.
        """
        usage = usage_for(path)
        with self._lock:
            others = sum(gb for key, gb in self._reserved.items() if key != job_id)
        effective = usage.free_gb - others

        if effective < self.min_free_gb:
            raise DiskSpaceError(
                f"Refusing to run {stage}: drive {usage.drive} has "
                f"{usage.free_gb:.2f} GB free, but {others:.2f} GB is reserved by "
                f"{len(self._reserved) - (1 if job_id in self._reserved else 0)} "
                f"other running job(s), leaving {effective:.2f} GB against a "
                f"{self.min_free_gb:.2f} GB minimum."
            )
        return effective


@dataclass
class Limits:
    """The three shared limiters, built from settings."""

    compute: ComputeLimiter
    downloads: DownloadBudget
    disk: DiskArbiter

    @classmethod
    def from_settings(cls, settings: Settings) -> Limits:
        return cls(
            compute=ComputeLimiter(settings.compute.max_concurrent_cpu_heavy),
            downloads=DownloadBudget(
                max_bytes=int(
                    settings.clips.max_project_gb * 1024**3 * settings.worker.max_concurrent
                ),
                min_interval_s=0.0,
            ),
            disk=DiskArbiter(settings.output.min_free_gb),
        )


_LIMITS: Limits | None = None
_LIMITS_LOCK = threading.Lock()


def get_limits(settings: Settings, *, refresh: bool = False) -> Limits:
    """The process-wide limits. One set per process, shared by every job."""
    global _LIMITS
    with _LIMITS_LOCK:
        if _LIMITS is None or refresh:
            _LIMITS = Limits.from_settings(settings)
        return _LIMITS
