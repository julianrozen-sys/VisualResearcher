"""Disk space guard (CLAUDE.md §3).

C: on the target machine runs at a few hundred megabytes free, so "is there
room" is not a theoretical question here. Every bulk stage asks before it
starts, and the error message always names the drive.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DiskUsage", "usage_for", "free_gb", "DiskSpaceError", "check_free_space", "drive_of"]

BYTES_PER_GB = 1024**3


class DiskSpaceError(RuntimeError):
    """Raised when a stage would run with too little headroom to finish."""


@dataclass(frozen=True)
class DiskUsage:
    drive: str
    total_gb: float
    used_gb: float
    free_gb: float

    @property
    def percent_free(self) -> float:
        return 100.0 * self.free_gb / self.total_gb if self.total_gb else 0.0


def drive_of(path: Path) -> str:
    r"""``D:\dev\x`` -> ``D:``. Falls back to the anchor on POSIX."""
    drive = Path(path).resolve().drive
    return drive or Path(path).resolve().anchor or "/"


def usage_for(path: Path) -> DiskUsage:
    """Disk usage for the volume holding ``path``, walking up to an existing parent."""
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    total, used, free = shutil.disk_usage(probe)
    return DiskUsage(
        drive=drive_of(probe),
        total_gb=total / BYTES_PER_GB,
        used_gb=used / BYTES_PER_GB,
        free_gb=free / BYTES_PER_GB,
    )


def free_gb(path: Path) -> float:
    return usage_for(path).free_gb


def check_free_space(path: Path, min_free_gb: float, *, stage: str = "stage") -> DiskUsage:
    """Abort ``stage`` if the target volume is below ``min_free_gb``.

    Args:
        path: any path on the volume being written to.
        min_free_gb: floor from ``output.min_free_gb``.
        stage: named in the error so the log says which stage refused to run.

    Raises:
        DiskSpaceError: naming the drive, the free space, and the requirement.
    """
    usage = usage_for(path)
    if usage.free_gb < min_free_gb:
        raise DiskSpaceError(
            f"Refusing to run {stage}: drive {usage.drive} has "
            f"{usage.free_gb:.2f} GB free, below the {min_free_gb:.2f} GB minimum. "
            f"Free up space or lower output.min_free_gb."
        )
    return usage
