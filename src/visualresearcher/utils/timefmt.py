"""Time formatting. Folder names, SRT stamps, and the human timecodes in reports.

Every function here feeds a filename or a subtitle file, so they are all
deterministic and never locale-dependent.
"""

from __future__ import annotations

__all__ = ["folder_stamp", "srt_stamp", "timecode", "parse_timecode", "yt_dlp_stamp"]


def folder_stamp(seconds: float) -> str:
    """``229.0`` -> ``03m49s``. Used in segment folder names (CLAUDE.md §9)."""
    total = int(seconds)
    return f"{total // 60:02d}m{total % 60:02d}s"


def timecode(seconds: float) -> str:
    """``229.0`` -> ``00:03:49``. Human-readable, used in CSV/markdown reports."""
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def srt_stamp(seconds: float) -> str:
    """``229.5`` -> ``00:03:49,500``. SRT uses a comma for the decimal separator."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def yt_dlp_stamp(seconds: float) -> str:
    """``102.5`` -> ``00:01:42.50``, the format ``--download-sections`` expects."""
    if seconds < 0:
        seconds = 0.0
    hours, rem = divmod(seconds, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def parse_timecode(text: str) -> float:
    """Parse ``HH:MM:SS``, ``MM:SS``, or ``SS`` (with optional ``.`` / ``,`` millis)."""
    text = text.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"not a timecode: {text!r}")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total
