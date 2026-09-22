"""Filesystem helpers.

Three jobs, all of which have bitten real pipelines before:

* turn arbitrary text (a narration topic, a remote filename) into something
  Windows will actually accept as a filename,
* refuse to write outside the directory we meant to write into,
* never leave a half-written file where a complete one is expected.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

__all__ = [
    "slugify",
    "sanitize_filename",
    "resolve_within",
    "is_within",
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_move",
    "backup_file",
    "pad_width",
    "selected_filename",
    "segment_dirname",
    "ensure_dir",
    "dir_size_bytes",
]

# Windows forbids these outright, plus the ASCII control range.
_ILLEGAL = re.compile(r'[<>:"/\|?*\x00-\x1f]')
_NON_SLUG = re.compile(r"[^a-z0-9]+")
# Device names Windows reserves regardless of extension.
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase ASCII slug joined by underscores.

    Truncation snaps to a word boundary where one is nearby, so
    ``darth_baras_dark_council_chamber`` does not become ``darth_baras_dark_c``.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _NON_SLUG.sub("_", ascii_text).strip("_")
    if not slug:
        return "untitled"
    if len(slug) <= max_len:
        return slug
    cut = slug[:max_len]
    if "_" in cut[max_len // 2 :]:
        cut = cut.rsplit("_", 1)[0]
    return cut.strip("_") or slug[:max_len]


def sanitize_filename(name: str, *, default: str = "file", max_len: int = 120) -> str:
    """Make ``name`` safe to use as a single path component.

    Never trust a remote filename (CLAUDE.md §11). Strips directory parts,
    illegal characters, traversal sequences, trailing dots/spaces, and Windows
    reserved device names. Always returns a non-empty single component.
    """
    # Drop anything that looks like a directory, whichever separator was used.
    name = name.replace("\\", "/").split("/")[-1]
    name = _ILLEGAL.sub("_", name).strip()
    # ".." and friends must not survive as a whole component.
    while name.startswith("."):
        name = name[1:]
    name = name.rstrip(". ")
    if not name:
        return default
    stem, dot, ext = name.partition(".")
    if stem.lower() in _RESERVED:
        stem = f"_{stem}"
    name = f"{stem}{dot}{ext}"
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 8:
            name = stem[: max_len - len(ext) - 1] + "." + ext
        else:
            name = name[:max_len]
    return name or default


def is_within(base: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves inside ``base``."""
    try:
        base_r = base.resolve()
        cand_r = candidate.resolve()
    except OSError:
        return False
    return cand_r == base_r or base_r in cand_r.parents


def resolve_within(base: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``base`` and refuse to escape it.

    Raises ``ValueError`` on traversal. This is the only sanctioned way to build
    a path from untrusted text.
    """
    safe_parts = [sanitize_filename(p) for p in parts if p not in ("", ".")]
    candidate = base.joinpath(*safe_parts)
    if not is_within(base, candidate):
        raise ValueError(f"path escapes {base}: {'/'.join(parts)}")
    return candidate


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_file(path: Path) -> Path | None:
    """Copy ``path`` to ``path.bak`` before it is overwritten (CLAUDE.md §2.8)."""
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup


#: How many times to retry the rename, and how long to back off between tries.
REPLACE_ATTEMPTS = 6
REPLACE_BACKOFF_S = 0.05


def atomic_write_bytes(path: Path, data: bytes, *, backup: bool = False) -> Path:
    """Write via a sibling temp file then replace, so readers never see a partial file.

    The rename is retried, which on Windows is not optional (§2.12). A file
    that was created moments ago can still be held open by the search indexer
    or an antivirus scanner, and ``os.replace`` then fails with ``WinError 5``
    (access denied) or ``WinError 32`` (in use). Those locks are transient --
    milliseconds -- but without a retry they surface as a stage crashing
    partway through a run, which is both alarming and untrue.

    Two concurrent jobs widen that window, which is how this was found: a
    worker-pool test failed writing a ``segment.json`` it alone owned.
    """
    ensure_dir(path.parent)
    if backup:
        backup_file(path)

    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    last: OSError | None = None
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError as exc:  # WinError 5 / 32
            last = exc
            time.sleep(REPLACE_BACKOFF_S * (2**attempt))
        except OSError as exc:
            # Anything else is a real problem: a bad path, a full disk, a
            # read-only volume. Retrying those would only delay the report.
            last = exc
            break

    tmp.unlink(missing_ok=True)
    raise OSError(f"could not replace {path} after {REPLACE_ATTEMPTS} attempt(s): {last}") from last


def atomic_write_text(path: Path, text: str, *, backup: bool = False) -> Path:
    """UTF-8 always (CLAUDE.md §2.12)."""
    return atomic_write_bytes(path, text.encode("utf-8"), backup=backup)


def atomic_move(src: Path, dst: Path) -> Path:
    """Move ``src`` onto ``dst``, falling back to copy+delete across volumes."""
    ensure_dir(dst.parent)
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        src.unlink()
    return dst


def pad_width(max_index: int) -> int:
    """Digits needed so lexicographic order matches numeric order.

    CLAUDE.md §6 specifies ``{segment:03d}``; three digits is therefore the
    floor. The stated *intent* of that padding is "lexicographic sort =
    chronological", which three digits stops delivering at segment 1000, so the
    width widens past 999 rather than silently breaking the ordering guarantee.
    """
    return max(3, len(str(max(0, max_index))))


def segment_dirname(index: int, start_s: float, end_s: float, width: int = 3) -> str:
    """``031_03m49s-03m57s`` (CLAUDE.md §9)."""
    from .timefmt import folder_stamp

    return f"{index:0{width}d}_{folder_stamp(start_s)}-{folder_stamp(end_s)}"


def candidate_filename(
    index: int,
    start_s: float,
    rank: int,
    slug: str,
    ext: str,
    width: int = 3,
    rank_width: int = 2,
) -> str:
    """``047_06m16s_01_dark_council_wide.jpg`` -- sorts chronologically.

    Three things make the ordering guarantee hold, and each has a way of
    quietly failing:

    * **The segment number comes first, zero-padded.** It is the only field
      that decides order. `width` comes from :func:`pad_width`, never a
      hardcoded 3, or a project past segment 999 loses the guarantee.
    * **The timecode is decorative.** It reads well and it must not be trusted
      to sort: past 99 minutes the field gains a digit and ``100m20s`` sorts
      *before* ``10m30s``. Moving it in front of the segment number looks like
      a harmless simplification and silently breaks every long narration.
    * **The rank is padded too.** ``keep_per_segment`` is configuration, not a
      constant. Above 9 an unpadded rank puts ``_10_`` before ``_2_``, so a
      segment's own candidates come out shuffled.
    """
    ext = ext.lower().lstrip(".")
    from .timefmt import folder_stamp

    return f"{index:0{width}d}_{folder_stamp(start_s)}_{rank:0{rank_width}d}_{slugify(slug)}.{ext}"


def rank_pad_width(max_rank: int) -> int:
    """Digits needed so ranks inside one segment sort in rank order."""
    return max(2, len(str(max(0, max_rank))))


def selected_filename(index: int, pick: int, slug: str, ext: str, width: int = 3) -> str:
    """``031_1_darth_baras_dark_council.jpg`` (CLAUDE.md §6)."""
    ext = ext.lower().lstrip(".")
    return f"{index:0{width}d}_{pick:d}_{slugify(slug)}.{ext}"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
