"""Stage 9: deduplication (CLAUDE.md §11).

Two passes, cheapest first:

1. **SHA256** catches byte-identical files. Free and certain.
2. **pHash and dHash**, Hamming distance <= ``images.phash_distance`` (6),
   catch the same picture after a crop, a resize or a recompression -- which
   is what image search actually returns. The same screenshot appears on four
   wikis at four sizes, and eight near-identical copies of one portrait is the
   failure mode §11 calls out by name.

Both hashes are used, and either one matching is enough. They fail
differently: dHash follows horizontal gradients and is robust to brightness
shifts, pHash works on frequency content and is robust to small crops. Where
one is fooled the other usually is not.

**Nothing is deleted.** The discarded record stays in the manifest with
``status="duplicate"`` and a note naming the file it duplicates, because §11
says to record discards and §2.8 says never to destroy user data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..logging_setup import get_logger
from ..schemas import ImageRecord

__all__ = [
    "compute_hashes",
    "hamming",
    "dedupe_records",
    "DedupeResult",
    "DEFAULT_DISTANCE",
]

log = get_logger("pipeline.dedupe")

DEFAULT_DISTANCE = 6


@dataclass
class DedupeResult:
    kept: list[ImageRecord] = field(default_factory=list)
    duplicates: list[ImageRecord] = field(default_factory=list)
    #: duplicate local_path -> the local_path it duplicates.
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.kept) + len(self.duplicates)


def compute_hashes(path: Path) -> tuple[str, str]:
    """``(phash, dhash)`` as hex strings, or ``("", "")`` if unreadable.

    An unreadable file is not an error here: it simply cannot be compared
    perceptually, and SHA256 still applies.
    """
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return str(imagehash.phash(rgb)), str(imagehash.dhash(rgb))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not hash %s: %s", path, exc)
        return "", ""


def hamming(a: str, b: str) -> int:
    """Hamming distance between two hex hash strings.

    Returns a large number when the hashes are missing or differ in length, so
    an unhashable image never looks like a duplicate of everything.
    """
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 999


def _quality(record: ImageRecord) -> tuple:
    """Sort key for "which copy do we keep".

    Bigger picture first, then more bytes (less compression damage), then a
    known licence over an unknown one, then the path, so the choice is stable
    across runs.
    """
    return (
        record.width * record.height,
        record.bytes,
        1 if record.license and record.license != "unknown" else 0,
        record.local_path,
    )


def dedupe_records(
    records: list[ImageRecord],
    *,
    distance: int = DEFAULT_DISTANCE,
    compute_missing: bool = True,
) -> DedupeResult:
    """Collapse duplicates, keeping the best copy of each picture.

    Deterministic: records are considered best-first, so the same input always
    yields the same survivor regardless of the order they arrived in.
    """
    result = DedupeResult()
    if not records:
        return result

    if compute_missing:
        for record in records:
            if not record.phash or not record.dhash:
                phash, dhash = compute_hashes(Path(record.local_path))
                record.phash, record.dhash = phash, dhash

    ordered = sorted(records, key=_quality, reverse=True)
    by_sha: dict[str, ImageRecord] = {}

    for record in ordered:
        # -- pass 1: exact bytes -------------------------------------------
        if record.sha256 and record.sha256 in by_sha:
            keeper = by_sha[record.sha256]
            _mark_duplicate(record, keeper, result, "exact SHA256 match")
            continue

        # -- pass 2: perceptual --------------------------------------------
        match: ImageRecord | None = None
        how = ""
        for keeper in result.kept:
            p = hamming(record.phash, keeper.phash)
            d = hamming(record.dhash, keeper.dhash)
            if p <= distance:
                match, how = keeper, f"pHash distance {p} <= {distance}"
                break
            if d <= distance:
                match, how = keeper, f"dHash distance {d} <= {distance}"
                break

        if match is not None:
            _mark_duplicate(record, match, result, how)
            continue

        record.status = "kept"
        result.kept.append(record)
        if record.sha256:
            by_sha[record.sha256] = record

    log.info(
        "dedupe: %d in, %d kept, %d duplicate(s) recorded (distance <= %d)",
        result.total,
        len(result.kept),
        len(result.duplicates),
        distance,
    )
    return result


def _mark_duplicate(
    record: ImageRecord, keeper: ImageRecord, result: DedupeResult, how: str
) -> None:
    record.status = "duplicate"
    record.notes = (
        f"{record.notes + '; ' if record.notes else ''}"
        f"duplicate of {Path(keeper.local_path).name} ({how})"
    )
    result.duplicates.append(record)
    result.mapping[record.local_path] = keeper.local_path
