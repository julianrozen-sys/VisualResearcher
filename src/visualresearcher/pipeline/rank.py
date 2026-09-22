"""Stage 10: ranking and diversity (CLAUDE.md §11).

§11 specifies the score exactly:

    CLIP similarity + entity bonus + exact-event bonus + source tier +
    resolution + sharpness (Laplacian) + watermark penalty + repetition penalty

with the weights coming from ``ranking.weights`` and the whole thing
deterministic under a fixed seed.

Two design points are worth stating because they are not obvious:

**Selection is not "take the top 8".** §11 also requires the final set to span
different shot types and forbids "eight near-identical portraits". A pure
top-N of a similarity score reliably produces exactly that, because the
highest-scoring images for one prompt tend to be the same picture. So
selection is greedy maximal-marginal-relevance: each pick is penalised by how
much it resembles what has already been picked. Diversity is a property of the
*selection procedure*, not a filter applied afterwards.

**The CLIP weight is dropped when the embedding provider says its similarity
is not meaningful.** Offline, image and text vectors exist in the same space
but are not semantically related, so the similarity is arbitrary. Scoring 45%
of the result on an arbitrary number would be worse than not scoring it at
all, so the weight is redistributed across the terms that do mean something
and the segment records that it was ranked without CLIP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..providers.embedding.base import EmbeddingProvider, cosine
from ..schemas import ImageRecord, ProjectContext, Segment
from .dedupe import hamming

__all__ = [
    "rank_segment",
    "build_prompt",
    "score_records",
    "select_diverse",
    "sharpness",
    "watermark_penalty",
    "source_tier",
    "SegmentHistory",
    "write_contact_sheet",
]

log = get_logger("pipeline.rank")

#: Source quality, highest first. §11 calls this the "source tier".
_SOURCE_TIERS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (1.00, ("commons.wikimedia.org", "upload.wikimedia.org", "wikipedia.org")),
    (0.85, ("swtor.com", "starwars.com", "lucasfilm.com", "ea.com", "bioware.com")),
    (0.70, ("fandom.com", "wikia.nocookie.net", "gamepedia.com", "wiki.gg")),
    (0.55, ("artstation.com", "deviantart.com", "flickr.com")),
    (0.40, ("reddit.com", "imgur.com", "pinterest.com", "tumblr.com")),
)
_DEFAULT_TIER = 0.5

#: How strongly a pick is penalised for resembling an already-picked image.
#: High enough that two near-identical images will not both be chosen while
#: any genuinely different candidate remains.
DIVERSITY_STRENGTH = 0.8

#: How many previous segments to remember for the cross-segment repetition
#: penalty (§11: "penalise repeats across recent segments").
HISTORY_SEGMENTS = 6


# ---------------------------------------------------------------------------
# Individual score components
# ---------------------------------------------------------------------------


def build_prompt(segment: Segment, context: ProjectContext) -> str:
    """The CLIP prompt: interpretation + entities + event + location (§11)."""
    parts: list[str] = []
    if segment.interpretation:
        parts.append(segment.interpretation)
    if segment.entities:
        parts.append(", ".join(segment.entities[:4]))
    if segment.event:
        parts.append(segment.event)
    if segment.location:
        parts.append(segment.location)
    if not parts and segment.narration:
        parts.append(segment.narration)
    if context.search_tag or context.franchise:
        parts.append(context.search_tag or context.franchise)
    return ". ".join(p.strip() for p in parts if p.strip())[:600]


def source_tier(domain: str) -> float:
    """0.0..1.0 by how reliable the hosting domain tends to be."""
    lowered = (domain or "").lower()
    for score, domains in _SOURCE_TIERS:
        if any(lowered.endswith(d) or d in lowered for d in domains):
            return score
    return _DEFAULT_TIER


def entity_bonus(record: ImageRecord, segment: Segment) -> float:
    """How many of the segment's entities appear in the record's own text.

    The query, title and source page are all we have without looking at the
    pixels; a page called ``Darth_Baras`` is decent evidence.
    """
    entities = [e.lower() for e in segment.entities if len(e) > 2]
    if not entities:
        return 0.0
    haystack = " ".join((record.query or "", record.source_page or "", record.notes or "")).lower()
    hits = sum(1 for e in entities if e in haystack)
    return hits / len(entities)


def exact_event_bonus(record: ImageRecord) -> float:
    """§11's exact-event bonus, driven by which query kind found the image."""
    return {
        "exact_event": 1.0,
        "quest": 0.6,
        "character_setting": 0.5,
        "location": 0.35,
        "object": 0.3,
        "youtube": 0.2,
        "fallback": 0.0,
    }.get(str(record.query_kind), 0.0)


def resolution_score(record: ImageRecord) -> float:
    """Megapixels on a saturating curve.

    Saturating rather than linear: 2MP is much better than 0.5MP, but 24MP is
    not meaningfully better than 12MP for a 1080p video, and a linear score
    would let one enormous image outrank a more relevant one.
    """
    megapixels = (record.width * record.height) / 1_000_000
    if megapixels <= 0:
        return 0.0
    return min(1.0, math.log1p(megapixels) / math.log1p(8.0))


def _grayscale_array(path: Path, size: int = 256):
    """Small grayscale numpy array, or None when the file cannot be read."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as source:
            gray = source.convert("L")
            gray.thumbnail((size, size), Image.BILINEAR)
            return np.asarray(gray, dtype=float)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read %s for quality analysis: %s", path, exc)
        return None


def sharpness(path: Path) -> float:
    """Laplacian variance, normalised to 0.0..1.0 (§11).

    A blurry upscale and a crisp screenshot score very differently, which is
    exactly the distinction worth making when both otherwise match the prompt.
    """
    array = _grayscale_array(path)
    if array is None or array.size == 0:
        return 0.0
    import numpy as np

    # 4-neighbour Laplacian, computed by shifting rather than convolving so
    # this needs only numpy and not scipy.
    laplacian = (
        -4 * array[1:-1, 1:-1]
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )
    if laplacian.size == 0:
        return 0.0
    variance = float(np.var(laplacian))
    # ~500 is a comfortably sharp photograph at this scale.
    return min(1.0, math.log1p(variance) / math.log1p(500.0))


def watermark_penalty(path: Path) -> float:
    """0.0..1.0, higher meaning "more likely to carry a watermark or border".

    §11 asks for border edge-density with no OCR dependency. Stock-photo
    watermarks, site banners and "click to enlarge" bars concentrate hard
    edges near the frame edge, while a photograph's detail is spread through
    the middle. So: compare edge energy in the outer band against the centre,
    and penalise only when the border is markedly busier.
    """
    array = _grayscale_array(path)
    if array is None or min(array.shape) < 32:
        return 0.0
    import numpy as np

    gradient_y, gradient_x = np.gradient(array)
    edges = np.abs(gradient_x) + np.abs(gradient_y)

    height, width = edges.shape
    band_h = max(2, height // 8)
    band_w = max(2, width // 8)

    border = np.concatenate(
        [
            edges[:band_h, :].ravel(),
            edges[-band_h:, :].ravel(),
            edges[:, :band_w].ravel(),
            edges[:, -band_w:].ravel(),
        ]
    )
    centre = edges[band_h:-band_h, band_w:-band_w]
    if centre.size == 0:
        return 0.0

    border_energy = float(np.mean(border))
    centre_energy = float(np.mean(centre)) or 1e-6
    ratio = border_energy / centre_energy

    # A ratio near 1.0 is ordinary. Penalise only a clearly busier border.
    if ratio <= 1.3:
        return 0.0
    return min(1.0, (ratio - 1.3) / 1.7)


# ---------------------------------------------------------------------------
# Cross-segment history (§11: penalise repeats across recent segments)
# ---------------------------------------------------------------------------


@dataclass
class SegmentHistory:
    """Perceptual hashes of images picked for the last few segments."""

    window: int = HISTORY_SEGMENTS
    entries: list[list[str]] = field(default_factory=list)

    def add(self, records: list[ImageRecord]) -> None:
        self.entries.append([r.phash for r in records if r.phash])
        if len(self.entries) > self.window:
            self.entries.pop(0)

    def repetition(self, record: ImageRecord) -> float:
        """0.0..1.0 -- how much this image resembles recent picks.

        Recency-weighted: repeating the shot from the previous segment is more
        jarring than repeating one from six segments ago.
        """
        if not record.phash or not self.entries:
            return 0.0
        worst = 0.0
        count = len(self.entries)
        for age, hashes in enumerate(reversed(self.entries)):
            recency = 1.0 - (age / max(1, count))
            for other in hashes:
                distance = hamming(record.phash, other)
                if distance <= 12:
                    similarity = 1.0 - (distance / 12.0)
                    worst = max(worst, similarity * recency)
        return worst


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_records(
    records: list[ImageRecord],
    segment: Segment,
    context: ProjectContext,
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    history: SegmentHistory | None = None,
) -> list[ImageRecord]:
    """Fill ``score`` and ``score_breakdown`` on every record.

    Deterministic: every component is a pure function of the record, the
    segment and the history.
    """
    if not records:
        return records

    weights = settings.ranking.weights
    clip_meaningful = bool(embedder and getattr(embedder, "meaningful", False))

    # §11's weights assume a real CLIP term. Without one, redistribute its
    # share rather than scoring 45% of the result on an arbitrary number.
    active = {
        "clip": weights.clip if clip_meaningful else 0.0,
        "entity": weights.entity,
        "source": weights.source,
        "resolution": weights.resolution,
        "sharpness": weights.sharpness,
    }
    if not clip_meaningful:
        share = weights.clip / 4
        for key in ("entity", "source", "resolution", "sharpness"):
            active[key] += share
        log.info(
            "segment %03d: ranking without CLIP (%s reports its similarity is "
            "not meaningful); the clip weight was redistributed",
            segment.index,
            embedder.name if embedder else "no embedder",
        )

    clip_scores = [0.0] * len(records)
    if embedder is not None:
        prompt = build_prompt(segment, context)
        try:
            text_vector = embedder.embed_text([prompt])[0]
            image_vectors = embedder.embed_images([Path(r.local_path) for r in records])
            clip_scores = [cosine(v, text_vector) for v in image_vectors]
            for record, vector in zip(records, image_vectors, strict=True):
                record.score_breakdown["_embedding_ok"] = 1.0 if any(vector) else 0.0
        except Exception as exc:  # noqa: BLE001 - never fail a segment on this
            log.warning(
                "segment %03d: embedding failed, continuing without it: %s", segment.index, exc
            )
            segment.notes.append(f"ranking ran without CLIP: {exc}")

    for record, clip_score in zip(records, clip_scores, strict=True):
        path = Path(record.local_path)
        components = {
            "clip": clip_score,
            "entity": max(entity_bonus(record, segment), exact_event_bonus(record) * 0.5),
            "exact_event": exact_event_bonus(record),
            "source": source_tier(record.domain),
            "resolution": resolution_score(record),
            "sharpness": sharpness(path),
            "watermark": watermark_penalty(path),
            "repetition": history.repetition(record) if history else 0.0,
        }
        total = (
            active["clip"] * components["clip"]
            + active["entity"] * components["entity"]
            + active["source"] * components["source"]
            + active["resolution"] * components["resolution"]
            + active["sharpness"] * components["sharpness"]
            # §11 lists the exact-event bonus as its own term.
            + 0.10 * components["exact_event"]
            # Both penalties are subtractive; repetition's weight is negative
            # in config, so its absolute value is used here.
            - abs(weights.repetition) * components["repetition"]
            - 0.15 * components["watermark"]
        )
        record.score = round(max(0.0, total), 6)
        record.score_breakdown = {k: round(v, 6) for k, v in components.items()}
        record.score_breakdown["_weighted_total"] = record.score

    # Sort by score, then sha256 so ties break identically on every run.
    records.sort(key=lambda r: (-r.score, r.sha256))
    return records


# ---------------------------------------------------------------------------
# Diverse selection
# ---------------------------------------------------------------------------


def _visual_distance(a: ImageRecord, b: ImageRecord) -> float:
    """0.0 (identical) .. 1.0 (unrelated), from the perceptual hashes."""
    if not a.phash or not b.phash:
        return 1.0
    distance = min(hamming(a.phash, b.phash), hamming(a.dhash, b.dhash))
    if distance >= 999:
        return 1.0
    return min(1.0, distance / 20.0)


def _shape_bucket(record: ImageRecord) -> str:
    """A coarse shot-type proxy from the frame shape.

    Not a real shot-type classifier -- that needs the model. But aspect ratio
    does separate a portrait crop from a widescreen establishing shot from a
    square icon, which is enough to stop the final eight all being the same
    kind of frame.
    """
    ratio = record.aspect_ratio or (record.width / record.height if record.height else 1.0)
    if ratio >= 1.9:
        return "ultrawide"
    if ratio >= 1.4:
        return "widescreen"
    if ratio >= 1.1:
        return "landscape"
    if ratio >= 0.9:
        return "square"
    return "portrait"


def select_diverse(
    records: list[ImageRecord],
    keep: int,
    *,
    strength: float = DIVERSITY_STRENGTH,
) -> list[ImageRecord]:
    """Greedy maximal-marginal-relevance selection (§11's diversity rule).

    Each candidate's effective score is its own score minus how much it
    resembles what has already been chosen, so the second copy of a picture
    loses to a different picture even when its raw score is higher.
    """
    if keep <= 0 or not records:
        return []
    pool = sorted(records, key=lambda r: (-r.score, r.sha256))
    chosen: list[ImageRecord] = [pool[0]]
    remaining = pool[1:]

    while remaining and len(chosen) < keep:
        best = None
        best_value = -math.inf
        for candidate in remaining:
            similarity = max(1.0 - _visual_distance(candidate, picked) for picked in chosen)
            # A shape already represented is mildly discouraged, so the final
            # set spans different kinds of frame.
            shapes = [_shape_bucket(p) for p in chosen]
            shape_penalty = 0.05 * shapes.count(_shape_bucket(candidate))
            value = candidate.score - strength * similarity - shape_penalty
            if value > best_value:
                best_value, best = value, candidate
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)

    for rank, record in enumerate(chosen, start=1):
        record.rank = rank
        record.status = "selected"
    return chosen


def rank_segment(
    records: list[ImageRecord],
    segment: Segment,
    context: ProjectContext,
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    history: SegmentHistory | None = None,
) -> list[ImageRecord]:
    """Score, select a diverse top-N, and record the outcome on the segment."""
    if not records:
        segment.notes.append("no images to rank")
        return []

    score_records(records, segment, context, settings, embedder=embedder, history=history)
    keep = settings.images.keep_per_segment
    chosen = select_diverse(records, keep)

    if history is not None:
        history.add(chosen)

    shapes = {_shape_bucket(r) for r in chosen}
    log.info(
        "segment %03d: ranked %d, kept %d (scores %.3f-%.3f, %d frame shape(s))",
        segment.index,
        len(records),
        len(chosen),
        chosen[-1].score if chosen else 0.0,
        chosen[0].score if chosen else 0.0,
        len(shapes),
    )
    return chosen


# ---------------------------------------------------------------------------
# Contact sheet (§6)
# ---------------------------------------------------------------------------

#: Grid cell size for the contact sheet, in pixels.
_CELL = 320
_LABEL_H = 22


def write_contact_sheet(
    records: list[ImageRecord], path: Path, *, columns: int = 4, cell: int = _CELL
) -> Path | None:
    """One JPEG per segment showing the selected images in rank order.

    This is how a human checks a segment in one glance instead of opening
    eight files. Each cell is labelled with its rank and score so a bad
    ranking is visible rather than merely suspected.
    """
    if not records:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return None

    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell, rows * (cell + _LABEL_H)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    for position, record in enumerate(records):
        column, row = position % columns, position // columns
        x, y = column * cell, row * (cell + _LABEL_H)
        try:
            with Image.open(record.local_path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
                sheet.paste(
                    thumb,
                    (x + (cell - thumb.width) // 2, y + (cell - thumb.height) // 2),
                )
        except Exception as exc:  # noqa: BLE001 - a missing thumb is not fatal
            log.debug("contact sheet: could not place %s: %s", record.local_path, exc)
            draw.rectangle([x + 4, y + 4, x + cell - 4, y + cell - 4], outline=(120, 60, 60))

        label = f"#{record.rank}  {record.score:.3f}  {Path(record.local_path).name}"
        draw.text((x + 6, y + cell + 4), label[:46], fill=(210, 210, 215))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=82)
    return path
