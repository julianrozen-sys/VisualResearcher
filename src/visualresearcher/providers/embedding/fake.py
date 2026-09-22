"""Deterministic stand-in for CLIP (CLAUDE.md §2.9).

It cannot do what CLIP does -- nothing offline can judge whether a picture
shows Darth Baras. What it *can* do is be honest and useful:

* **images** are embedded from their actual pixels: a coarse colour and
  edge-density signature, so visually similar images land near each other and
  different ones do not. That makes the diversity rule (§11) genuinely
  testable offline;
* **text** is embedded from a hash of its words. Text and image vectors
  therefore live in the same space but are not *meaningfully* related, so an
  offline similarity score is arbitrary.

Because that score is arbitrary, this provider reports ``meaningful = False``
and the ranking stage drops the CLIP term's weight to zero and redistributes
it, rather than pretending to a relevance judgement it did not make.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ...logging_setup import get_logger
from ..base import Availability
from .base import EmbeddingProvider

__all__ = ["FakeEmbeddingProvider"]

log = get_logger("providers.embedding.fake")

_GRID = 8  # 8x8 colour grid -> 192 dims, padded out to `dimensions`


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return [0.0] * len(vector)
    return [v / norm for v in vector]


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake"
    is_fake = True
    dimensions = 512

    #: Ranking reads this to decide whether the CLIP term means anything.
    meaningful = False

    def __init__(self, **_ignored) -> None:
        # The registry passes whatever the real provider takes (model,
        # pretrained, device, seed). A fake that cannot be constructed the
        # same way is not a substitute for it.
        pass

    def availability(self) -> Availability:
        return Availability.available(
            "pixel-signature embeddings; similarity to text is not meaningful",
            offline_safe=True,
        )

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        return [self._embed_image(Path(p)) for p in paths]

    def _embed_image(self, path: Path) -> list[float]:
        try:
            from PIL import Image

            with Image.open(path) as source:
                rgb = source.convert("RGB").resize((_GRID, _GRID), Image.BILINEAR)
                pixels = list(rgb.getdata())
        except Exception as exc:  # noqa: BLE001 - unreadable means zero vector
            log.debug("could not embed %s: %s", path, exc)
            return [0.0] * self.dimensions

        vector: list[float] = []
        for r, g, b in pixels:
            vector.extend((r / 255.0, g / 255.0, b / 255.0))
        # Pad deterministically so every vector has the declared dimensionality.
        vector.extend([0.0] * (self.dimensions - len(vector)))
        return _normalise(vector[: self.dimensions])

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        # Stretch the digest deterministically to the full dimensionality.
        stream = bytearray()
        counter = 0
        while len(stream) < self.dimensions:
            stream.extend(hashlib.sha256(digest + counter.to_bytes(4, "big")).digest())
            counter += 1
        return _normalise([(b / 255.0) - 0.5 for b in stream[: self.dimensions]])
