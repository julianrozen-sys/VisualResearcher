"""Embedding provider interface (CLAUDE.md §2.7).

§5's provider list does not name this kind, but §2.7's rule is unconditional:
pipeline logic never calls an SDK directly. CLIP is an SDK. So ranking talks
to this interface and ``open_clip`` stays behind it, which also means the
whole ranking stage runs offline against the fake.

Embeddings are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from ..base import Provider

__all__ = ["EmbeddingProvider", "cosine"]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two normalised vectors, clamped to 0.0..1.0.

    CLIP similarities are small positive numbers in practice; the clamp keeps
    a negative value from flipping the sign of a weighted score.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


class EmbeddingProvider(Provider):
    kind = "embedding"

    #: Dimensionality of the vectors this provider returns.
    dimensions: int = 512

    @abstractmethod
    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        """Embed images. Returns one normalised vector per path.

        A path that cannot be read yields a zero vector rather than raising:
        one unreadable file must not fail a segment (§8).
        """

    @abstractmethod
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed text prompts into the same space as the images."""
