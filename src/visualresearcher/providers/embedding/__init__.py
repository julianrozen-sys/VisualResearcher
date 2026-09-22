"""Embedding providers. Importing this package registers them."""

from ..registry import register
from .base import EmbeddingProvider, cosine
from .fake import FakeEmbeddingProvider
from .openclip import OpenClipEmbeddingProvider

register("embedding", "fake", FakeEmbeddingProvider, is_fake=True)
register("embedding", "openclip", OpenClipEmbeddingProvider)

__all__ = [
    "EmbeddingProvider",
    "cosine",
    "FakeEmbeddingProvider",
    "OpenClipEmbeddingProvider",
]
