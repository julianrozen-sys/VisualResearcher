"""Image search providers. Importing this package registers them."""

from ..registry import register
from .base import ImageCandidate, ImageSearchProvider
from .ddgs_provider import DdgsImageProvider
from .fake import FakeImageSearchProvider
from .wikimedia import WikimediaImageProvider

register("images", "fake", FakeImageSearchProvider, is_fake=True)
register("images", "ddgs", DdgsImageProvider)
register("images", "wikimedia", WikimediaImageProvider)

__all__ = [
    "ImageCandidate",
    "ImageSearchProvider",
    "DdgsImageProvider",
    "FakeImageSearchProvider",
    "WikimediaImageProvider",
]
