"""Video providers. Importing this package registers them."""

from ..registry import register
from .base import SectionRequest, SectionResult, VideoHit, VideoProvider
from .fake import FakeVideoProvider
from .ytdlp import YtDlpVideoProvider

register("video", "fake", FakeVideoProvider, is_fake=True)
register("video", "ytdlp", YtDlpVideoProvider)

__all__ = [
    "SectionRequest",
    "SectionResult",
    "VideoHit",
    "VideoProvider",
    "FakeVideoProvider",
    "YtDlpVideoProvider",
]
