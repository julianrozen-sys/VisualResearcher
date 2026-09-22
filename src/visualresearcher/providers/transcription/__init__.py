"""Transcription providers. Importing this package registers them."""

from ..registry import register
from .base import TranscriptionProvider
from .fake import FakeTranscriptionProvider
from .faster_whisper import FasterWhisperProvider

register("transcription", "fake", FakeTranscriptionProvider, is_fake=True)
register("transcription", "faster_whisper", FasterWhisperProvider)

__all__ = ["TranscriptionProvider", "FakeTranscriptionProvider", "FasterWhisperProvider"]
