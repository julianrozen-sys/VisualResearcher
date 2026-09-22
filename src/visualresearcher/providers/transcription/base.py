"""Transcription provider interface."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from ...schemas import Transcript
from ..base import Provider

__all__ = ["TranscriptionProvider"]


class TranscriptionProvider(Provider):
    kind = "transcription"

    @abstractmethod
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript:
        """Return a transcript with word-level timestamps.

        Word timestamps are not optional: segmentation snaps boundaries to real
        silence using them (§9), so a provider that cannot produce them is not
        usable here.
        """
