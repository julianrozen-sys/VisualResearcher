"""faster-whisper transcription (CLAUDE.md §4 — the default real transcriber).

The import is lazy on purpose: ``doctor`` must be able to report "not
installed" without the process blowing up at import time, and ``VR_OFFLINE=1``
must never need the package at all.

Model weights land wherever ``HF_HOME`` points (§3). No cache path is
hardcoded here.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ...logging_setup import get_logger
from ...schemas import Transcript, TranscriptSegment, Word
from ..base import Availability, ProviderError
from .base import TranscriptionProvider

__all__ = ["FasterWhisperProvider"]

log = get_logger("providers.transcription.faster_whisper")


class FasterWhisperProvider(TranscriptionProvider):
    name = "faster_whisper"

    def __init__(
        self,
        *,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        **_ignored,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def availability(self) -> Availability:
        if importlib.util.find_spec("faster_whisper") is None:
            return Availability.unavailable(
                "faster-whisper is not installed",
                "uv pip install faster-whisper",
            )
        if not os.environ.get("HF_HOME"):
            # Not fatal, but on this machine an unset HF_HOME means weights go
            # to C:, which §3 forbids.
            return Availability.available(
                f"model={self.model_name} (warning: HF_HOME unset; weights may land on C:)"
            )
        return Availability.available(f"model={self.model_name} device={self.device}")

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - guarded by availability()
            raise ProviderError(f"faster-whisper not installed: {exc}") from exc
        log.info("loading faster-whisper model %r (%s)", self.model_name, self.device)
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )
        return self._model

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        segments: list[TranscriptSegment] = []
        texts: list[str] = []
        for idx, seg in enumerate(segments_iter):
            words = [
                Word(
                    word=w.word.strip(),
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    probability=round(float(getattr(w, "probability", 1.0)), 4),
                )
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ]
            text = seg.text.strip()
            texts.append(text)
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=text,
                    words=words,
                )
            )
        return Transcript(
            language=getattr(info, "language", language or "en"),
            duration=round(float(getattr(info, "duration", 0.0)), 3),
            text=" ".join(texts).strip(),
            segments=segments,
            provider=self.name,
            model=self.model_name,
        )
