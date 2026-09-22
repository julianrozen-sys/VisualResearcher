"""Fixture-backed transcription (CLAUDE.md §2.9).

Two modes, in priority order:

1. a sidecar ``<audio>.transcript.json`` next to the audio, which is how tests
   pin an exact expected transcript;
2. otherwise a deterministic synthetic transcript generated from the audio's
   real duration, so ``VR_OFFLINE=1`` produces plausible, correctly-timed
   output for any WAV without needing a recorded fixture for it.

Deterministic in both modes — the same input always yields the same transcript.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from ...logging_setup import get_logger
from ...schemas import Transcript, TranscriptSegment, Word
from ..base import Availability
from .base import TranscriptionProvider

__all__ = ["FakeTranscriptionProvider", "audio_duration_s", "SAMPLE_NARRATION"]

log = get_logger("providers.transcription.fake")

#: Sentences used to build synthetic narration. SWTOR-flavoured on purpose:
#: the entity-resolution fixtures in §10 need "Barriss", "Valron", "Sanx" and
#: "Drog" to appear as the mishearings they are.
SAMPLE_NARRATION: tuple[str, ...] = (
    "When the Sith Warrior first walks into the chamber, the Dark Council is already waiting.",
    "Barriss stands at the centre of the circle, wrapped in that heavy mask he never removes.",
    "He has spent years convincing the council that he speaks for the Emperor himself.",
    "Valron watches from the far side, saying nothing, weighing every word that is spoken.",
    "The apprentice has just returned from Korriban with proof that the claim is a lie.",
    "Sanx had given up the location of the safe house before the interrogation even began.",
    "Drog fell on Hoth, and with him went the last witness who could have spoken for the defence.",
    "The council chamber falls silent as the accusation is finally made out loud.",
    "Barriss reaches for his lightsaber, and the vote turns against him in that instant.",
    "What follows is the single most satisfying confrontation in the entire Sith Warrior story.",
    "The mask comes away, and the Voice of the Emperor is revealed as nothing more than a man.",
    "Korriban itself seems to hold its breath while the duel plays out beneath the academy.",
)


def audio_duration_s(path: Path) -> float:
    """Duration of a WAV in seconds, or 0.0 if it cannot be read.

    Uses the stdlib ``wave`` module — no ffprobe subprocess, no extra
    dependency, and it works with zero network.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception as exc:  # noqa: BLE001 - any unreadable wav means "unknown"
        log.debug("could not read wav duration from %s: %s", path, exc)
        return 0.0


def _sidecar_for(audio_path: Path) -> Path:
    return audio_path.with_suffix(audio_path.suffix + ".transcript.json")


class FakeTranscriptionProvider(TranscriptionProvider):
    name = "fake"
    is_fake = True

    def __init__(self, *, words_per_second: float = 2.6, **_ignored) -> None:
        self.words_per_second = words_per_second

    def availability(self) -> Availability:
        return Availability.available("fixture-backed, no network", offline_safe=True)

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript:
        audio_path = Path(audio_path)
        sidecar = _sidecar_for(audio_path)
        if sidecar.exists():
            log.info("fake transcription: replaying fixture %s", sidecar.name)
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            transcript = Transcript.model_validate(data)
            transcript.provider = self.name
            return transcript

        duration = audio_duration_s(audio_path)
        if duration <= 0:
            # Not a readable WAV. Still produce something coherent rather than
            # failing the job, and say so in the log.
            log.warning("fake transcription: %s has no readable duration; assuming 60s", audio_path)
            duration = 60.0
        return self._synthesize(duration, language or "en")

    def _synthesize(self, duration: float, language: str) -> Transcript:
        """Lay sentences end to end across ``duration`` with plausible word timings.

        Every word gets a real start/end, monotonically increasing, with a small
        inter-sentence pause so silence-aware segmentation has genuine gaps to
        snap to.
        """
        pause = 0.35
        sentences: list[str] = []
        # Repeat the sample set until it covers the duration.
        approx_needed = max(1, int(duration * self.words_per_second / 14) + 1)
        while len(sentences) < approx_needed:
            sentences.extend(SAMPLE_NARRATION)
        sentences = sentences[:approx_needed]

        total_words = sum(len(s.split()) for s in sentences)
        speaking_time = max(0.1, duration - pause * max(0, len(sentences) - 1))
        per_word = speaking_time / total_words if total_words else 0.3

        segments: list[TranscriptSegment] = []
        cursor = 0.0
        for idx, sentence in enumerate(sentences):
            words: list[Word] = []
            seg_start = cursor
            for token in sentence.split():
                start = cursor
                end = min(duration, start + per_word)
                words.append(Word(word=token, start=round(start, 3), end=round(end, 3)))
                cursor = end
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=round(seg_start, 3),
                    end=round(cursor, 3),
                    text=sentence,
                    words=words,
                )
            )
            if idx < len(sentences) - 1:
                cursor = min(duration, cursor + pause)

        return Transcript(
            language=language,
            duration=round(duration, 3),
            text=" ".join(sentences),
            segments=segments,
            provider=self.name,
            model="synthetic",
        )
