"""Stage 2: transcribe (CLAUDE.md §8).

Writes three artifacts, all UTF-8:

* ``input/transcript.json`` — the full structured transcript with word timings,
* ``input/transcript.txt`` — plain text,
* ``input/narration.srt`` — subtitles, when ``output.write_srt``.

The transcript on disk always keeps the ORIGINAL words (§10.2). Entity
corrections never touch these files; they apply to search queries only.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..project import ProjectPaths
from ..providers.registry import resolve
from ..providers.transcription.base import TranscriptionProvider
from ..schemas import Transcript
from ..utils.files import atomic_write_text
from ..utils.timefmt import srt_stamp

__all__ = ["transcribe", "write_srt", "render_srt", "enforce_monotonic"]

log = get_logger("pipeline.transcribe")


def enforce_monotonic(transcript: Transcript) -> Transcript:
    """Clamp word timings so they never move backwards.

    Whisper occasionally emits a word whose start precedes the previous word's
    end by a few milliseconds. Segmentation looks for gaps between words; a
    negative gap there produces nonsense boundaries, so the invariant is
    enforced once, here, rather than defended against in every consumer.
    """
    cursor = 0.0
    for seg in transcript.segments:
        seg.start = max(seg.start, 0.0)
        for word in seg.words:
            if word.start < cursor:
                word.start = cursor
            if word.end < word.start:
                word.end = word.start
            cursor = word.end
        if seg.words:
            seg.start = min(seg.start, seg.words[0].start)
            seg.end = max(seg.end, seg.words[-1].end)
        if seg.end < seg.start:
            seg.end = seg.start
    return transcript


def render_srt(transcript: Transcript) -> str:
    """Standard SRT: 1-based index, ``HH:MM:SS,mmm --> HH:MM:SS,mmm``, blank-line separated."""
    blocks: list[str] = []
    counter = 0
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        counter += 1
        # A zero-length cue is not renderable; give it a minimum on-screen time.
        end = seg.end if seg.end > seg.start else seg.start + 0.5
        blocks.append(f"{counter}\n{srt_stamp(seg.start)} --> {srt_stamp(end)}\n{text}\n")
    return "\n".join(blocks)


def write_srt(transcript: Transcript, path: Path) -> Path:
    return atomic_write_text(path, render_srt(transcript))


def transcribe(
    audio_path: Path,
    paths: ProjectPaths,
    settings: Settings,
    *,
    force: bool = False,
    provider: TranscriptionProvider | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` and write the three input artifacts.

    Skips the provider call entirely when ``transcript.json`` already exists and
    ``force`` is false (§20 — stages skip if artifact + checkpoint exist).
    """
    if paths.transcript_json.exists() and not force:
        log.info("transcript exists, skipping transcription (use --force to redo)")
        transcript = Transcript.model_validate_json(
            paths.transcript_json.read_text(encoding="utf-8")
        )
        return transcript

    if provider is None:
        provider = resolve(  # type: ignore[assignment]
            "transcription",
            settings.transcription.provider,
            model=settings.transcription.model,
            device=settings.transcription.device,
            compute_type=settings.transcription.compute_type,
        )
    log.info("transcribing with %s provider", provider.name)
    transcript = provider.transcribe(audio_path, language=settings.transcription.language)
    transcript = enforce_monotonic(transcript)

    words = transcript.all_words()
    log.info(
        "transcript: %d segments, %d words, %.1fs, language=%s",
        len(transcript.segments),
        len(words),
        transcript.duration,
        transcript.language,
    )
    if not words:
        log.warning(
            "no word timestamps returned by %s; segmentation will fall back to "
            "segment-level boundaries",
            provider.name,
        )

    atomic_write_text(paths.transcript_json, transcript.model_dump_json(indent=2), backup=force)
    atomic_write_text(paths.transcript_txt, transcript.text.strip() + "\n", backup=force)
    if settings.output.write_srt:
        write_srt(transcript, paths.narration_srt)
        log.info("wrote %s", paths.narration_srt.name)
    return transcript
