"""Shared test fixtures.

Every test runs against a sandboxed install root under ``tmp_path``, so no test
can write into the real ``projects/`` tree or the real job database. Offline
mode is forced on for the whole suite: the test suite must never touch the
network (CLAUDE.md §21).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from visualresearcher.config import Settings, load_settings
from visualresearcher.db import init_db, reset_engines
from visualresearcher.logging_setup import setup_logging
from visualresearcher.schemas import Transcript, TranscriptSegment, Word

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _offline() -> None:
    os.environ["VR_OFFLINE"] = "1"
    setup_logging("WARNING")


@pytest.fixture(autouse=True)
def _clean_engines():
    """A fresh engine per test, so temp databases never leak between tests."""
    reset_engines()
    yield
    reset_engines()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """An install root with the standard subfolders, entirely inside tmp_path."""
    for name in ("projects", "data", ".cache", ".tmp", "domain_packs"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(sandbox: Path) -> Settings:
    cfg = load_settings(config_path=sandbox / "does-not-exist.yaml", root=sandbox)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def db_path(settings: Settings) -> Path:
    init_db(settings.db_path)
    return settings.db_path


# Session-scoped: these are constant paths to files on disk, not state, and
# a module-scoped fixture elsewhere needs to depend on them.
@pytest.fixture(scope="session")
def sample_wav() -> Path:
    path = FIXTURES / "sample.wav"
    assert path.exists(), "run the fixture generation step in README.md"
    return path


@pytest.fixture(scope="session")
def silence_wav() -> Path:
    return FIXTURES / "silence.wav"


@pytest.fixture(scope="session")
def awkward_wav() -> Path:
    return FIXTURES / "awkward_11s.wav"


def make_transcript(
    sentences: list[str],
    *,
    word_s: float = 0.35,
    pause_s: float = 0.4,
    start: float = 0.0,
) -> Transcript:
    """Build a transcript with exact, predictable word timings.

    Tests that assert on segmentation need to know precisely where the
    sentence boundaries and silences are, which a synthetic provider's output
    does not guarantee. This does.
    """
    segments: list[TranscriptSegment] = []
    cursor = start
    for idx, sentence in enumerate(sentences):
        words: list[Word] = []
        seg_start = cursor
        for token in sentence.split():
            words.append(Word(word=token, start=round(cursor, 3), end=round(cursor + word_s, 3)))
            cursor += word_s
        segments.append(
            TranscriptSegment(
                id=idx,
                start=round(seg_start, 3),
                end=round(cursor, 3),
                text=sentence,
                words=words,
            )
        )
        cursor += pause_s
    return Transcript(
        language="en",
        duration=round(cursor, 3),
        text=" ".join(sentences),
        segments=segments,
        provider="test",
    )
