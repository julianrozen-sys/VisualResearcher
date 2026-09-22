"""Fixture-backed LLM (CLAUDE.md §2.9).

Deliberately not clever. It does not attempt to be a small language model; it
produces structurally valid, deterministic answers derived from the prompt so
that every downstream stage can be exercised offline.

Where a real model would reason, this returns a defensible default and says so
in the ``reason`` field. That keeps the offline path honest: the shape is
right, the confidences are plausible, and nothing pretends to be an insight it
did not have.

A recorded fixture at ``tests/fixtures/llm/<task>.json`` always wins, which is
how tests pin exact answers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ...logging_setup import get_logger
from ..base import Availability
from .base import LLMProvider, LLMResponse

__all__ = ["FakeLLMProvider"]

log = get_logger("providers.llm.fake")

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llm"

#: Words that look like proper nouns but are not, when they open a sentence.
_SENTENCE_OPENERS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "but",
        "when",
        "what",
        "where",
        "who",
        "that",
        "this",
        "these",
        "those",
        "he",
        "she",
        "they",
        "it",
        "his",
        "her",
        "their",
        "its",
        "every",
        "nobody",
        "everyone",
        "there",
        "here",
        "for",
        "from",
        "with",
        "without",
        "before",
        "after",
        "if",
        "because",
        "while",
        "as",
        "at",
        "by",
        "in",
        "into",
        "on",
        "of",
        "to",
        "up",
        "down",
        "then",
        "now",
        "so",
        "yet",
        "still",
        "even",
        "one",
        "two",
        "three",
        "all",
    }
)

_PROPER = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+([A-Z][a-z]{2,}))?")


class FakeLLMProvider(LLMProvider):
    name = "fake"
    is_fake = True

    def __init__(self, *, fixture_dir: Path | None = None, **_ignored) -> None:
        self.fixture_dir = Path(fixture_dir) if fixture_dir else FIXTURE_DIR

    def availability(self) -> Availability:
        return Availability.available("deterministic, no network", offline_safe=True)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        task: str,
        schema_hint: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        fixture = self.fixture_dir / f"{task}.json"
        if fixture.exists():
            log.debug("fake llm: replaying fixture %s", fixture.name)
            return LLMResponse(json.loads(fixture.read_text(encoding="utf-8")), model="fixture")

        handler = {
            "project_context": self._project_context,
            "entity_resolution": self._entity_resolution,
            "segment_analysis": self._segment_analysis,
        }.get(task)
        if handler is None:
            log.warning("fake llm: no handler for task %r; returning an empty object", task)
            return LLMResponse({}, model="fake")
        return LLMResponse(handler(user), model="fake")

    # -- handlers ----------------------------------------------------------

    def _project_context(self, user: str) -> dict:
        """Guess a subject from the most frequent proper nouns."""
        names = _proper_nouns(user)
        return {
            "subject": names[0] if names else "",
            "franchise": "",
            "era": "",
            "characters": names[:8],
            "people": [],
            "places": [],
            "events": [],
            "terminology": [],
            "works": [],
        }

    def _entity_resolution(self, user: str) -> dict:
        """Resolve nothing.

        An offline stand-in that invented corrections would be worse than
        useless: §10 says corrections are never applied blindly, and a guess
        with a fabricated reason is exactly that. Returning an empty list
        leaves the domain pack and phonetic passes as the only sources of
        truth offline, which is the honest outcome.
        """
        return {"corrections": []}

    def _segment_analysis(self, user: str) -> dict:
        """One structurally valid analysis per segment in the batch.

        Derives a topic from the segment's own narration so the output varies
        per segment and query generation has something real to work with.
        """
        segments = _parse_batch(user)
        out = []
        for item in segments:
            narration = item.get("narration", "")
            entities = _proper_nouns(narration)
            out.append(
                {
                    "index": item.get("index"),
                    "topic": _topic_from(narration),
                    "entities": entities[:4],
                    "event": "",
                    "location": "",
                    "interpretation": (
                        "Offline stand-in: no model reasoning available. "
                        "Fall back to the strongest entity plus the setting."
                    ),
                    "visual_intent": "CHARACTER" if entities else "B_ROLL",
                    "confidence": 0.45,
                }
            )
        return {"segments": out}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _proper_nouns(text: str) -> list[str]:
    """Capitalised words that are not merely sentence-initial, most frequent first."""
    counts: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for match in _PROPER.finditer(sentence):
            first, second = match.group(1), match.group(2)
            if first.lower() in _SENTENCE_OPENERS:
                if not second:
                    continue
                name = second
            else:
                name = f"{first} {second}" if second else first
            # Skip a name that only ever appears at the very start of a sentence.
            if match.start() == 0 and first.lower() in _SENTENCE_OPENERS:
                continue
            counts[name] = counts.get(name, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _topic_from(narration: str) -> str:
    """A short topic line: the first clause, trimmed."""
    first = re.split(r"[.;:,]", narration.strip())[0].strip()
    words = first.split()
    return " ".join(words[:10]) if words else ""


def _parse_batch(user: str) -> list[dict]:
    """Pull the segment list back out of the prompt.

    The pipeline embeds the batch as a JSON array under a ``SEGMENTS:`` marker
    precisely so the fake can round-trip it without guessing.
    """
    marker = user.find("SEGMENTS:")
    if marker == -1:
        return []
    payload = user[marker + len("SEGMENTS:") :].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        log.debug("fake llm: could not parse the segment batch out of the prompt")
        return []
    return data if isinstance(data, list) else []


def stable_seed(text: str) -> int:
    """Deterministic seed from text, when a handler needs stable variation."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
