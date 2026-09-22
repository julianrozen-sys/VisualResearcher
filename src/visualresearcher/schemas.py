"""The on-disk contracts from CLAUDE.md §7.

These models are the interchange format between stages and the shape of the
JSON the user reads. Field names match §7 exactly — if a name here drifts, the
output drifts with it, so treat them as frozen.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "VisualIntent",
    "QueryKind",
    "Query",
    "EntityCorrection",
    "Work",
    "ProjectContext",
    "SubShot",
    "Pick",
    "Segment",
    "Word",
    "TranscriptSegment",
    "Transcript",
    "ImageRecord",
    "TimestampCandidate",
    "YouTubeRecord",
    "Classification",
]


class VisualIntent(StrEnum):
    EXACT_EVENT = "EXACT_EVENT"
    CHARACTER = "CHARACTER"
    LOCATION = "LOCATION"
    GAME_SCENE = "GAME_SCENE"
    FILM_SCENE = "FILM_SCENE"
    COMIC = "COMIC"
    ABSTRACT = "ABSTRACT"
    B_ROLL = "B_ROLL"


class QueryKind(StrEnum):
    #: Built from the segment's own transcribed words plus the franchise tag.
    #: Every segment gets one, because the narration is the only description
    #: of the shot that is guaranteed to exist.
    NARRATION = "narration"
    EXACT_EVENT = "exact_event"
    CHARACTER_SETTING = "character_setting"
    QUEST = "quest"
    YOUTUBE = "youtube"
    FALLBACK = "fallback"
    LOCATION = "location"
    OBJECT = "object"


class Classification(StrEnum):
    EXACT_SCENE = "EXACT_SCENE"
    LIKELY_EXACT_SCENE = "LIKELY_EXACT_SCENE"
    RELATED_FOOTAGE = "RELATED_FOOTAGE"
    CONTEXTUAL = "CONTEXTUAL"


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------


class Word(BaseModel):
    """A single word with timestamps. Drives silence-aware segmentation (§9)."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    """A transcriber's own segment, before we re-segment semantically."""

    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str = "en"
    duration: float = 0.0
    text: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider: str = "unknown"
    model: str = ""

    def all_words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]


# --------------------------------------------------------------------------
# Project context (§7)
# --------------------------------------------------------------------------


class Work(BaseModel):
    title: str
    type: str = "unknown"


class EntityCorrection(BaseModel):
    """A mishearing and its resolution. Never applied blindly (§10)."""

    original: str
    resolved: str
    confidence: float = 0.0
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    method: str = "unknown"
    applied: bool = False

    @field_validator("confidence")
    @classmethod
    def _in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be within 0.0..1.0")
        return v


class ProjectContext(BaseModel):
    subject: str = ""
    franchise: str = ""
    era: str = ""
    characters: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)
    works: list[Work] = Field(default_factory=list)
    entity_corrections: list[EntityCorrection] = Field(default_factory=list)
    domain_packs: list[str] = Field(default_factory=list)
    provider: str = "unknown"
    #: Short qualifier appended to search queries. Additive to §7's example
    #: JSON, whose queries ("Darth Baras Dark Council SWTOR") show the
    #: abbreviation being used rather than the full subject title.
    search_tag: str = ""


# --------------------------------------------------------------------------
# Segments (§7)
# --------------------------------------------------------------------------


class Query(BaseModel):
    kind: QueryKind | str
    text: str


class SubShot(BaseModel):
    start: float
    end: float
    topic: str = ""


class Pick(BaseModel):
    """One chosen asset for a segment.

    §7's example carries ``kind``, ``path`` and ``rank``. The rest is additive:
    ``score`` and ``confidence`` feed §14's bands, ``slug`` names the file in
    ``selected/``, and ``use`` is what the review UI's USE THIS toggles -- it
    is the single switch that decides whether this asset is in the deliverable.
    """

    kind: Literal["image", "clip"]
    path: str
    rank: int = 1
    score: float = 0.0
    confidence: float = 0.0
    slug: str = ""
    #: Included in selected/. Set for the top image and any clip by default.
    use: bool = False
    #: True when a human chose this in the review UI rather than the
    #: pipeline picking it. §14's confidence bands govern what is copied
    #: *automatically*; an explicit choice is not automatic and is honoured
    #: regardless of band.
    user_set: bool = False
    favorite: bool = False
    rejected: bool = False
    note: str = ""


class Segment(BaseModel):
    """``segments/<dir>/segment.json``."""

    index: int
    start: float
    end: float
    duration: float = 0.0
    narration: str = ""
    topic: str = ""
    entities: list[str] = Field(default_factory=list)
    event: str = ""
    location: str = ""
    interpretation: str = ""
    visual_intent: VisualIntent = VisualIntent.B_ROLL
    queries: list[Query] = Field(default_factory=list)
    confidence: float = 0.0
    sub_shots: list[SubShot] = Field(default_factory=list)
    picks: list[Pick] = Field(default_factory=list)
    status: Literal["ok", "degraded", "failed"] = "ok"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_duration(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"segment {self.index}: end {self.end} precedes start {self.start}")
        object.__setattr__(self, "duration", round(self.end - self.start, 3))
        return self

    @property
    def dirname(self) -> str:
        from .utils.files import segment_dirname

        return segment_dirname(self.index, self.start, self.end)


# --------------------------------------------------------------------------
# Assets (§7)
# --------------------------------------------------------------------------


class ImageRecord(BaseModel):
    local_path: str = ""
    image_url: str = ""
    source_page: str = ""
    domain: str = ""
    provider: str = ""
    query: str = ""
    query_kind: str = ""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    bytes: int = 0
    sha256: str = ""
    phash: str = ""
    dhash: str = ""
    segment_index: int = -1
    rank: int = 0
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    creator: str = ""
    license: str = "unknown"
    license_url: str = ""
    status: str = "candidate"
    notes: str = ""

    @model_validator(mode="after")
    def _fill_aspect(self) -> ImageRecord:
        if self.height and not self.aspect_ratio:
            object.__setattr__(self, "aspect_ratio", round(self.width / self.height, 4))
        return self


class TimestampCandidate(BaseModel):
    start_s: float
    end_s: float
    method: str = "unknown"
    evidence: str = ""
    confidence: float = 0.0
    url_with_t: str = ""


class YouTubeRecord(BaseModel):
    title: str = ""
    url: str = ""
    video_id: str = ""
    channel: str = ""
    duration: float = 0.0
    thumbnail: str = ""
    query: str = ""
    reason: str = ""
    relevance: float = 0.0
    classification: Classification = Classification.CONTEXTUAL
    timestamp_candidates: list[TimestampCandidate] = Field(default_factory=list)
    downloaded_clip_path: str = ""

    def best_timestamp(self) -> TimestampCandidate | None:
        if not self.timestamp_candidates:
            return None
        return max(self.timestamp_candidates, key=lambda c: c.confidence)


def dump_json(model: BaseModel) -> str:
    """Stable, human-diffable JSON for everything we write to disk."""
    return model.model_dump_json(indent=2)


def dump_any(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
