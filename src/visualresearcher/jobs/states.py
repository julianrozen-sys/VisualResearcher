"""Job lifecycle (CLAUDE.md §8).

The stage list is the pipeline's spine: ``resume`` replays from the first
stage that has no checkpoint, so the order here is authoritative.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["JobState", "Stage", "STAGE_ORDER", "STAGE_STATE", "TERMINAL_STATES", "next_stage"]


class JobState(StrEnum):
    QUEUED = "QUEUED"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    SEARCHING_IMAGES = "SEARCHING_IMAGES"
    RANKING = "RANKING"
    SEARCHING_VIDEO = "SEARCHING_VIDEO"
    DOWNLOADING_CLIPS = "DOWNLOADING_CLIPS"
    COLLECTING = "COLLECTING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Stage(StrEnum):
    INGEST = "ingest"
    TRANSCRIBE = "transcribe"
    CONTEXT = "context"
    ENTITIES = "entities"
    SEGMENT = "segment"
    QUERIES = "queries"
    IMAGE_SEARCH = "image_search"
    DOWNLOAD = "download"
    DEDUPE = "dedupe"
    RANK = "rank"
    VIDEO_SEARCH = "video_search"
    TIMESTAMPS = "timestamps"
    CLIPS = "clips"
    COLLECT = "collect"
    REPORT = "report"


#: Execution order. Index in this list is also the resume position.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.INGEST,
    Stage.TRANSCRIBE,
    Stage.CONTEXT,
    Stage.ENTITIES,
    Stage.SEGMENT,
    Stage.QUERIES,
    Stage.IMAGE_SEARCH,
    Stage.DOWNLOAD,
    Stage.DEDUPE,
    Stage.RANK,
    Stage.VIDEO_SEARCH,
    Stage.TIMESTAMPS,
    Stage.CLIPS,
    Stage.COLLECT,
    Stage.REPORT,
)

#: The state a job reports while each stage runs (§8).
#:
#: **No stage may map to QUEUED.** QUEUED means "nobody is working on this",
#: and `claim_next` hands out anything in that state. A running stage that
#: reported QUEUED made its own job claimable again, so a second worker picked
#: it up and two threads wrote the same project folder at once. Ingest is
#: therefore part of the TRANSCRIBING phase, which is what it is: the work
#: that gets us to a transcript.
STAGE_STATE: dict[Stage, JobState] = {
    Stage.INGEST: JobState.TRANSCRIBING,
    Stage.TRANSCRIBE: JobState.TRANSCRIBING,
    Stage.CONTEXT: JobState.ANALYZING,
    Stage.ENTITIES: JobState.ANALYZING,
    Stage.SEGMENT: JobState.ANALYZING,
    Stage.QUERIES: JobState.ANALYZING,
    Stage.IMAGE_SEARCH: JobState.SEARCHING_IMAGES,
    Stage.DOWNLOAD: JobState.SEARCHING_IMAGES,
    Stage.DEDUPE: JobState.RANKING,
    Stage.RANK: JobState.RANKING,
    Stage.VIDEO_SEARCH: JobState.SEARCHING_VIDEO,
    Stage.TIMESTAMPS: JobState.SEARCHING_VIDEO,
    Stage.CLIPS: JobState.DOWNLOADING_CLIPS,
    Stage.COLLECT: JobState.COLLECTING,
    Stage.REPORT: JobState.COLLECTING,
}

TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.COMPLETE, JobState.FAILED, JobState.REVIEW_REQUIRED}
)


#: States that mean "available to claim". Nothing a running job reports may
#: appear here (see STAGE_STATE).
CLAIMABLE_STATES: frozenset[JobState] = frozenset({JobState.QUEUED})


def next_stage(stage: Stage) -> Stage | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None
