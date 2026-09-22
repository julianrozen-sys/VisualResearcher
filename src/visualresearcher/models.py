"""SQLite tables (SQLModel).

Three concerns live here: the job queue, the provider cache (§20), and the
watcher's seen-file ledger (§18.6 — "never process the same filename twice",
which has to survive a restart, so it is a table and not a set in memory).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import JSON, Column, Field, SQLModel

from .jobs.states import JobState

__all__ = ["Job", "StageCheckpoint", "CacheEntry", "SeenFile", "utcnow", "new_job_id"]


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    """Short, sortable-enough, collision-free job id."""
    return uuid.uuid4().hex[:12]


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(default_factory=new_job_id, primary_key=True)
    project: str = Field(index=True)
    source_path: str = ""
    audio_path: str = ""
    state: JobState = Field(default=JobState.QUEUED, index=True)
    stage: str = ""
    error: str = ""
    # Provenance for the watcher; empty for CLI-started jobs.
    origin: str = "cli"
    origin_filename: str = ""
    no_clips: bool = False
    force: bool = False
    segments_total: int = 0
    segments_done: int = 0
    gaps: int = 0
    bytes_used: int = 0
    created_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    @property
    def duration_s(self) -> float | None:
        if not self.started_at:
            return None
        end = self.finished_at or utcnow()
        start = self.started_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return (end - start).total_seconds()


class StageCheckpoint(SQLModel, table=True):
    """One row per completed stage, so ``resume`` knows where to pick up (§8)."""

    __tablename__ = "stage_checkpoints"
    __table_args__ = (UniqueConstraint("job_id", "stage", name="uq_checkpoint_job_stage"),)

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, foreign_key="jobs.id")
    stage: str = Field(index=True)
    ok: bool = True
    artifact: str = ""
    detail: str = ""
    duration_s: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)


class CacheEntry(SQLModel, table=True):
    """Provider response cache keyed by ``sha256(provider+endpoint+params)`` (§20)."""

    __tablename__ = "cache_entries"

    key: str = Field(primary_key=True)
    provider: str = Field(index=True)
    endpoint: str = ""
    value: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    expires_at: datetime | None = Field(default=None, index=True)
    hits: int = 0


class SeenFile(SQLModel, table=True):
    """Watcher ledger. A filename here is never processed a second time (§18.6)."""

    __tablename__ = "seen_files"

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(index=True, unique=True)
    source_path: str = ""
    size: int = 0
    job_id: str = ""
    project: str = ""
    status: str = "claimed"
    created_at: datetime = Field(default_factory=utcnow)
