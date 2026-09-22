"""Job queue (CLAUDE.md §8).

SQLite is the queue. That is deliberate: the CLI, the worker, the web UI and
the folder watcher all have to agree on job state across processes and across
restarts, and the database is already there. No broker (§4 forbids one).

Claiming is a conditional UPDATE, so two workers racing for the same job
cannot both win.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import update
from sqlmodel import select

from ..db import session_scope
from ..logging_setup import get_logger
from ..models import Job, SeenFile, StageCheckpoint, utcnow
from .states import STAGE_ORDER, JobState, Stage

__all__ = [
    "enqueue",
    "get_job",
    "list_jobs",
    "claim_next",
    "set_state",
    "record_checkpoint",
    "completed_stages",
    "stage_timings",
    "resume_point",
    "clear_checkpoints",
    "heartbeat",
    "claim_filename",
    "release_filename",
    "discard_filename",
    "JobNotFound",
]

log = get_logger("jobs.queue")

#: A job whose worker has not checked in for this long is considered dead.
STALE_AFTER = timedelta(minutes=30)


class JobNotFound(LookupError):
    pass


def enqueue(
    db_path: Path,
    *,
    project: str,
    source_path: Path,
    no_clips: bool = False,
    force: bool = False,
    origin: str = "cli",
    origin_filename: str = "",
    job_id: str | None = None,
) -> Job:
    """Create a QUEUED job.

    Passing an existing ``job_id`` re-queues that job rather than creating a
    duplicate -- the "id reuse doesn't duplicate" rule from §21.
    """
    if not project or not project.strip():
        raise ValueError("job rejected: project name is empty")
    if not str(source_path).strip():
        raise ValueError("job rejected: source_path is empty")

    with session_scope(db_path) as session:
        if job_id:
            existing = session.get(Job, job_id)
            if existing is not None:
                existing.state = JobState.QUEUED
                existing.error = ""
                existing.finished_at = None
                session.add(existing)
                session.flush()
                session.refresh(existing)
                log.info("re-queued existing job %s (%s)", existing.id, existing.project)
                return _detach(existing)

        job = Job(
            project=project,
            source_path=str(source_path),
            no_clips=no_clips,
            force=force,
            origin=origin,
            origin_filename=origin_filename,
        )
        if job_id:
            job.id = job_id
        session.add(job)
        session.flush()
        session.refresh(job)
        log.info("queued job %s for project %r", job.id, job.project)
        return _detach(job)


def _detach(job: Job) -> Job:
    """Return a plain copy that stays usable after the session closes."""
    return Job.model_validate(job.model_dump())


def get_job(db_path: Path, job_id: str) -> Job:
    with session_scope(db_path) as session:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFound(f"no job with id {job_id!r}")
        return _detach(job)


def list_jobs(db_path: Path, *, limit: int = 50, state: JobState | None = None) -> list[Job]:
    with session_scope(db_path) as session:
        stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
        if state is not None:
            stmt = stmt.where(Job.state == state)
        return [_detach(j) for j in session.exec(stmt).all()]


def claim_next(db_path: Path, *, worker_id: str) -> Job | None:
    """Atomically take the oldest QUEUED job, or None if there is nothing to do."""
    with session_scope(db_path) as session:
        stmt = select(Job).where(Job.state == JobState.QUEUED).order_by(Job.created_at)  # type: ignore[arg-type]
        for candidate in session.exec(stmt).all():
            # Conditional update: only the worker that flips QUEUED wins the job.
            result = session.exec(
                update(Job)
                .where(Job.id == candidate.id, Job.state == JobState.QUEUED)  # type: ignore[arg-type]
                .values(
                    state=JobState.TRANSCRIBING,
                    started_at=utcnow(),
                    heartbeat_at=utcnow(),
                    meta={**(candidate.meta or {}), "worker": worker_id},
                )
            )
            if result.rowcount:
                session.flush()
                job = session.get(Job, candidate.id)
                log.info("worker %s claimed job %s", worker_id, candidate.id)
                return _detach(job)  # type: ignore[arg-type]
        return None


def set_state(
    db_path: Path,
    job_id: str,
    state: JobState,
    *,
    stage: str | None = None,
    error: str = "",
    **fields,
) -> Job:
    from .states import TERMINAL_STATES

    with session_scope(db_path) as session:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFound(f"no job with id {job_id!r}")
        job.state = state
        if stage is not None:
            job.stage = stage
        if error:
            job.error = error
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.heartbeat_at = utcnow()
        if state in TERMINAL_STATES:
            job.finished_at = utcnow()
        session.add(job)
        session.flush()
        session.refresh(job)
        return _detach(job)


def heartbeat(db_path: Path, job_id: str) -> None:
    with session_scope(db_path) as session:
        job = session.get(Job, job_id)
        if job is not None:
            job.heartbeat_at = utcnow()
            session.add(job)


def record_checkpoint(
    db_path: Path,
    job_id: str,
    stage: Stage | str,
    *,
    ok: bool = True,
    artifact: str = "",
    detail: str = "",
    duration_s: float = 0.0,
) -> None:
    """Mark a stage finished so ``resume`` can skip it (§8, §20)."""
    stage_name = str(stage)
    with session_scope(db_path) as session:
        existing = session.exec(
            select(StageCheckpoint).where(
                StageCheckpoint.job_id == job_id, StageCheckpoint.stage == stage_name
            )
        ).first()
        if existing is not None:
            existing.ok = ok
            existing.artifact = artifact
            existing.detail = detail
            existing.duration_s = duration_s
            existing.created_at = utcnow()
            session.add(existing)
            return
        session.add(
            StageCheckpoint(
                job_id=job_id,
                stage=stage_name,
                ok=ok,
                artifact=artifact,
                detail=detail,
                duration_s=duration_s,
            )
        )


def completed_stages(db_path: Path, job_id: str) -> set[str]:
    with session_scope(db_path) as session:
        rows = session.exec(
            select(StageCheckpoint).where(
                StageCheckpoint.job_id == job_id,
                StageCheckpoint.ok == True,  # noqa: E712
            )
        ).all()
        return {r.stage for r in rows}


def resume_point(db_path: Path, job_id: str) -> Stage:
    """First stage with no successful checkpoint (§8)."""
    done = completed_stages(db_path, job_id)
    for stage in STAGE_ORDER:
        if str(stage) not in done:
            return stage
    return STAGE_ORDER[-1]


def clear_checkpoints(db_path: Path, job_id: str, *, stages: list[str] | None = None) -> int:
    """Drop checkpoints so ``--force`` or ``rerun`` redoes those stages."""
    with session_scope(db_path) as session:
        stmt = select(StageCheckpoint).where(StageCheckpoint.job_id == job_id)
        rows = [r for r in session.exec(stmt).all() if stages is None or r.stage in stages]
        for row in rows:
            session.delete(row)
        return len(rows)


def stale_jobs(db_path: Path) -> list[Job]:
    """Jobs whose worker died mid-stage. ``resume`` is the cure."""
    cutoff = datetime.now(UTC) - STALE_AFTER
    out: list[Job] = []
    for job in list_jobs(db_path, limit=500):
        if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.REVIEW_REQUIRED}:
            continue
        beat = job.heartbeat_at
        if beat is None:
            continue
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=UTC)
        if beat < cutoff:
            out.append(job)
    return out


# ---------------------------------------------------------------------------
# Watcher ledger (§18.6)
# ---------------------------------------------------------------------------


def claim_filename(db_path: Path, filename: str, *, source_path: Path, size: int) -> bool:
    """Reserve ``filename`` for processing. False when it was already claimed.

    The unique index on ``filename`` is what makes this safe: a second watcher
    (or a restarted one) trying to claim the same drop loses on the insert.
    """
    key = filename.lower()
    with session_scope(db_path) as session:
        existing = session.exec(select(SeenFile).where(SeenFile.filename == key)).first()
        if existing is not None:
            return False
        session.add(
            SeenFile(filename=key, source_path=str(source_path), size=size, status="claimed")
        )
        try:
            session.flush()
        except Exception:  # pragma: no cover - concurrent claim
            session.rollback()
            return False
        return True


def release_filename(
    db_path: Path, filename: str, *, job_id: str = "", project: str = "", status: str = "processed"
) -> None:
    key = filename.lower()
    with session_scope(db_path) as session:
        row = session.exec(select(SeenFile).where(SeenFile.filename == key)).first()
        if row is None:
            return
        row.job_id = job_id
        row.project = project
        row.status = status
        session.add(row)


def discard_filename(db_path: Path, filename: str) -> bool:
    """Forget a claim entirely, so the file can be taken again.

    Only for a claim that was never turned into work: the watcher claims
    *before* moving (§18.6), and a move that fails must not blacklist the file
    forever. A transient disk-full or file-locked error would otherwise mean
    the user's narration is silently never processed -- the worst kind of
    failure, because nothing reports it.

    A successfully processed file is released with :func:`release_filename`
    instead, which keeps the row and therefore keeps the guarantee.
    """
    key = filename.lower()
    with session_scope(db_path) as session:
        row = session.exec(select(SeenFile).where(SeenFile.filename == key)).first()
        if row is None:
            return False
        session.delete(row)
        log.info("discarded the claim on %s so it can be retried", filename)
        return True


def stage_timings(db_path: Path, job_id: str) -> list[tuple[str, float, str]]:
    """``(stage, seconds, detail)`` for every checkpoint, in run order.

    The durations are already recorded per stage; this just reads them back so
    the report can show where a long run actually spent its time (§22 P8).
    """
    order = {str(stage): index for index, stage in enumerate(STAGE_ORDER)}
    with session_scope(db_path) as session:
        rows = session.exec(select(StageCheckpoint).where(StageCheckpoint.job_id == job_id)).all()
        # Read every attribute inside the session. A SQLModel row detaches
        # when the session closes, and touching it afterwards raises rather
        # than returning the value it already had.
        out = [(r.stage, r.duration_s, r.detail) for r in rows]
    return sorted(out, key=lambda item: order.get(item[0], 999))
