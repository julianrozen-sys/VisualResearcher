"""Job queue tests (CLAUDE.md §21).

"id reuse doesn't duplicate; malformed job rejected", plus the claiming and
checkpoint behaviour that ``resume`` depends on.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from visualresearcher.db import get_engine
from visualresearcher.jobs.queue import (
    JobNotFound,
    claim_filename,
    claim_next,
    clear_checkpoints,
    completed_stages,
    enqueue,
    get_job,
    list_jobs,
    record_checkpoint,
    release_filename,
    resume_point,
    set_state,
)
from visualresearcher.jobs.states import STAGE_ORDER, JobState, Stage
from visualresearcher.models import Job


def _count_jobs(db_path) -> int:
    with Session(get_engine(db_path)) as session:
        return len(session.exec(select(Job)).all())


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_enqueue_creates_one_job(db_path, tmp_path):
    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    assert job.state == JobState.QUEUED
    assert job.project == "demo"
    assert _count_jobs(db_path) == 1


def test_reusing_a_job_id_does_not_duplicate(db_path, tmp_path):
    """§21: id reuse doesn't duplicate."""
    first = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    set_state(db_path, first.id, JobState.FAILED, error="boom")

    second = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav", job_id=first.id)
    assert second.id == first.id
    assert _count_jobs(db_path) == 1, "re-queueing must not create a second row"
    assert second.state == JobState.QUEUED, "re-queueing must reset the state"
    assert second.error == "", "re-queueing must clear the previous error"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"project": "", "source_path": "x.wav"},
        {"project": "   ", "source_path": "x.wav"},
        {"project": "demo", "source_path": ""},
    ],
)
def test_malformed_job_is_rejected(db_path, kwargs):
    """§21: malformed job rejected."""
    with pytest.raises(ValueError):
        enqueue(db_path, **kwargs)
    assert _count_jobs(db_path) == 0, "a rejected job must leave no row behind"


def test_get_job_raises_for_unknown_id(db_path):
    with pytest.raises(JobNotFound):
        get_job(db_path, "does-not-exist")


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------


def test_claim_next_returns_the_oldest_queued_job(db_path, tmp_path):
    first = enqueue(db_path, project="one", source_path=tmp_path / "1.wav")
    enqueue(db_path, project="two", source_path=tmp_path / "2.wav")
    claimed = claim_next(db_path, worker_id="w1")
    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.state != JobState.QUEUED


def test_a_job_can_only_be_claimed_once(db_path, tmp_path):
    """Two workers must not both get the same job (§19)."""
    enqueue(db_path, project="only", source_path=tmp_path / "1.wav")
    first = claim_next(db_path, worker_id="w1")
    second = claim_next(db_path, worker_id="w2")
    assert first is not None
    assert second is None, "the second worker claimed an already-claimed job"


def test_claim_next_returns_none_when_idle(db_path):
    assert claim_next(db_path, worker_id="w1") is None


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


def test_resume_point_is_the_first_unfinished_stage(db_path, tmp_path):
    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    assert resume_point(db_path, job.id) == STAGE_ORDER[0]

    record_checkpoint(db_path, job.id, Stage.INGEST)
    record_checkpoint(db_path, job.id, Stage.TRANSCRIBE)
    assert resume_point(db_path, job.id) == Stage.CONTEXT


def test_a_failed_checkpoint_does_not_count_as_done(db_path, tmp_path):
    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    record_checkpoint(db_path, job.id, Stage.INGEST)
    record_checkpoint(db_path, job.id, Stage.TRANSCRIBE, ok=False, detail="provider blew up")
    assert resume_point(db_path, job.id) == Stage.TRANSCRIBE
    assert str(Stage.TRANSCRIBE) not in completed_stages(db_path, job.id)


def test_recording_a_checkpoint_twice_updates_rather_than_duplicates(db_path, tmp_path):
    from visualresearcher.models import StageCheckpoint

    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    record_checkpoint(db_path, job.id, Stage.INGEST, detail="first")
    record_checkpoint(db_path, job.id, Stage.INGEST, detail="second")
    with Session(get_engine(db_path)) as session:
        rows = session.exec(select(StageCheckpoint).where(StageCheckpoint.job_id == job.id)).all()
    assert len(rows) == 1
    assert rows[0].detail == "second"


def test_clear_checkpoints_lets_a_stage_run_again(db_path, tmp_path):
    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    record_checkpoint(db_path, job.id, Stage.INGEST)
    record_checkpoint(db_path, job.id, Stage.TRANSCRIBE)
    removed = clear_checkpoints(db_path, job.id, stages=[str(Stage.TRANSCRIBE)])
    assert removed == 1
    assert resume_point(db_path, job.id) == Stage.TRANSCRIBE


def test_terminal_state_records_a_finish_time(db_path, tmp_path):
    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    updated = set_state(db_path, job.id, JobState.COMPLETE)
    assert updated.finished_at is not None


def test_list_jobs_filters_by_state(db_path, tmp_path):
    a = enqueue(db_path, project="a", source_path=tmp_path / "a.wav")
    enqueue(db_path, project="b", source_path=tmp_path / "b.wav")
    set_state(db_path, a.id, JobState.COMPLETE)
    assert [j.id for j in list_jobs(db_path, state=JobState.COMPLETE)] == [a.id]
    assert len(list_jobs(db_path)) == 2


# ---------------------------------------------------------------------------
# Watcher ledger (§18.6) -- the watcher itself lands in P8, the ledger it
# relies on is built and tested now.
# ---------------------------------------------------------------------------


def test_a_filename_can_only_be_claimed_once(db_path, tmp_path):
    src = tmp_path / "Tight_episode.wav"
    assert claim_filename(db_path, "Tight_episode.wav", source_path=src, size=10) is True
    assert claim_filename(db_path, "Tight_episode.wav", source_path=src, size=10) is False


def test_filename_claiming_is_case_insensitive(db_path, tmp_path):
    """§18.2 matches the prefix case-insensitively, so the ledger must agree."""
    src = tmp_path / "Tight_episode.wav"
    assert claim_filename(db_path, "Tight_Episode.WAV", source_path=src, size=10) is True
    assert claim_filename(db_path, "tight_episode.wav", source_path=src, size=10) is False


def test_releasing_a_filename_keeps_it_claimed(db_path, tmp_path):
    """Processing a file must not make it eligible again."""
    src = tmp_path / "Tight_x.wav"
    claim_filename(db_path, "Tight_x.wav", source_path=src, size=1)
    release_filename(db_path, "Tight_x.wav", job_id="abc", project="tight_x")
    assert claim_filename(db_path, "Tight_x.wav", source_path=src, size=1) is False


def test_a_discarded_claim_can_be_taken_again(db_path, tmp_path):
    """A claim that never became work must not blacklist the file forever.

    The watcher claims before moving (§18.6). If the move fails, the file has
    to be retryable, or a transient disk error means the user's narration is
    silently never processed.
    """
    from visualresearcher.jobs.queue import discard_filename

    src = tmp_path / "Tight_x.wav"
    assert claim_filename(db_path, "Tight_x.wav", source_path=src, size=1) is True
    assert discard_filename(db_path, "Tight_x.wav") is True
    assert claim_filename(db_path, "Tight_x.wav", source_path=src, size=1) is True


def test_discarding_an_unknown_filename_is_harmless(db_path):
    from visualresearcher.jobs.queue import discard_filename

    assert discard_filename(db_path, "never-seen.wav") is False


def test_a_processed_file_is_still_never_retaken(db_path, tmp_path):
    """release_filename keeps the row, which is what §18.6 guarantees."""
    src = tmp_path / "Tight_done.wav"
    claim_filename(db_path, "Tight_done.wav", source_path=src, size=1)
    release_filename(db_path, "Tight_done.wav", job_id="j", project="p", status="processed")
    assert claim_filename(db_path, "Tight_done.wav", source_path=src, size=1) is False


# ---------------------------------------------------------------------------
# The invariant that makes concurrent running safe (§19)
# ---------------------------------------------------------------------------


def test_no_running_stage_reports_a_claimable_state():
    """A running job must never look available to another worker.

    `Stage.INGEST` used to map to QUEUED. Single-threaded that was harmless;
    with a worker pool it meant the first stage put its own job back in the
    queue, a second worker claimed it, and two threads wrote the same project
    folder at once.
    """
    from visualresearcher.jobs.states import CLAIMABLE_STATES, STAGE_ORDER, STAGE_STATE

    offenders = {
        str(stage): str(state) for stage, state in STAGE_STATE.items() if state in CLAIMABLE_STATES
    }
    assert not offenders, f"these stages report a claimable state while running: {offenders}"
    assert set(STAGE_STATE) == set(STAGE_ORDER), "every stage needs a running state"


def test_starting_a_stage_does_not_make_the_job_claimable(db_path, tmp_path):
    """The same invariant, exercised through the real calls."""
    from visualresearcher.jobs.states import STAGE_ORDER, STAGE_STATE

    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    assert claim_next(db_path, worker_id="w1") is not None

    for stage in STAGE_ORDER:
        set_state(db_path, job.id, STAGE_STATE[stage], stage=str(stage))
        assert claim_next(db_path, worker_id="w2") is None, (
            f"the job became claimable again while {stage} was running"
        )


def test_stage_timings_survive_the_session_closing(db_path, tmp_path):
    """Reading a SQLModel row after its session closes raises, so don't.

    `stage_timings` used to return the rows and let the caller read them,
    which blew up with DetachedInstanceError at the end of a real run -- after
    all the work was done, in the very last stage.
    """
    from visualresearcher.jobs.queue import stage_timings

    job = enqueue(db_path, project="demo", source_path=tmp_path / "a.wav")
    record_checkpoint(db_path, job.id, Stage.TRANSCRIBE, duration_s=2.5, detail="180 words")
    record_checkpoint(db_path, job.id, Stage.INGEST, duration_s=0.1, detail="60s of audio")

    timings = stage_timings(db_path, job.id)
    # Ordered by STAGE_ORDER, not by insertion.
    assert [t[0] for t in timings] == ["ingest", "transcribe"]
    assert timings[1][1] == 2.5
    assert timings[0][2] == "60s of audio"


def test_stage_timings_of_an_unknown_job_is_empty(db_path):
    from visualresearcher.jobs.queue import stage_timings

    assert stage_timings(db_path, "nope") == []


# ---------------------------------------------------------------------------
# Heartbeats during a long stage
#
# Caught when a reader saw two jobs on one project in `status` and could not
# tell which was alive: a killed worker leaves its running state in the table
# forever. Flagging staleness only works if a live job actually beats, and the
# heartbeat used to fire once per *stage* -- so `image_search` over a
# 20-minute narration went quiet for longer than STALE_AFTER while perfectly
# healthy.
# ---------------------------------------------------------------------------


def test_a_long_stage_keeps_beating(db_path, tmp_path, monkeypatch):
    """The beat must be on a clock, not on stage boundaries."""
    import time as _time

    from visualresearcher.jobs import worker as worker_mod
    from visualresearcher.jobs.queue import enqueue, get_job

    monkeypatch.setattr(worker_mod, "HEARTBEAT_EVERY_S", 0.05)
    job = enqueue(db_path, project="beat", source_path=tmp_path / "a.wav")
    before = get_job(db_path, job.id).heartbeat_at

    with worker_mod._Heartbeat(db_path, job.id):
        _time.sleep(0.4)  # stand-in for a stage that takes minutes

    after = get_job(db_path, job.id).heartbeat_at
    assert after is not None
    assert before is None or after > before, (
        "the heartbeat did not advance during a long stage; a healthy job "
        "would be reported stale"
    )


def test_the_ticker_stops_when_the_job_does(db_path, tmp_path, monkeypatch):
    import time as _time

    from visualresearcher.jobs import worker as worker_mod
    from visualresearcher.jobs.queue import enqueue, get_job

    monkeypatch.setattr(worker_mod, "HEARTBEAT_EVERY_S", 0.05)
    job = enqueue(db_path, project="beat", source_path=tmp_path / "a.wav")
    with worker_mod._Heartbeat(db_path, job.id):
        _time.sleep(0.2)
    settled = get_job(db_path, job.id).heartbeat_at
    _time.sleep(0.3)
    assert get_job(db_path, job.id).heartbeat_at == settled, (
        "a finished job is still beating; it would never be reported stale"
    )


def test_a_dead_job_is_reported_stale(db_path, tmp_path, monkeypatch):
    """`stale_jobs` existed but nothing called it, so corpses looked alive."""
    from datetime import UTC, datetime, timedelta

    from visualresearcher.jobs import queue as queue_mod
    from visualresearcher.jobs.queue import enqueue, stale_jobs
    from visualresearcher.models import Job

    job = enqueue(db_path, project="corpse", source_path=tmp_path / "a.wav")
    with queue_mod.session_scope(db_path) as session:
        row = session.get(Job, job.id)
        row.state = "SEARCHING_IMAGES"
        row.heartbeat_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
        session.add(row)

    assert job.id in {j.id for j in stale_jobs(db_path)}
