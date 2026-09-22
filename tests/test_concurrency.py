"""Concurrency tests (CLAUDE.md §19, §21).

§21 asks for three things:

* two simultaneous jobs do not write into each other's folders,
* CPU-heavy stages serialise,
* the disk guard sums across jobs.

The last one is the subtle one. Two jobs that each individually fit on the
remaining space can still fill the drive between them, and a per-job check
passes both times on the way to doing exactly that.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from visualresearcher.jobs.limits import (
    ComputeLimiter,
    DiskArbiter,
    DownloadBudget,
    Limits,
    get_limits,
)
from visualresearcher.jobs.pool import WorkerPool
from visualresearcher.jobs.queue import enqueue, list_jobs
from visualresearcher.jobs.states import JobState
from visualresearcher.utils.disk import DiskSpaceError, free_gb

# ---------------------------------------------------------------------------
# §21: CPU-heavy stages serialise
# ---------------------------------------------------------------------------


def test_only_one_cpu_heavy_stage_runs_at_a_time():
    """§19: without a GPU they do not parallelise."""
    limiter = ComputeLimiter(max_concurrent=1)
    concurrent = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal concurrent, peak
        with limiter.slot("test"):
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak == 1, f"{peak} CPU-heavy stages ran at once; the semaphore is not holding"
    assert limiter.active == 0, "a slot leaked"


def test_the_limit_is_configurable():
    limiter = ComputeLimiter(max_concurrent=2)
    peak = 0
    concurrent = 0
    lock = threading.Lock()

    def work():
        nonlocal concurrent, peak
        with limiter.slot():
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert peak <= 2


def test_an_exception_does_not_leak_a_slot():
    """A leaked slot wedges every other job behind it, forever."""
    limiter = ComputeLimiter(max_concurrent=1)
    with pytest.raises(RuntimeError):
        with limiter.slot("boom"):
            raise RuntimeError("stage blew up")
    assert limiter.active == 0
    # The slot must still be obtainable.
    assert limiter.acquire(timeout=1) is True
    limiter.release()


def test_zero_or_negative_concurrency_is_clamped_to_one():
    assert ComputeLimiter(0).max_concurrent == 1
    assert ComputeLimiter(-5).max_concurrent == 1


# ---------------------------------------------------------------------------
# §21: the disk guard sums across jobs
# ---------------------------------------------------------------------------


def test_the_disk_guard_counts_other_running_jobs(sandbox):
    """§19: two jobs that each fit can still fill the drive together."""
    available = free_gb(sandbox)
    arbiter = DiskArbiter(min_free_gb=1.0)

    # On its own, this job fits comfortably.
    arbiter.check(sandbox, job_id="a", stage="download")

    # Another job reserves almost everything that is left.
    arbiter.reserve("b", available - 0.5)

    with pytest.raises(DiskSpaceError) as excinfo:
        arbiter.check(sandbox, job_id="a", stage="download")

    message = str(excinfo.value)
    assert "reserved by" in message, "the error must explain that another job is holding space"
    assert "other running job" in message
    assert "download" in message


def test_a_jobs_own_reservation_does_not_block_itself(sandbox):
    arbiter = DiskArbiter(min_free_gb=1.0)
    arbiter.reserve("a", free_gb(sandbox) * 2)
    # Its own reservation must not count against it, or it could never proceed.
    arbiter.check(sandbox, job_id="a", stage="download")


def test_releasing_frees_the_reservation(sandbox):
    available = free_gb(sandbox)
    arbiter = DiskArbiter(min_free_gb=1.0)
    arbiter.reserve("b", available - 0.5)
    with pytest.raises(DiskSpaceError):
        arbiter.check(sandbox, job_id="a")

    arbiter.release("b")
    arbiter.check(sandbox, job_id="a")
    assert arbiter.reserved_gb == 0


def test_reservations_are_summed_not_maxed(sandbox):
    available = free_gb(sandbox)
    arbiter = DiskArbiter(min_free_gb=1.0)
    share = (available - 0.5) / 3
    for job in ("b", "c", "d"):
        arbiter.reserve(job, share)

    assert arbiter.reserved_gb == pytest.approx(share * 3)
    with pytest.raises(DiskSpaceError):
        arbiter.check(sandbox, job_id="a")


# ---------------------------------------------------------------------------
# §19: the download budget is shared, not per job
# ---------------------------------------------------------------------------


def test_the_download_budget_is_shared_across_jobs():
    budget = DownloadBudget(max_bytes=1000)
    assert budget.can_spend(600) is True
    budget.spend(600)  # job A
    assert budget.remaining == 400
    assert budget.can_spend(600) is False, (
        "a second job should see the first job's spend; a per-job budget would not"
    )
    budget.spend(400)  # job B
    assert budget.remaining == 0


def test_an_unlimited_budget_never_refuses():
    budget = DownloadBudget(max_bytes=0)
    assert budget.can_spend(1 << 40) is True
    budget.spend(1 << 40)
    assert budget.can_spend(1 << 40) is True


def test_the_rate_limit_is_shared():
    """§19: it is the remote host's view of the request rate that matters."""
    budget = DownloadBudget(min_interval_s=0.05)
    budget.throttle()
    started = time.monotonic()
    budget.throttle()
    budget.throttle()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.08, (
        f"three throttled requests took {elapsed:.3f}s; the shared interval is not holding"
    )


def test_budget_spending_is_thread_safe():
    budget = DownloadBudget(max_bytes=0)

    def spend():
        for _ in range(200):
            budget.spend(1)

    threads = [threading.Thread(target=spend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert budget.spent == 1600, "a lost update means the budget under-counts"


def test_limits_are_built_from_settings(settings):
    settings.compute.max_concurrent_cpu_heavy = 3
    limits = Limits.from_settings(settings)
    assert limits.compute.max_concurrent == 3
    assert limits.disk.min_free_gb == settings.output.min_free_gb


def test_get_limits_returns_one_set_per_process(settings):
    first = get_limits(settings, refresh=True)
    second = get_limits(settings)
    assert first is second, "every job must share the same limiters"


# ---------------------------------------------------------------------------
# §21: two simultaneous jobs do not write into each other's folders
# ---------------------------------------------------------------------------


#: A two-job run takes about 50s on this machine. 300s is generous headroom
#: while still failing fast: a pool that has genuinely wedged should surface
#: as a failed test in minutes, not sit there looking like slow progress.
DRAIN_TIMEOUT = 300.0


def _collect_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}


def test_two_simultaneous_jobs_keep_their_output_separate(sample_wav, settings, db_path):
    """§21's requirement, run for real through the pool."""
    settings.worker.max_concurrent = 2
    for name in ("alpha", "beta"):
        enqueue(db_path, project=name, source_path=sample_wav)

    pool = WorkerPool(settings, db_path)
    outcomes = pool.drain(timeout=DRAIN_TIMEOUT)

    assert len(outcomes) == 2, f"expected two jobs to finish, got {len(outcomes)}"
    for outcome in outcomes:
        assert outcome.error == "", outcome.error

    alpha = settings.project_dir("alpha")
    beta = settings.project_dir("beta")
    assert alpha.exists() and beta.exists()

    # Neither project may contain a path belonging to the other.
    for root, other in ((alpha, "beta"), (beta, "alpha")):
        for path in root.rglob("*"):
            assert other not in str(path), f"{path} leaked into the wrong project"

    # Both produced a real, independent deliverable.
    alpha_files = _collect_files(alpha)
    beta_files = _collect_files(beta)
    assert alpha_files and beta_files
    assert any(f.startswith("selected/") for f in alpha_files)
    assert any(f.startswith("selected/") for f in beta_files)


def test_both_jobs_reach_a_terminal_state(sample_wav, settings, db_path):
    settings.worker.max_concurrent = 2
    for name in ("one", "two"):
        enqueue(db_path, project=name, source_path=sample_wav)

    WorkerPool(settings, db_path).drain(timeout=DRAIN_TIMEOUT)

    states = {j.project: j.state for j in list_jobs(db_path)}
    assert set(states) == {"one", "two"}
    for project, state in states.items():
        assert state in {JobState.COMPLETE, JobState.REVIEW_REQUIRED}, f"{project} ended {state}"


def test_a_job_that_crashes_does_not_stop_the_other(sample_wav, settings, db_path, monkeypatch):
    """§8: one failure must not take the pool down with it."""
    from visualresearcher.jobs import worker as worker_mod

    original = worker_mod.IMPLEMENTED_STAGES[worker_mod.Stage.SEGMENT]

    def explode_for_bad(ctx):
        if ctx.project == "bad":
            raise RuntimeError("segmentation exploded")
        return original(ctx)

    monkeypatch.setitem(worker_mod.IMPLEMENTED_STAGES, worker_mod.Stage.SEGMENT, explode_for_bad)

    settings.worker.max_concurrent = 2
    enqueue(db_path, project="bad", source_path=sample_wav)
    enqueue(db_path, project="good", source_path=sample_wav)

    WorkerPool(settings, db_path).drain(timeout=DRAIN_TIMEOUT)

    states = {j.project: j.state for j in list_jobs(db_path)}
    assert states["bad"] == JobState.FAILED
    assert states["good"] in {JobState.COMPLETE, JobState.REVIEW_REQUIRED}, (
        "the healthy job was taken down with the failing one"
    )


def test_the_pool_reports_what_it_is_running(sample_wav, settings, db_path):
    settings.worker.max_concurrent = 1
    enqueue(db_path, project="watched", source_path=sample_wav)

    pool = WorkerPool(settings, db_path)
    pool.start()
    try:
        deadline = time.monotonic() + 60
        seen = {}
        while time.monotonic() < deadline and not seen:
            seen = pool.active
            time.sleep(0.1)
        assert seen, "the pool never reported an active job"
        assert "watched" in seen.values()
    finally:
        pool.stop()


def test_an_empty_queue_is_not_an_error(settings, db_path):
    assert WorkerPool(settings, db_path).drain(timeout=10) == []
