"""Folder watcher tests (CLAUDE.md §18, §21).

§21 asks for five things: the prefix matched case-insensitively, non-matching
files ignored, waiting for stability, never double-processing, and a bad file
skipped without killing the loop.

The stability test is the one that matters most in practice. VoiceCleaner
writes the file over several seconds, and a watcher that grabs it early
transcribes half a narration — which looks like a successful run, not a
failure, so nobody notices until the video is wrong.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from visualresearcher.jobs.queue import list_jobs
from visualresearcher.jobs.watcher import (
    FolderWatcher,
    derive_project_name,
    matches_prefix,
)


@pytest.fixture
def drop(sandbox, settings) -> Path:
    folder = sandbox / "incoming"
    folder.mkdir(parents=True, exist_ok=True)
    settings.watch.dir = folder
    settings.watch.stable_secs = 0.2
    settings.watch.poll_interval_s = 0.1
    return folder


@pytest.fixture
def watcher(settings, db_path, drop) -> FolderWatcher:
    return FolderWatcher(settings, db_path, use_watchdog=False)


def _drop_wav(folder: Path, name: str, size: int = 2048) -> Path:
    path = folder / name
    path.write_bytes(b"RIFF" + b"\x00" * (size - 4))
    return path


# ---------------------------------------------------------------------------
# §21: the prefix is matched case-insensitively; non-matching is ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Tight_episode.wav", "tight_episode.wav", "TIGHT_EPISODE.WAV", "TiGhT_x.WaV"],
)
def test_the_prefix_matches_case_insensitively(name):
    """§18.2."""
    assert matches_prefix(name, "Tight") is True


@pytest.mark.parametrize(
    "name",
    [
        "episode_Tight.wav",  # prefix is not at the start
        "Tight_episode.mp3",  # not a wav
        "Tight_episode.wav.tmp",
        "random.wav",
        "Tight",  # no extension
        "",
    ],
)
def test_non_matching_names_are_ignored(name):
    assert matches_prefix(name, "Tight") is False


def test_an_empty_prefix_accepts_any_wav():
    assert matches_prefix("anything.wav", "") is True
    assert matches_prefix("anything.mp3", "") is False


def test_a_non_matching_file_is_left_alone(watcher, drop, db_path):
    ignored = _drop_wav(drop, "holiday_photos.wav")
    other = drop / "notes.txt"
    other.write_text("not audio", encoding="utf-8")

    assert watcher.scan_once() == []
    assert ignored.exists(), "a file we do not want must not be moved"
    assert other.exists()
    assert list_jobs(db_path) == []


# ---------------------------------------------------------------------------
# §21: it waits for stability
# ---------------------------------------------------------------------------


def test_a_growing_file_is_not_taken_until_it_settles(watcher, drop, db_path, settings):
    """§18.3: VoiceCleaner may still be writing."""
    settings.watch.stable_secs = 0.6
    path = drop / "Tight_growing.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 100)

    stop = threading.Event()

    def keep_writing():
        # Grow the file for a while, then stop.
        for _ in range(6):
            if stop.is_set():
                return
            with open(path, "ab") as handle:
                handle.write(b"\x00" * 4096)
            time.sleep(0.15)

    writer = threading.Thread(target=keep_writing, daemon=True)
    writer.start()

    # While it is still growing, the stability check must refuse.
    assert watcher.is_stable(path, timeout=0.5) is False, (
        "the watcher accepted a file that was still being written"
    )

    writer.join(timeout=5)
    stop.set()
    assert watcher.is_stable(path, timeout=5) is True, (
        "the watcher never accepted a file that had finished being written"
    )


def test_a_settled_file_is_accepted(watcher, drop, db_path):
    path = _drop_wav(drop, "Tight_settled.wav")
    assert watcher.is_stable(path, timeout=5) is True

    events = watcher.scan_once()
    assert len(events) == 1
    assert events[0].project == "tight_settled"


def test_a_zero_byte_file_is_refused(watcher, drop, db_path):
    (drop / "Tight_empty.wav").write_bytes(b"")
    assert watcher.scan_once() == []
    assert list_jobs(db_path) == []


def test_a_file_that_vanishes_mid_check_is_not_an_error(watcher, drop):
    path = _drop_wav(drop, "Tight_vanishing.wav")
    path.unlink()
    assert watcher.is_stable(path, timeout=1) is False


# ---------------------------------------------------------------------------
# §21: it never double-processes
# ---------------------------------------------------------------------------


def test_a_file_is_processed_exactly_once(watcher, drop, db_path):
    """§18.6, across restarts."""
    _drop_wav(drop, "Tight_once.wav")
    first = watcher.scan_once()
    assert len(first) == 1

    # The same name arrives again -- a re-export, say.
    _drop_wav(drop, "Tight_once.wav")
    second = watcher.scan_once()
    assert second == [], "the same filename was processed twice"
    assert len(list_jobs(db_path)) == 1


def test_the_ledger_survives_a_restart(settings, db_path, drop):
    """A fresh watcher object must not reprocess what the last one took."""
    _drop_wav(drop, "Tight_restart.wav")
    first = FolderWatcher(settings, db_path, use_watchdog=False)
    assert len(first.scan_once()) == 1

    _drop_wav(drop, "Tight_restart.wav")
    second = FolderWatcher(settings, db_path, use_watchdog=False)
    assert second.scan_once() == [], "a restarted watcher reprocessed the file"


def test_a_second_sweep_does_not_requeue(watcher, drop, db_path):
    _drop_wav(drop, "Tight_sweep.wav")
    watcher.scan_once()
    watcher.scan_once()
    watcher.scan_once()
    assert len(list_jobs(db_path)) == 1


# ---------------------------------------------------------------------------
# §21: a bad file is skipped without killing the loop
# ---------------------------------------------------------------------------


def test_one_unreadable_file_does_not_stop_the_others(watcher, drop, db_path, monkeypatch):
    """§18.8: log, leave the source alone, keep watching."""
    bad = _drop_wav(drop, "Tight_aaa_bad.wav")
    good = _drop_wav(drop, "Tight_zzz_good.wav")

    real_stat = Path.stat

    def exploding_stat(self, *args, **kwargs):
        if self.name == bad.name:
            raise PermissionError("locked by another process")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", exploding_stat)
    events = watcher.scan_once()
    monkeypatch.undo()

    names = [e.source.name for e in events]
    assert good.name in names, "a good file was skipped because a bad one came first"
    assert bad.exists(), "§18.8: the source file must be left untouched"


def test_a_failing_callback_does_not_kill_the_loop(settings, db_path, drop):
    def explode(event):
        raise RuntimeError("callback blew up")

    watcher = FolderWatcher(settings, db_path, on_accept=explode, use_watchdog=False)
    _drop_wav(drop, "Tight_callback.wav")
    events = watcher.scan_once()
    assert len(events) == 1, "the file should still have been accepted"


def test_a_failed_move_releases_the_claim(settings, db_path, drop, monkeypatch):
    """A file we could not take must be retryable, not silently burned."""
    watcher = FolderWatcher(settings, db_path, use_watchdog=False)
    source = _drop_wav(drop, "Tight_movefail.wav")

    def failing_move(src, dst):
        raise OSError("destination is read-only")

    monkeypatch.setattr(FolderWatcher, "_move", staticmethod(failing_move))
    assert watcher.scan_once() == []
    assert source.exists(), "§18.8: the source must be left untouched"
    monkeypatch.undo()

    # After the failure, a retry must be able to claim it again.
    retry = FolderWatcher(settings, db_path, use_watchdog=False)
    assert len(retry.scan_once()) == 1, "the claim was not released after a failed move"


def test_a_missing_watch_directory_is_not_fatal(settings, db_path, sandbox):
    settings.watch.dir = sandbox / "does-not-exist"
    watcher = FolderWatcher(settings, db_path, use_watchdog=False)
    assert watcher.scan_once() == []


# ---------------------------------------------------------------------------
# §18.4-18.5: naming, moving, and the shared queue
# ---------------------------------------------------------------------------


def test_the_project_name_is_the_slugified_stem(watcher, drop, db_path):
    _drop_wav(drop, "Tight_Baras Episode 12.wav")
    events = watcher.scan_once()
    assert events[0].project == "tight_baras_episode_12"


def test_a_name_collision_gets_a_timestamp_suffix(settings, sandbox):
    projects = sandbox / "projects"
    (projects / "tight_dupe").mkdir(parents=True)
    name = derive_project_name(Path("Tight_dupe.wav"), projects)
    assert name.startswith("tight_dupe_")
    assert name != "tight_dupe"


def test_the_file_is_moved_not_copied(watcher, drop, db_path, settings):
    """§18.5: MOVE, so the drop folder does not fill up."""
    source = _drop_wav(drop, "Tight_moved.wav")
    events = watcher.scan_once()

    assert not source.exists(), "the source was left in the drop folder"
    destination = settings.project_dir(events[0].project) / "input" / "narration.wav"
    assert destination.exists()
    assert destination.stat().st_size == events[0].size


def test_the_job_lands_on_the_same_queue_as_the_cli(watcher, drop, db_path):
    """§18.5: identical downstream path, no special-casing."""
    _drop_wav(drop, "Tight_queue.wav")
    watcher.scan_once()

    jobs = list_jobs(db_path)
    assert len(jobs) == 1
    job = jobs[0]
    assert str(job.state) == "QUEUED", "it must be queued, not run inline"
    assert job.origin == "watcher"
    assert job.origin_filename == "Tight_queue.wav"
    assert job.source_path.endswith("narration.wav")


def test_the_watcher_only_enqueues(watcher, drop, db_path, settings):
    """It must not run the pipeline itself; the worker does that."""
    _drop_wav(drop, "Tight_enqueue.wav")
    events = watcher.scan_once()
    project = settings.project_dir(events[0].project)
    assert not (project / "segments").exists(), "the watcher ran the pipeline"
    assert not (project / "selected").exists()


def test_several_drops_are_all_taken(watcher, drop, db_path):
    for index in range(4):
        _drop_wav(drop, f"Tight_batch_{index}.wav")
    events = watcher.scan_once()
    assert len(events) == 4
    assert len({e.project for e in events}) == 4
    assert len(list_jobs(db_path)) == 4


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_the_background_loop_picks_up_a_later_drop(settings, db_path, drop):
    watcher = FolderWatcher(settings, db_path, use_watchdog=False)
    watcher.start_background()
    try:
        _drop_wav(drop, "Tight_later.wav")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not watcher.accepted:
            time.sleep(0.1)
        assert watcher.accepted, "the running watcher never picked up a new file"
    finally:
        watcher.stop()


def test_the_loop_processes_a_backlog_on_startup(settings, db_path, drop):
    _drop_wav(drop, "Tight_backlog.wav")
    watcher = FolderWatcher(settings, db_path, use_watchdog=False)
    watcher.run(once=True)
    assert len(watcher.accepted) == 1, "files dropped while down must still be taken"


def test_stop_is_idempotent(settings, db_path, drop):
    watcher = FolderWatcher(settings, db_path, use_watchdog=False)
    watcher.stop()
    watcher.stop()
