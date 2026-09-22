"""End-to-end pipeline tests (CLAUDE.md §21, §22 P1).

The P1 acceptance criterion is: ``run tests/fixtures/sample.wav`` writes
transcript, SRT, and correctly named folders. These tests assert exactly that,
plus the resume behaviour that §8 promises.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from visualresearcher.jobs.queue import (
    completed_stages,
    enqueue,
    get_job,
    resume_point,
)
from visualresearcher.jobs.states import STAGE_ORDER, JobState, Stage
from visualresearcher.jobs.worker import RunContext, run_pipeline
from visualresearcher.schemas import Segment

SEGMENT_DIR = re.compile(r"^\d{3,}_\d{2}m\d{2}s-\d{2}m\d{2}s$")


def _context(sample_wav: Path, settings, db_path, name="demo", **kwargs) -> RunContext:
    job = enqueue(db_path, project=name, source_path=sample_wav)
    return RunContext(
        job_id=job.id,
        project=name,
        project_root=settings.project_dir(name),
        source_path=sample_wav,
        settings=settings,
        db_path=db_path,
        **kwargs,
    )


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, sample_wav):
    """One full pipeline run, shared by every test that only reads it.

    Module-scoped deliberately. A full run costs about half a minute, and
    rebuilding it per test made this file alone take ten minutes -- which is
    at odds with §2.5's "run pytest constantly". Every consumer of this
    fixture only reads; the tests that mutate, resume or force a failure build
    their own context with `_context(...)` and are unaffected.
    """
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db

    root = tmp_path_factory.mktemp("full_run")
    for name in ("projects", "data", ".cache", ".tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path=root / "absent.yaml", root=root)
    settings.ensure_dirs()
    init_db(settings.db_path)

    ctx = _context(sample_wav, settings, settings.db_path)
    results = run_pipeline(ctx)
    return ctx, results


# ---------------------------------------------------------------------------
# P1 acceptance
# ---------------------------------------------------------------------------


def test_run_writes_every_phase_one_artifact(completed_run):
    ctx, _ = completed_run
    for label, path in (
        ("narration.wav", ctx.paths.narration_wav),
        ("transcript.json", ctx.paths.transcript_json),
        ("transcript.txt", ctx.paths.transcript_txt),
        ("narration.srt", ctx.paths.narration_srt),
        ("timeline.csv", ctx.paths.timeline_csv),
    ):
        assert path.exists(), f"{label} was not written"
        assert path.stat().st_size > 0, f"{label} is empty"


def test_run_creates_correctly_named_segment_folders(completed_run):
    ctx, _ = completed_run
    dirs = ctx.paths.existing_segment_dirs()
    assert len(dirs) >= 5, f"60s of narration should give several segments, got {len(dirs)}"
    for path in dirs:
        assert SEGMENT_DIR.match(path.name), f"{path.name} is not in NNN_MMmSSs-MMmSSs form"
        assert (path / "segment.json").exists(), f"{path.name} has no segment.json"


def test_segment_folders_are_in_chronological_order_on_disk(completed_run):
    """Sorted by name must equal sorted by start time -- that is the product (§6)."""
    ctx, _ = completed_run
    dirs = ctx.paths.existing_segment_dirs()
    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8")) for d in dirs
    ]
    assert [s.index for s in segments] == sorted(s.index for s in segments)
    assert [s.start for s in segments] == sorted(s.start for s in segments)
    for prev, nxt in zip(segments[:-1], segments[1:], strict=True):
        assert prev.end == nxt.start, "segments on disk are not contiguous"


def test_segment_json_matches_its_folder_name(completed_run):
    ctx, _ = completed_run
    for path in ctx.paths.existing_segment_dirs():
        segment = Segment.model_validate_json((path / "segment.json").read_text(encoding="utf-8"))
        assert path.name.startswith(f"{segment.index:03d}_")


def test_every_segment_has_narration_text(completed_run):
    ctx, _ = completed_run
    for path in ctx.paths.existing_segment_dirs():
        segment = Segment.model_validate_json((path / "segment.json").read_text(encoding="utf-8"))
        assert segment.narration.strip(), f"segment {segment.index} has no narration"


def test_timeline_csv_has_one_row_per_segment(completed_run):
    import csv

    ctx, _ = completed_run
    with open(ctx.paths.timeline_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(ctx.paths.existing_segment_dirs())
    assert rows[0]["folder"] == ctx.paths.existing_segment_dirs()[0].name


def test_source_file_is_copied_not_moved(sample_wav, settings, db_path):
    """§2.8: never destroy user data."""
    ctx = _context(sample_wav, settings, db_path, name="copytest")
    run_pipeline(ctx)
    assert sample_wav.exists(), "the CLI must not move the user's narration file"


def test_unimplemented_stages_are_skipped_not_faked(completed_run):
    """A stage that is not built says so; it never writes a placeholder (§6).

    The expected sets are read from the runner rather than hardcoded, so this
    keeps testing the rule as phases land instead of going stale every time.
    """
    from visualresearcher.jobs.worker import _PLANNED, IMPLEMENTED_STAGES

    ctx, results = completed_run
    by_stage = {r.stage: r for r in results}

    for stage in IMPLEMENTED_STAGES:
        assert by_stage[stage].ok, f"{stage} failed"
        assert not by_stage[stage].skipped, f"{stage} is implemented but was skipped"

    for stage in _PLANNED:
        assert by_stage[stage].skipped, f"{stage} claimed to have run"
        assert "not implemented" in by_stage[stage].detail

    assert set(IMPLEMENTED_STAGES) | set(_PLANNED) == set(STAGE_ORDER), (
        "every stage must be either implemented or explicitly planned"
    )
    # §6: never fabricate a placeholder. Every file in selected/ must be a
    # real copy of a real pick, not something invented to fill a slot.
    picks = {Path(p.path).name for segment in ctx.segments for p in segment.picks if p.use}
    originals = {s.stat().st_size for s in ctx.paths.segments_dir.rglob("*") if s.is_file()}
    for path in ctx.paths.selected_dir.iterdir():
        assert path.stat().st_size > 0, f"{path.name} is empty"
        assert path.stat().st_size in originals, (
            f"{path.name} in selected/ has no matching original in segments/; it looks fabricated"
        )
    assert picks, "the pipeline should have chosen something to deliver"


def test_the_terminal_state_reflects_whether_there_are_gaps(completed_run):
    """§14: a job with a gap ends REVIEW_REQUIRED, not COMPLETE."""
    ctx, _ = completed_run
    job = get_job(ctx.db_path, ctx.job_id)
    assert job.segments_total == len(ctx.paths.existing_segment_dirs())

    if job.gaps:
        assert job.state == JobState.REVIEW_REQUIRED, (
            f"{job.gaps} gap(s) but the job claimed {job.state}"
        )
    else:
        assert job.state == JobState.COMPLETE
        assert list(ctx.paths.selected_dir.iterdir()), (
            "a COMPLETE job with no gaps must have delivered something"
        )


def test_a_gap_forces_review_required(sample_wav, settings, db_path):
    """Raise the confidence floor so nothing qualifies, and check the state."""
    settings.confidence.medium = 0.999
    settings.confidence.high = 0.9999
    ctx = _context(sample_wav, settings, db_path, name="allgaps")
    run_pipeline(ctx)

    job = get_job(db_path, ctx.job_id)
    assert job.gaps > 0
    assert job.state == JobState.REVIEW_REQUIRED
    assert list(ctx.paths.selected_dir.iterdir()) == [], (
        "§6: nothing may be copied when no pick is confident enough"
    )


def test_every_implemented_stage_is_checkpointed(completed_run):
    ctx, _ = completed_run
    done = completed_stages(ctx.db_path, ctx.job_id)
    assert {"ingest", "transcribe", "segment"} <= done


# ---------------------------------------------------------------------------
# Resume (§8, §21)
# ---------------------------------------------------------------------------


def _fingerprint(root: Path) -> dict[str, str]:
    """Content hash of every file in the project, keyed by relative path.

    Two things are normalised out before hashing, and only two:

    * **the project's own name**, because the reports legitimately print it
      (``# Shot list — <project>``) and two differently-named projects are
      supposed to differ there;
    * **the Timing section of research_report.md**, because it records how
      long each stage took. Those numbers are wall-clock and will never match
      between two runs. Excluding the section rather than the whole file keeps
      the rest of the report byte-compared.

    Everything else must match byte for byte. That is the point of the test.
    """
    out: dict[str, str] = {}
    name = root.name.encode("utf-8")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "job.log":
            continue
        data = path.read_bytes().replace(name, b"<PROJECT>")
        if path.name == "research_report.md":
            data = _strip_timing(data)
        out[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
    return out


def _strip_timing(data: bytes) -> bytes:
    """Drop the ``## Timing`` section, which is wall-clock and never repeats."""
    text = data.decode("utf-8", errors="replace")
    start = text.find("## Timing")
    if start == -1:
        return data
    end = text.find("## How to read", start)
    if end == -1:
        end = len(text)
    return (text[:start] + text[end:]).encode("utf-8")


def test_resume_after_a_kill_reproduces_identical_output(sample_wav, settings, db_path):
    """§21: kill mid-stage -> resume gives identical output, no duplicated work."""
    reference = _context(sample_wav, settings, db_path, name="reference")
    run_pipeline(reference)
    expected = _fingerprint(reference.project_root)

    # Simulate a process killed after transcription: ingest and transcribe are
    # checkpointed, segmentation never ran.
    killed = _context(sample_wav, settings, db_path, name="killed")
    run_pipeline(killed, stages=[Stage.INGEST, Stage.TRANSCRIBE])
    assert not killed.paths.segments_dir.exists() or not list(killed.paths.segments_dir.iterdir())

    start = resume_point(db_path, killed.job_id)
    assert start == Stage.CONTEXT, f"resume should pick up after transcribe, got {start}"

    resumed = RunContext(
        job_id=killed.job_id,
        project=killed.project,
        project_root=killed.project_root,
        source_path=sample_wav,
        settings=settings,
        db_path=db_path,
    )
    results = run_pipeline(resumed, stages=list(STAGE_ORDER[STAGE_ORDER.index(start) :]))
    assert all(r.ok for r in results)

    got = _fingerprint(resumed.project_root)
    assert got == expected, (
        "resumed output differs from an uninterrupted run:\n"
        f"only in resumed: {sorted(set(got) - set(expected))}\n"
        f"only in reference: {sorted(set(expected) - set(got))}\n"
        f"differing: {sorted(k for k in set(got) & set(expected) if got[k] != expected[k])}"
    )


def test_resume_does_not_redo_checkpointed_work(sample_wav, settings, db_path):
    ctx = _context(sample_wav, settings, db_path, name="noredo")
    run_pipeline(ctx)
    transcript_mtime = ctx.paths.transcript_json.stat().st_mtime_ns

    again = RunContext(
        job_id=ctx.job_id,
        project=ctx.project,
        project_root=ctx.project_root,
        source_path=sample_wav,
        settings=settings,
        db_path=db_path,
    )
    results = run_pipeline(again)
    ran = [r for r in results if not r.skipped]
    assert not ran, f"stages re-ran despite checkpoints: {[str(r.stage) for r in ran]}"
    assert ctx.paths.transcript_json.stat().st_mtime_ns == transcript_mtime


def test_force_redoes_checkpointed_stages(sample_wav, settings, db_path):
    ctx = _context(sample_wav, settings, db_path, name="forced")
    run_pipeline(ctx)

    again = RunContext(
        job_id=ctx.job_id,
        project=ctx.project,
        project_root=ctx.project_root,
        source_path=sample_wav,
        settings=settings,
        db_path=db_path,
        force=True,
    )
    results = run_pipeline(again)
    ran = {str(r.stage) for r in results if not r.skipped}
    assert {"ingest", "transcribe", "segment"} <= ran


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_a_stage_failure_marks_the_job_failed_and_keeps_the_error(
    sample_wav, settings, db_path, monkeypatch
):
    from visualresearcher.jobs import worker as worker_mod

    def explode(ctx):
        raise RuntimeError("segmentation exploded")

    monkeypatch.setitem(worker_mod.IMPLEMENTED_STAGES, Stage.SEGMENT, explode)

    ctx = _context(sample_wav, settings, db_path, name="boom")
    results = run_pipeline(ctx)

    assert any(not r.ok for r in results)
    job = get_job(db_path, ctx.job_id)
    assert job.state == JobState.FAILED
    assert "segmentation exploded" in job.error
    assert str(Stage.SEGMENT) not in completed_stages(db_path, ctx.job_id)


def test_disk_guard_stops_the_run_before_writing(sample_wav, settings, db_path):
    """§3: below the floor, abort the stage with a clear message."""
    from visualresearcher.utils.disk import free_gb

    settings.output.min_free_gb = free_gb(settings.projects_dir) + 10_000
    ctx = _context(sample_wav, settings, db_path, name="nodisk")
    results = run_pipeline(ctx)

    assert results[0].stage == Stage.INGEST
    assert not results[0].ok
    job = get_job(db_path, ctx.job_id)
    assert job.state == JobState.FAILED
    assert "GB free" in job.error
    assert not ctx.paths.transcript_json.exists(), "nothing should have been written"


def test_missing_source_file_is_reported_clearly(settings, db_path, tmp_path):
    ctx = _context(tmp_path / "nope.wav", settings, db_path, name="missing")
    results = run_pipeline(ctx)
    assert not results[0].ok
    assert "not found" in results[0].detail.lower()


def test_unsupported_audio_format_is_rejected(settings, db_path, tmp_path):
    bad = tmp_path / "narration.txt"
    bad.write_text("not audio", encoding="utf-8")
    ctx = _context(bad, settings, db_path, name="badformat")
    results = run_pipeline(ctx)
    assert not results[0].ok
    assert "unsupported" in results[0].detail.lower()
