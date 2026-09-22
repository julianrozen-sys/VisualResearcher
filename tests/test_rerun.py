"""Re-running one segment (CLAUDE.md §17, §20).

§20's rule is the one under test: **re-running one segment must not re-run the
project.** A reviewer who edits a query for segment 31 wants segment 31
searched again, not four hundred segments re-downloaded — and the test for
that is that every *other* segment's files are byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from visualresearcher.jobs.rerun import RERUNNABLE_STAGES, rerun_segment
from visualresearcher.jobs.worker import RunContext, run_pipeline
from visualresearcher.project import ProjectPaths
from visualresearcher.schemas import Segment


@pytest.fixture(scope="module")
def _built_once(tmp_path_factory, sample_wav) -> Path:
    """Run the whole pipeline once for this module, into a shared directory.

    Re-running a segment needs a real project to re-run, and building one
    costs about half a minute. Doing that per test made this file alone take
    five minutes, which is at odds with §2.5's "run pytest constantly". So it
    is built once and each test gets a copy.
    """
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db
    from visualresearcher.jobs.queue import enqueue

    root = tmp_path_factory.mktemp("rerun_source")
    for name in ("projects", "data", ".cache", ".tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path=root / "absent.yaml", root=root)
    settings.ensure_dirs()
    init_db(settings.db_path)

    job = enqueue(settings.db_path, project="demo", source_path=sample_wav)
    ctx = RunContext(
        job_id=job.id,
        project="demo",
        project_root=settings.project_dir("demo"),
        source_path=sample_wav,
        settings=settings,
        db_path=settings.db_path,
    )
    run_pipeline(ctx)
    return ctx.paths.root


@pytest.fixture
def project(_built_once, settings) -> ProjectPaths:
    """A fresh copy of that project, so each test can mutate it freely."""
    import shutil

    destination = settings.project_dir("demo")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_built_once, destination)
    return ProjectPaths(destination)


def _fingerprint(root: Path, *, skip: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "job.log":
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if skip and relative.startswith(skip):
            continue
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _segment(paths: ProjectPaths, index: int) -> Segment:
    for directory in paths.existing_segment_dirs():
        segment = Segment.model_validate_json(
            (directory / "segment.json").read_text(encoding="utf-8")
        )
        if segment.index == index:
            return segment
    raise AssertionError(f"no segment {index}")


# ---------------------------------------------------------------------------
# §20: re-running one segment must not re-run the project
# ---------------------------------------------------------------------------


def test_rerunning_one_segment_leaves_the_others_untouched(project, settings):
    """The §20 requirement, stated literally."""
    target = _segment(project, 2)
    folder = project.segment_dir(target).relative_to(project.root).as_posix()

    before = _fingerprint(project.root, skip=folder)
    rerun_segment("demo", 2, settings)
    after = _fingerprint(project.root, skip=folder)

    # selected/ and the reports may legitimately change; nothing else may.
    volatile = {
        "selected/",
        ".selected_manifest.json",
        "sources.csv",
        "shotlist.md",
        "research_report.md",
    }

    def stable(items):
        return {
            k: v for k, v in items.items() if not any(k.startswith(prefix) for prefix in volatile)
        }

    assert stable(after) == stable(before), (
        "re-running one segment changed files belonging to other segments:\n"
        f"{sorted(set(stable(after)) ^ set(stable(before)))}"
    )


def test_rerunning_changes_the_targeted_segment(project, settings):
    folder = project.segment_dir(_segment(project, 2))
    before = (folder / "segment.json").read_text(encoding="utf-8")

    result = rerun_segment("demo", 2, settings)

    after = (folder / "segment.json").read_text(encoding="utf-8")
    assert after != before, "the targeted segment was not touched"
    assert "re-ran" in after
    assert result.segment_index == 2
    assert result.stages


def test_the_default_stage_redoes_the_image_research(project, settings):
    result = rerun_segment("demo", 1, settings)
    assert "image_search" in result.stages
    assert "download" in result.stages
    assert "dedupe" in result.stages
    assert "rank" in result.stages
    assert result.candidates > 0
    assert result.kept > 0
    assert result.selected > 0


def test_rank_only_does_not_redownload(project, settings):
    result = rerun_segment("demo", 1, settings, stage="rank")
    assert result.stages == ["rank"]
    assert result.candidates == 0, "rank-only must not search"
    assert result.downloaded == 0, "rank-only must not download"
    assert result.selected > 0


def test_rerunning_keeps_selected_consistent(project, settings):
    """§15: the deliverable must still match the picks afterwards."""
    from visualresearcher.pipeline.collect import plan_selected

    rerun_segment("demo", 3, settings)

    segments = [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in project.existing_segment_dirs()
    ]
    planned, _ = plan_selected(segments, project, settings)
    on_disk = {p.name for p in project.selected_dir.iterdir() if p.is_file()}
    assert {f.name for f in planned} == on_disk, "selected/ drifted from the picks after a re-run"


def test_an_image_rerun_does_not_drop_the_clip(project, settings):
    """A clip already downloaded must survive re-searching the images."""
    before = _segment(project, 2)
    had_clip = any(p.kind == "clip" for p in before.picks)

    rerun_segment("demo", 2, settings)

    after = _segment(project, 2)
    if had_clip:
        assert any(p.kind == "clip" for p in after.picks), (
            "re-running the image search threw away the clip"
        )


def test_a_video_rerun_is_available(project, settings):
    result = rerun_segment("demo", 2, settings, stage="video_search")
    assert "video_search" in result.stages
    assert result.videos > 0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_an_unknown_project_raises(settings):
    with pytest.raises(LookupError, match="no project"):
        rerun_segment("nope", 1, settings)


def test_an_unknown_segment_raises(project, settings):
    with pytest.raises(LookupError, match="no segment"):
        rerun_segment("demo", 999, settings)


def test_an_unknown_stage_is_rejected(project, settings):
    with pytest.raises(ValueError, match="--stage"):
        rerun_segment("demo", 1, settings, stage="not_a_stage")


def test_the_advertised_stages_all_work(project, settings):
    """Every value the CLI accepts must actually run."""
    assert set(RERUNNABLE_STAGES) == {"image_search", "video_search", "clips", "rank"}
    for stage in RERUNNABLE_STAGES:
        result = rerun_segment("demo", 1, settings, stage=stage)
        assert result.stages, f"{stage} produced no stages"


# ---------------------------------------------------------------------------
# Resuming a pipelined run (§21)
#
# A long run is exactly the kind that gets interrupted. When one died at
# segment 21 of 170, resuming re-ran every stage from `ingest` and re-searched
# the 20 finished segments: the batch runner's "already checkpointed" skip
# lived in its own loop, and per-segment progress was recorded nowhere, because
# a pipelined run touches each stage once per segment.
# ---------------------------------------------------------------------------


def _all(ctx) -> list[Segment]:
    """Every segment, re-read from disk so picks reflect what was written."""
    return [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in ctx.paths.existing_segment_dirs()
        if (d / "segment.json").exists()
    ]


def _finished_run(tmp_path_factory, sample_wav, name):
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db
    from visualresearcher.jobs.queue import enqueue
    from visualresearcher.jobs.worker import RunContext, run_segment_pipelined

    root = tmp_path_factory.mktemp(name)
    for sub in ("projects", "data", ".cache", ".tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path=root / "absent.yaml", root=root)
    settings.ensure_dirs()
    init_db(settings.db_path)
    job = enqueue(settings.db_path, project=name, source_path=sample_wav)
    ctx = RunContext(
        job_id=job.id,
        project=name,
        project_root=settings.project_dir(name),
        source_path=sample_wav,
        settings=settings,
        db_path=settings.db_path,
    )
    run_segment_pipelined(ctx)
    return ctx, settings


def test_a_finished_segment_is_recognised_as_complete(tmp_path_factory, sample_wav):
    from visualresearcher.jobs.worker import segment_is_complete

    ctx, _settings = _finished_run(tmp_path_factory, sample_wav, "complete")
    segments = _all(ctx)
    assert segments, "no segments, so this proves nothing"
    assert all(segment_is_complete(ctx, s) for s in segments)


def test_a_segment_killed_mid_download_is_redone(tmp_path_factory, sample_wav):
    """No manifest means the segment is redone whole, not half-trusted."""
    from visualresearcher.jobs.worker import segment_is_complete

    ctx, _settings = _finished_run(tmp_path_factory, sample_wav, "killed")
    victim = _all(ctx)[-1]
    ctx.paths.segment_image_manifest(victim, width=ctx.width).unlink()
    assert not segment_is_complete(ctx, victim)


def test_resuming_does_not_redo_finished_segments(tmp_path_factory, sample_wav):
    """The whole point: picking up at segment N leaves 1..N-1 alone."""
    from visualresearcher.jobs.worker import run_segment_pipelined

    ctx, _settings = _finished_run(tmp_path_factory, sample_wav, "resume")
    segments = _all(ctx)
    victim = segments[-1]
    ctx.paths.segment_image_manifest(victim, width=ctx.width).unlink()

    # Fingerprint the untouched segments so any re-search would show up.
    survivors = {
        s.index: ctx.paths.segment_image_manifest(s, width=ctx.width).stat().st_mtime_ns
        for s in segments
        if s.index != victim.index
    }

    ctx.segments = []
    ctx.delivered = set()
    touched: list[int] = []
    run_segment_pipelined(ctx, on_segment=lambda i, p, t: touched.append(i))

    assert touched == [victim.index], (
        f"resume reprocessed {touched} when only segment {victim.index} was incomplete"
    )
    for s in segments:
        if s.index == victim.index:
            continue
        now = ctx.paths.segment_image_manifest(s, width=ctx.width).stat().st_mtime_ns
        assert now == survivors[s.index], f"segment {s.index} was re-searched on resume"


def test_resuming_does_not_re_transcribe(tmp_path_factory, sample_wav):
    """`transcribe` is checkpointed; a resumed run must honour that."""
    from visualresearcher.jobs.states import Stage
    from visualresearcher.jobs.worker import run_segment_pipelined

    ctx, _settings = _finished_run(tmp_path_factory, sample_wav, "notrans")
    ctx.segments = []
    ctx.delivered = set()
    # Checked on the returned results, not the `on_stage` callback: a skipped
    # stage returns before the callback fires, the same way the batch runner
    # handles one, so the callback is silent precisely when the skip worked.
    results = run_segment_pipelined(ctx)
    by_stage = {str(r.stage): r for r in results}
    assert str(Stage.TRANSCRIBE) in by_stage, "transcribe was not reported at all"
    assert by_stage[str(Stage.TRANSCRIBE)].skipped, (
        "transcribe was re-run on resume despite being checkpointed"
    )
    assert by_stage[str(Stage.TRANSCRIBE)].detail == "checkpointed"


def test_force_overrides_resume(tmp_path_factory, sample_wav):
    """`--force` still means redo everything."""
    from visualresearcher.jobs.worker import run_segment_pipelined

    ctx, _settings = _finished_run(tmp_path_factory, sample_wav, "forced")
    expected = len(_all(ctx))
    ctx.segments = []
    ctx.delivered = set()
    ctx.force = True
    touched: list[int] = []
    run_segment_pipelined(ctx, on_segment=lambda i, p, t: touched.append(i))
    assert len(touched) == expected, "--force skipped segments it was told to redo"
