"""Segment pipelining (§21).

The batch order runs every segment through a stage before any segment sees the
next, so ``selected/`` stays empty until the final stage and a long run shows
nothing until it is nearly over. Pipelining walks the same stages in the same
order one segment at a time, delivering each as it finishes.

The whole claim is that this **reorders and does not change** the work. These
tests are mostly here to hold that claim down: same files, same credits, the
model loaded once, and §21's row/file equality true at every step rather than
only at the end.
"""

from __future__ import annotations

import csv

import pytest

from visualresearcher.jobs.queue import enqueue, get_job
from visualresearcher.jobs.worker import RunContext, run_pipeline, run_segment_pipelined


def _fresh(tmp_path_factory, sample_wav, name):
    """An isolated settings root, database and context."""
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db

    root = tmp_path_factory.mktemp(name)
    for sub in ("projects", "data", ".cache", ".tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path=root / "absent.yaml", root=root)
    settings.ensure_dirs()
    init_db(settings.db_path)
    job = enqueue(settings.db_path, project=name, source_path=sample_wav)
    return RunContext(
        job_id=job.id,
        project=name,
        project_root=settings.project_dir(name),
        source_path=sample_wav,
        settings=settings,
        db_path=settings.db_path,
    )


def _selected(ctx) -> list[str]:
    d = ctx.paths.selected_dir
    return sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []


def _rows(ctx) -> list[dict]:
    path = ctx.paths.sources_csv
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory, sample_wav):
    """The same narration, once batched and once pipelined."""
    batched = _fresh(tmp_path_factory, sample_wav, "batched")
    run_pipeline(batched)
    piped = _fresh(tmp_path_factory, sample_wav, "piped")
    run_segment_pipelined(piped)
    return batched, piped


# ---------------------------------------------------------------------------
# The reordering must not change the deliverable
# ---------------------------------------------------------------------------


def test_pipelining_delivers_the_same_files_as_batching(both_runs):
    batched, piped = both_runs
    assert _selected(piped) == _selected(batched)
    assert _selected(piped), "neither run produced anything, so this proves nothing"


def test_pipelining_delivers_the_same_credits_as_batching(both_runs):
    """Same rows, same order, same provenance -- not merely the same count."""
    batched, piped = both_runs
    left, right = _rows(batched), _rows(piped)
    assert [r["file"] for r in left] == [r["file"] for r in right]
    for a, b in zip(left, right, strict=True):
        assert a == b, f"provenance differs for {a['file']}"


# ---------------------------------------------------------------------------
# ...while actually delivering incrementally
# ---------------------------------------------------------------------------


def test_selected_fills_up_as_segments_finish(tmp_path_factory, sample_wav):
    """The point of the exercise: output before the last stage, not after it."""
    ctx = _fresh(tmp_path_factory, sample_wav, "incremental")
    counts: list[int] = []

    run_segment_pipelined(
        ctx, on_segment=lambda index, position, total: counts.append(len(_selected(ctx)))
    )

    assert counts, "no segment was ever reported delivered"
    assert counts == sorted(counts), f"selected/ shrank during the run: {counts}"
    assert counts[0] > 0, "the first segment delivered nothing; this is still batching"
    assert counts[-1] == len(_selected(ctx))
    assert counts[0] < counts[-1], "every file appeared at once, so nothing was pipelined"


def test_segments_are_delivered_in_narration_order(tmp_path_factory, sample_wav):
    ctx = _fresh(tmp_path_factory, sample_wav, "ordered")
    order: list[int] = []
    run_segment_pipelined(ctx, on_segment=lambda index, pos, total: order.append(index))
    assert order == sorted(order), f"segments were delivered out of order: {order}"


def test_sources_csv_matches_selected_at_every_step(tmp_path_factory, sample_wav):
    """§21 is an invariant, not an end state.

    A reader opening the folder mid-run must not find an image with no credit,
    or a credit pointing at a file that is not there.
    """
    ctx = _fresh(tmp_path_factory, sample_wav, "invariant")
    breaches: list[str] = []

    def on_segment(index, position, total):
        files, rows = _selected(ctx), _rows(ctx)
        if len(files) != len(rows):
            breaches.append(f"after segment {index}: {len(files)} file(s), {len(rows)} row(s)")
        named = {r["file"] for r in rows}
        if named != set(files):
            breaches.append(f"after segment {index}: {named ^ set(files)} unaccounted for")

    run_segment_pipelined(ctx, on_segment=on_segment)
    assert not breaches, "§21 broke mid-run:\n  " + "\n  ".join(breaches)


def test_progress_is_visible_in_the_database(tmp_path_factory, sample_wav):
    """``status`` reads these, so "N of 95" depends on them being written."""
    ctx = _fresh(tmp_path_factory, sample_wav, "progress")
    snapshots: list[tuple[int, int]] = []

    def on_segment(index, position, total):
        job = get_job(ctx.db_path, ctx.job_id)
        snapshots.append((job.segments_done, job.segments_total))

    run_segment_pipelined(ctx, on_segment=on_segment)
    assert snapshots, "no progress was recorded at all"
    done = [d for d, _ in snapshots]
    totals = {t for _, t in snapshots}
    assert done == sorted(done) and done[-1] == len(done)
    assert len(totals) == 1 and totals.pop() == len(done)


# ---------------------------------------------------------------------------
# The expensive things must not be rebuilt once per segment
# ---------------------------------------------------------------------------


def test_the_embedding_model_is_built_once(tmp_path_factory, sample_wav, monkeypatch):
    """Per-segment ranking must not reload CLIP for every segment.

    This is the failure that would turn "reordering" into "ten times slower":
    the batch stage built the embedder once for the whole run, and a naive
    per-segment loop rebuilds it 95 times.
    """
    from visualresearcher.providers import registry

    real = registry.resolve
    built: list[str] = []

    def counting_resolve(kind, name, **kwargs):
        if kind == "embedding":
            built.append(name)
        return real(kind, name, **kwargs)

    monkeypatch.setattr(registry, "resolve", counting_resolve)
    ctx = _fresh(tmp_path_factory, sample_wav, "oneclip")
    delivered: list[int] = []
    run_segment_pipelined(ctx, on_segment=lambda i, p, t: delivered.append(i))

    assert len(delivered) > 1, "need several segments for this to mean anything"
    assert len(built) <= 1, (
        f"the embedder was built {len(built)} times for {len(delivered)} segments; "
        "CLIP would be reloaded on every segment of a real run"
    )
