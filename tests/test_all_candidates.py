"""``all_candidates/`` — the flat per-project dump and its ordering guarantee.

The folder is only useful if sorting it alphabetically — what Explorer, ``ls``,
a zip listing and Python's ``sorted`` all do — reproduces the narration exactly.
These tests check that claim against the cases that actually break it, and each
ordering test carries a negative control so it cannot pass vacuously.
"""

from __future__ import annotations

from visualresearcher.pipeline.all_candidates import (
    DIR_NAME,
    MANIFEST_NAME,
    SELECTED_MARKER,
    dump_all_candidates,
    plan_all_candidates,
)
from visualresearcher.utils.files import candidate_filename, pad_width, rank_pad_width


def _names(segments, ranks, *, width=None, rank_width=None, seconds=lambda s: s * 8.0):
    """Filenames built in true narration order: segment asc, then rank asc."""
    width = width if width is not None else pad_width(max(segments))
    rank_width = rank_width if rank_width is not None else rank_pad_width(max(ranks))
    return [
        candidate_filename(s, seconds(s), r, f"shot {r}", ".jpg", width, rank_width)
        for s in segments
        for r in ranks
    ]


def _ordered(names: list[str]) -> bool:
    return sorted(names) == names


def _listing(paths) -> list[str]:
    folder = paths.root / DIR_NAME
    return sorted(p.name for p in folder.iterdir() if p.name != MANIFEST_NAME)


def _project(tmp_path_factory, sample_wav, name):
    """A complete finished run in its own settings root."""
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db
    from visualresearcher.jobs.queue import enqueue
    from visualresearcher.jobs.worker import RunContext, run_pipeline

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
    run_pipeline(ctx)
    return ctx, settings


def _segments(ctx):
    from visualresearcher.schemas import Segment

    return [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in ctx.paths.existing_segment_dirs()
        if (d / "segment.json").exists()
    ]


# ---------------------------------------------------------------------------
# Ordering, at the scale that matters
# ---------------------------------------------------------------------------


def test_ninety_five_segments_sort_in_narration_order():
    """The real case: 95 segments x 8 ranked candidates."""
    assert _ordered(_names(range(1, 96), range(1, 9)))


def test_segment_9_sorts_before_segment_90():
    """The exact bug that would silently break the point of the folder."""
    assert _ordered(_names(range(1, 96), [1]))
    assert not _ordered(_names(range(1, 96), [1], width=1)), (
        "unpadded segment numbers sorted correctly, so this test proves nothing"
    )


def test_rank_10_sorts_after_rank_2():
    """`keep_per_segment` is configuration; above 9 an unpadded rank shuffles."""
    assert _ordered(_names([1, 2], range(1, 13)))
    assert not _ordered(_names([1, 2], range(1, 13), rank_width=1)), (
        "an unpadded rank sorted correctly, so this test proves nothing"
    )


def test_a_project_past_999_segments_still_sorts():
    names = _names(range(1, 1201), [1])
    assert _ordered(names)
    assert names[0].startswith("0001_")


def test_a_narration_past_99_minutes_still_sorts():
    """`100m20s` sorts before `10m30s`, so the segment number must carry order."""
    names = _names(range(1, 121), [1], seconds=lambda s: s * 70.0)
    assert _ordered(names)
    assert "100m" in "".join(names), "the >99 minute case was never exercised"


def test_the_selected_marker_does_not_disturb_the_order():
    """A suffix keeps order; a leading marker would clump every pick together."""
    plain = _names(range(1, 40), range(1, 9))
    marked = [
        f"{n.rpartition('.')[0]}{SELECTED_MARKER}.jpg" if i % 8 == 0 else n
        for i, n in enumerate(plain)
    ]
    assert _ordered(marked), "the SELECTED suffix broke chronological ordering"
    leading = [
        f"{SELECTED_MARKER.lstrip('_')}_{n}" if i % 8 == 0 else n for i, n in enumerate(plain)
    ]
    assert not _ordered(leading), "a leading marker sorted fine, so this proves nothing"


# ---------------------------------------------------------------------------
# The dump itself
# ---------------------------------------------------------------------------


def test_it_holds_the_ranked_set_not_the_searched_pool(tmp_path_factory, sample_wav):
    """Ranking keeps N per segment; the rest of `kept` carries rank 0.

    Including the unranked leftovers would triple the folder with images the
    ranker never endorsed, and numbering them would invent a ranking.
    """
    from visualresearcher.pipeline.image_search import read_candidate_manifest

    ctx, settings = _project(tmp_path_factory, sample_wav, "ranked")
    segments = _segments(ctx)
    planned = plan_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    assert planned, "the run produced no candidates, so this proves nothing"

    for segment in segments:
        manifest = read_candidate_manifest(
            ctx.paths.segment_image_manifest(segment, width=ctx.width), root=ctx.paths.root
        )
        kept = [r for r in manifest.get("kept", []) if r.local_path]
        ranked = [r for r in kept if (r.rank or 0) >= 1]
        mine = [c for c in planned if c.segment_index == segment.index]
        assert len(mine) == len(ranked)
        assert all(c.rank >= 1 for c in mine), "an unranked candidate was given a rank"
        assert len(mine) <= settings.images.keep_per_segment


def test_the_dump_is_ordered_and_grouped_on_disk(tmp_path_factory, sample_wav):
    ctx, settings = _project(tmp_path_factory, sample_wav, "ordered")
    segments = _segments(ctx)
    planned = plan_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    result = dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)

    on_disk = _listing(ctx.paths)
    assert on_disk == [c.name for c in planned]
    assert result.total == len(on_disk)

    prefixes = [n.split("_")[0] for n in on_disk]
    assert prefixes == sorted(prefixes)
    # Each segment is one contiguous block, never interleaved.
    blocks: list[str] = []
    for prefix in prefixes:
        if not blocks or blocks[-1] != prefix:
            assert prefix not in blocks, f"segment {prefix} was split into two blocks"
            blocks.append(prefix)
    for prefix in set(prefixes):
        ranks = [int(n.split("_")[2]) for n in on_disk if n.startswith(prefix + "_")]
        assert ranks == sorted(ranks), f"segment {prefix} is out of rank order"


def test_exactly_the_delivered_images_are_marked(tmp_path_factory, sample_wav):
    """The marker must describe `selected/` as it actually is.

    Compared against the *images* in `selected/`, not its whole contents: a
    segment can also deliver a clip, and a clip has no image candidate to mark.
    Counting clips here would demand a marker that could never exist.
    """
    from visualresearcher.pipeline.collect import plan_selected

    ctx, settings = _project(tmp_path_factory, sample_wav, "marked")
    segments = _segments(ctx)
    dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)

    planned, _gaps = plan_selected(segments, ctx.paths, settings, width=ctx.width)
    delivered_images = [f for f in planned if f.kind == "image"]
    marked = [n for n in _listing(ctx.paths) if SELECTED_MARKER in n]

    assert marked, "nothing was marked, so this proves nothing"
    assert len(marked) == len(delivered_images), (
        f"{len(marked)} file(s) marked SELECTED but selected/ holds "
        f"{len(delivered_images)} image(s)"
    )
    # Each marked file is the exact original that selected/ was built from.
    marked_sources = {
        c.source.resolve()
        for c in plan_all_candidates(segments, ctx.paths, settings, width=ctx.width)
        if c.selected
    }
    expected = {
        (f.source if f.source.is_absolute() else ctx.paths.root / f.source).resolve()
        for f in delivered_images
    }
    assert marked_sources == expected


def test_the_marker_moves_when_the_pick_changes(tmp_path_factory, sample_wav):
    """A swap in the review UI must not leave the folder claiming the old one."""
    ctx, settings = _project(tmp_path_factory, sample_wav, "swap")
    segments = _segments(ctx)
    dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    before = [n for n in _listing(ctx.paths) if SELECTED_MARKER in n]
    assert before, "nothing was marked, so this proves nothing"

    # Swap one segment's pick from rank 1 to rank 2, the way the review UI does.
    target = next(s for s in segments if len([p for p in s.picks if p.kind == "image"]) > 1)
    images = sorted((p for p in target.picks if p.kind == "image"), key=lambda p: p.rank)
    images[0].use = False
    images[1].use = True
    images[1].user_set = True
    ctx.paths.segment_json(target, width=ctx.width).write_text(
        target.model_dump_json(indent=2), encoding="utf-8"
    )

    segments = _segments(ctx)
    result = dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    after = [n for n in _listing(ctx.paths) if SELECTED_MARKER in n]

    moved = set(before) ^ set(after)
    assert moved, "the marker did not move when the pick changed"
    assert result.remarked, "the move was a re-copy rather than a rename"
    assert len(after) == len(before)
    # The newly marked file for that segment is the rank-2 one.
    prefix = f"{target.index:0{ctx.width}d}_"
    now = [n for n in after if n.startswith(prefix)]
    assert now and int(now[0].split("_")[2]) == images[1].rank


def test_rerunning_changes_nothing(tmp_path_factory, sample_wav):
    ctx, settings = _project(tmp_path_factory, sample_wav, "idem")
    segments = _segments(ctx)
    first = dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    second = dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    assert second.total == first.total
    assert not second.written, "a second dump re-copied files that were already correct"
    assert not second.removed, "a second dump deleted files it had just written"
    assert not second.remarked


def test_originals_are_copied_never_moved(tmp_path_factory, sample_wav):
    ctx, settings = _project(tmp_path_factory, sample_wav, "copies")
    segments = _segments(ctx)
    planned = plan_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    for candidate in planned:
        assert candidate.source.exists(), f"the original {candidate.source} was moved or deleted"


def test_a_file_a_person_added_is_left_alone(tmp_path_factory, sample_wav):
    ctx, settings = _project(tmp_path_factory, sample_wav, "foreign")
    segments = _segments(ctx)
    dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    mine = ctx.paths.root / DIR_NAME / "my_own_note.txt"
    mine.write_text("keep me", encoding="utf-8")
    result = dump_all_candidates(segments, ctx.paths, settings, width=ctx.width)
    assert mine.exists(), "the dump deleted a file a person added"
    assert "my_own_note.txt" in result.foreign


# ---------------------------------------------------------------------------
# Per-project isolation
# ---------------------------------------------------------------------------


def test_two_projects_never_share_a_folder(tmp_path_factory, sample_wav):
    """Nothing from one video may appear in another's output."""
    ctx_a, settings_a = _project(tmp_path_factory, sample_wav, "alpha")
    ctx_b, settings_b = _project(tmp_path_factory, sample_wav, "beta")
    dump_all_candidates(_segments(ctx_a), ctx_a.paths, settings_a, width=ctx_a.width)
    dump_all_candidates(_segments(ctx_b), ctx_b.paths, settings_b, width=ctx_b.width)

    a, b = ctx_a.paths.root / DIR_NAME, ctx_b.paths.root / DIR_NAME
    assert a.is_dir() and b.is_dir()
    assert a != b and not a.is_relative_to(b) and not b.is_relative_to(a)
    # Each folder's files live under its own project root, nowhere else.
    for folder, root in ((a, ctx_a.paths.root), (b, ctx_b.paths.root)):
        for f in folder.iterdir():
            assert f.resolve().is_relative_to(root.resolve())


def test_the_dump_stays_inside_the_project(tmp_path_factory, sample_wav):
    """Every path the planner produces is under this project's own root."""
    ctx, settings = _project(tmp_path_factory, sample_wav, "scoped")
    segments = _segments(ctx)
    root = ctx.paths.root.resolve()
    for candidate in plan_all_candidates(segments, ctx.paths, settings, width=ctx.width):
        assert candidate.source.resolve().is_relative_to(root), (
            f"{candidate.source} is outside the project folder"
        )
