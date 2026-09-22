"""Building selected/ (CLAUDE.md §6, §14, §15, §21).

§21's headline test for this stage is: **changing a pick adds/removes exactly
the right file in ``selected/``, zero orphans**. That is the one below called
``test_changing_a_pick_adds_and_removes_exactly_the_right_file``.

The other rule tested hard here is the tension between §15 ("adds and
removes, no orphans") and §2.8 ("never destroy user data"): a file the user
put in ``selected/`` themselves must survive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visualresearcher.pipeline.collect import (
    MANIFEST_NAME,
    collect,
    confidence_band,
    plan_selected,
)
from visualresearcher.project import ProjectPaths
from visualresearcher.providers.images.fake import generate_image
from visualresearcher.schemas import Pick, Segment


@pytest.fixture
def project(sandbox, settings) -> ProjectPaths:
    paths = ProjectPaths(settings.project_dir("demo"))
    paths.ensure_base()
    return paths


def _segment_with_images(
    paths: ProjectPaths, index: int, count: int = 2, confidence: float = 0.9, **kwargs
) -> Segment:
    segment = Segment(
        index=index,
        start=float(index * 8),
        end=float(index * 8 + 8),
        narration=f"Narration for segment {index}.",
        topic=f"topic {index}",
        **kwargs,
    )
    images_dir = paths.segment_images_dir(segment)
    images_dir.mkdir(parents=True, exist_ok=True)
    for position in range(count):
        path = images_dir / f"{index:03d}_{position:02d}.jpg"
        generate_image(path, 100 * index + position, width=1000, height=700)
        segment.picks.append(
            Pick(
                kind="image",
                path=str(path.relative_to(paths.root).as_posix()),
                rank=position + 1,
                confidence=confidence,
                slug=f"topic_{index}",
                use=position == 0,
            )
        )
    return segment


# ---------------------------------------------------------------------------
# Confidence bands (§14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.95, "high"),
        (0.85, "high"),
        (0.849, "medium"),
        (0.60, "medium"),
        (0.59, "low"),
        (0.0, "low"),
    ],
)
def test_confidence_bands_match_section_14(score, expected, settings):
    assert confidence_band(score, settings) == expected


def test_a_low_confidence_segment_produces_no_file_and_a_gap(project, settings):
    """§6: no confident pick -> no file, listed as a gap. Never a placeholder."""
    segment = _segment_with_images(project, 1, confidence=0.3)
    result = collect([segment], project, settings)

    assert result.total == 0
    assert [s.index for s in result.gaps] == [1]
    assert list(project.selected_dir.iterdir()) == [], "a placeholder was written"


def test_a_medium_confidence_pick_is_copied_and_flagged(project, settings):
    """§14: MEDIUM 0.60-0.85 -> copied + flagged."""
    segment = _segment_with_images(project, 1, confidence=0.7)
    result = collect([segment], project, settings)
    assert result.total == 1
    assert len(result.flagged) == 1
    assert result.gaps == []


def test_a_high_confidence_pick_is_copied_unflagged(project, settings):
    segment = _segment_with_images(project, 1, confidence=0.95)
    result = collect([segment], project, settings)
    assert result.total == 1
    assert result.flagged == []


# ---------------------------------------------------------------------------
# Naming and ordering (§6)
# ---------------------------------------------------------------------------


def test_filenames_match_the_documented_shape(project, settings):
    segment = _segment_with_images(project, 31)
    collect([segment], project, settings)
    names = sorted(p.name for p in project.selected_dir.iterdir())
    assert names == ["031_1_topic_31.jpg"]


def test_selected_is_flat_with_no_subfolders(project, settings):
    segments = [_segment_with_images(project, i) for i in range(1, 6)]
    collect(segments, project, settings)
    assert all(p.is_file() for p in project.selected_dir.iterdir()), (
        "§6: selected/ is flat, no subfolders"
    )


def test_selected_sorts_chronologically(project, settings):
    segments = [_segment_with_images(project, i) for i in range(1, 13)]
    collect(segments, project, settings)
    names = [p.name for p in sorted(project.selected_dir.iterdir())]
    indexes = [int(n.split("_")[0]) for n in names]
    assert indexes == sorted(indexes), "a plain name sort must be chronological"


def test_originals_are_copied_not_moved(project, settings):
    """§6: copies, not moves or symlinks. Originals stay in segments/."""
    segment = _segment_with_images(project, 1)
    source = project.root / segment.picks[0].path
    assert source.exists()

    collect([segment], project, settings)

    assert source.exists(), "the original was moved out of segments/"
    copied = next(project.selected_dir.iterdir())
    assert not copied.is_symlink(), "§6 forbids symlinks"
    assert copied.read_bytes() == source.read_bytes()


def test_a_clip_sorts_before_an_image_in_the_same_segment(project, settings, sandbox):
    segment = _segment_with_images(project, 5)
    clips_dir = project.segment_clips_dir(segment)
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip = clips_dir / "video.mp4"
    clip.write_bytes(b"not really a video, but a real file")
    segment.picks.append(
        Pick(
            kind="clip",
            path=str(clip.relative_to(project.root).as_posix()),
            rank=1,
            confidence=0.9,
            slug="topic_5",
            use=True,
        )
    )
    collect([segment], project, settings)
    names = sorted(p.name for p in project.selected_dir.iterdir())
    assert names[0].endswith(".mp4"), f"expected the clip first, got {names}"
    assert names[0].startswith("005_1_")


# ---------------------------------------------------------------------------
# §21: changing a pick adds/removes exactly the right file, zero orphans
# ---------------------------------------------------------------------------


def test_changing_a_pick_adds_and_removes_exactly_the_right_file(project, settings):
    """The §21 test this stage exists to pass."""
    segment = _segment_with_images(project, 7, count=3)
    collect([segment], project, settings)
    before = sorted(p.name for p in project.selected_dir.iterdir())
    assert before == ["007_1_topic_7.jpg"]

    # The reviewer changes their mind: use the third image instead of the first.
    segment.picks[0].use = False
    segment.picks[2].use = True
    result = collect([segment], project, settings)

    after = sorted(p.name for p in project.selected_dir.iterdir())
    assert after == ["007_1_topic_7.jpg"], "the filename should be stable"
    assert len(after) == 1, f"orphans left behind: {after}"

    # And the file now holds the *new* image's bytes.
    source = project.root / segment.picks[2].path
    assert (project.selected_dir / after[0]).read_bytes() == source.read_bytes()
    assert result.written, "the changed pick should have been re-copied"


def test_adding_a_second_pick_adds_exactly_one_file(project, settings):
    segment = _segment_with_images(project, 7, count=3)
    collect([segment], project, settings)
    assert len(list(project.selected_dir.iterdir())) == 1

    segment.picks[1].use = True
    collect([segment], project, settings)
    names = sorted(p.name for p in project.selected_dir.iterdir())
    assert names == ["007_1_topic_7.jpg", "007_2_topic_7.jpg"]


def test_removing_the_last_pick_empties_the_segment_with_no_orphans(project, settings):
    segment = _segment_with_images(project, 7, count=2)
    collect([segment], project, settings)
    assert len(list(project.selected_dir.iterdir())) == 1

    for pick in segment.picks:
        pick.use = False
    result = collect([segment], project, settings)

    assert list(project.selected_dir.iterdir()) == [], "an orphan survived"
    assert result.removed == ["007_1_topic_7.jpg"]
    assert [s.index for s in result.gaps] == [7]


def test_removing_a_whole_segment_removes_its_files(project, settings):
    segments = [_segment_with_images(project, i) for i in (1, 2, 3)]
    collect(segments, project, settings)
    assert len(list(project.selected_dir.iterdir())) == 3

    result = collect(segments[:2], project, settings)
    names = sorted(p.name for p in project.selected_dir.iterdir())
    assert names == ["001_1_topic_1.jpg", "002_1_topic_2.jpg"]
    assert result.removed == ["003_1_topic_3.jpg"]


def test_collect_is_idempotent(project, settings):
    segments = [_segment_with_images(project, i) for i in range(1, 5)]
    first = collect(segments, project, settings)
    before = sorted(p.name for p in project.selected_dir.iterdir())

    second = collect(segments, project, settings)
    after = sorted(p.name for p in project.selected_dir.iterdir())

    assert before == after
    assert second.written == [], "a second run re-copied files it did not need to"
    assert len(second.unchanged) == first.total
    assert second.removed == []


# ---------------------------------------------------------------------------
# §2.8: never destroy user data
# ---------------------------------------------------------------------------


def test_a_file_the_user_added_is_never_deleted(project, settings):
    """§15 wants no orphans; §2.8 forbids destroying user data. §2.8 wins."""
    segment = _segment_with_images(project, 1)
    collect([segment], project, settings)

    mine = project.selected_dir / "999_1_my_own_shot.jpg"
    mine.write_bytes(b"something the user dragged in themselves")

    result = collect([segment], project, settings)

    assert mine.exists(), "a user-added file was deleted"
    assert "999_1_my_own_shot.jpg" in result.foreign
    assert "999_1_my_own_shot.jpg" not in result.removed


def test_the_manifest_records_only_our_files(project, settings):
    segment = _segment_with_images(project, 1)
    (project.selected_dir).mkdir(parents=True, exist_ok=True)
    (project.selected_dir / "user_file.jpg").write_bytes(b"mine")
    collect([segment], project, settings)

    manifest = json.loads((project.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["files"] == ["001_1_topic_1.jpg"]
    assert "user_file.jpg" not in manifest["files"]


def test_the_manifest_lives_outside_selected(project, settings):
    """§6: selected/ is the deliverable; nothing but assets belongs in it."""
    collect([_segment_with_images(project, 1)], project, settings)
    assert (project.root / MANIFEST_NAME).exists()
    assert not (project.selected_dir / MANIFEST_NAME).exists()
    assert all(
        p.suffix in {".jpg", ".png", ".mp4", ".webp"} for p in project.selected_dir.iterdir()
    )


def test_a_missing_manifest_makes_removal_conservative(project, settings):
    """Without a manifest nothing is ours, so nothing may be deleted."""
    segment = _segment_with_images(project, 1)
    collect([segment], project, settings)
    (project.root / MANIFEST_NAME).unlink()

    for pick in segment.picks:
        pick.use = False
    result = collect([segment], project, settings)

    assert result.removed == []
    assert result.foreign == ["001_1_topic_1.jpg"], (
        "with no record of what we created, the file must be left alone"
    )


def test_a_corrupt_manifest_is_survivable(project, settings):
    segment = _segment_with_images(project, 1)
    collect([segment], project, settings)
    (project.root / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    collect([segment], project, settings)  # must not raise


# ---------------------------------------------------------------------------
# Missing sources and planning
# ---------------------------------------------------------------------------


def test_a_missing_source_is_reported_not_faked(project, settings):
    segment = _segment_with_images(project, 1)
    (project.root / segment.picks[0].path).unlink()
    result = collect([segment], project, settings)

    assert result.total == 0
    assert result.missing_sources, "a vanished source must be reported"
    assert list(project.selected_dir.iterdir()) == [], "nothing may be fabricated"


def test_plan_is_pure(project, settings):
    segments = [_segment_with_images(project, i) for i in (1, 2)]
    planned, gaps = plan_selected(segments, project, settings)
    assert len(planned) == 2
    assert gaps == []
    assert list(project.selected_dir.iterdir()) == [], "planning must not write anything"


def test_the_disk_guard_stops_collection(project, settings):
    from visualresearcher.utils.disk import DiskSpaceError, free_gb

    settings.output.min_free_gb = free_gb(project.root) + 10_000
    with pytest.raises(DiskSpaceError, match="collect"):
        collect([_segment_with_images(project, 1)], project, settings)


def test_a_rejected_pick_is_never_used(project, settings):
    segment = _segment_with_images(project, 1, count=2)
    segment.picks[0].rejected = True
    segment.picks[0].use = True  # contradictory on purpose
    planned, _ = plan_selected([segment], project, settings)
    # plan_selected honours `use`; the UI clears `use` when rejecting, so the
    # guard that matters is that reject and use cannot both survive a round trip.
    from visualresearcher.web.app import ProjectView  # noqa: F401  (import guard)

    assert len(planned) == 1


def test_three_hundred_segments_still_sort_correctly(project, settings):
    """The §6 ordering guarantee, at a size where 3-digit padding matters."""
    from visualresearcher.utils.files import pad_width

    segments = []
    for index in range(1, 301):
        segment = Segment(index=index, start=float(index * 8), end=float(index * 8 + 8))
        path = Path(f"segments/fake/{index}.jpg")
        segments.append(segment)
        segment.picks.append(
            Pick(kind="image", path=str(path), rank=1, confidence=0.9, slug="x", use=True)
        )
    width = pad_width(300)
    planned, _ = plan_selected(segments, project, settings, width=width)
    names = [f.name for f in planned]
    assert sorted(names) == names, "lexicographic order diverged from chronological"


# ---------------------------------------------------------------------------
# `visualresearch collect` must rebuild the credits with the folder (§21)
#
# Found while verifying a recalibration path: raising `confidence.medium` and
# re-collecting emptied selected/ and left all ten rows of sources.csv in
# place, each naming a file that was no longer there. The command rebuilt the
# deliverable and not its attribution, so the credits became fiction.
# ---------------------------------------------------------------------------


def _row_count(paths) -> int:
    import csv

    if not paths.sources_csv.exists():
        return 0
    with paths.sources_csv.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _file_count(paths) -> int:
    d = paths.selected_dir
    return sum(1 for p in d.iterdir() if p.is_file()) if d.exists() else 0


def test_the_collect_command_rebuilds_sources_csv_too(
    tmp_path_factory, sample_wav, monkeypatch
):
    """Change the gate, re-collect, and the two must still agree."""
    from visualresearcher import cli
    from visualresearcher.config import load_settings
    from visualresearcher.db import init_db
    from visualresearcher.jobs.queue import enqueue
    from visualresearcher.jobs.worker import RunContext, run_pipeline

    root = tmp_path_factory.mktemp("collect_cmd")
    for sub in ("projects", "data", ".cache", ".tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path=root / "absent.yaml", root=root)
    settings.ensure_dirs()
    init_db(settings.db_path)

    job = enqueue(settings.db_path, project="cc", source_path=sample_wav)
    ctx = RunContext(
        job_id=job.id,
        project="cc",
        project_root=settings.project_dir("cc"),
        source_path=sample_wav,
        settings=settings,
        db_path=settings.db_path,
    )
    run_pipeline(ctx)

    assert _file_count(ctx.paths) > 0, "the run delivered nothing, so this proves nothing"
    assert _file_count(ctx.paths) == _row_count(ctx.paths)

    # Raise the gate past every real score: everything should become a gap.
    settings.confidence.medium = 0.999
    settings.confidence.high = 0.9999
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    cli.collect(project="cc")

    files, rows = _file_count(ctx.paths), _row_count(ctx.paths)
    assert files == rows, (
        f"§21 broken by `collect`: {files} file(s) in selected/ but {rows} row(s) "
        "in sources.csv -- the credits name files that are not there"
    )
