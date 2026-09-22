"""sources.csv, shotlist.md and research_report.md (CLAUDE.md §6, §13, §21).

§21's requirement here is blunt: **sources.csv row count == selected/ file
count**. §13's reason for it is bluntier — every asset gets a row, no silent
assets — because this file is the credits list that goes in the video
description, and an asset with no row is an uncredited asset.
"""

from __future__ import annotations

import csv

import pytest

from visualresearcher.pipeline.collect import collect
from visualresearcher.pipeline.image_search import write_candidate_manifest
from visualresearcher.pipeline.report import (
    SOURCES_COLUMNS,
    write_reports,
    write_research_report,
    write_shotlist,
    write_sources_csv,
)
from visualresearcher.project import ProjectPaths
from visualresearcher.providers.images.fake import generate_image
from visualresearcher.schemas import (
    EntityCorrection,
    ImageRecord,
    Pick,
    ProjectContext,
    Segment,
)
from visualresearcher.utils.files import atomic_write_text
from visualresearcher.utils.hashing import sha256_file


@pytest.fixture
def context() -> ProjectContext:
    return ProjectContext(
        subject="Star Wars: The Old Republic",
        franchise="Star Wars",
        era="Old Republic",
        search_tag="SWTOR",
        characters=["Darth Baras"],
        places=["Korriban"],
        domain_packs=["swtor"],
        provider="domain_packs+fake",
        entity_corrections=[
            EntityCorrection(
                original="Barriss",
                resolved="Darth Baras",
                confidence=0.93,
                reason="Sith Warrior + Dark Council context.",
                evidence=["domain_pack:swtor"],
                method="domain_pack_exact",
                applied=True,
            ),
            EntityCorrection(
                original="Maybe",
                resolved="Maybee",
                confidence=0.4,
                reason="uncertain",
                method="phonetic",
                applied=False,
            ),
        ],
    )


@pytest.fixture
def project(sandbox, settings) -> ProjectPaths:
    """Three segments: two with picks, one that will be a gap."""
    paths = ProjectPaths(settings.project_dir("demo"))
    paths.ensure_base()

    for index in (1, 2, 3):
        segment = Segment(
            index=index,
            start=float(index * 8),
            end=float(index * 8 + 8),
            narration=f"Narration for segment {index}.",
            topic=f"topic {index}",
            entities=["Darth Baras"],
            location="Korriban",
            interpretation="Exact moment; fall back to a portrait.",
        )
        images_dir = paths.segment_images_dir(segment)
        images_dir.mkdir(parents=True, exist_ok=True)
        path = images_dir / f"{index:03d}_00.jpg"
        generate_image(path, 700 + index, width=1200, height=800)
        relative = path.relative_to(paths.root).as_posix()

        write_candidate_manifest(
            paths.segment_image_manifest(segment),
            [
                ImageRecord(
                    local_path=relative,
                    image_url="https://example.invalid/pic.jpg",
                    source_page="https://commons.wikimedia.org/wiki/File:Pic",
                    domain="commons.wikimedia.org",
                    creator="A Contributor",
                    license="CC BY-SA 4.0",
                    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                    query="Darth Baras SWTOR",
                    query_kind="exact_event",
                    width=1200,
                    height=800,
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    segment_index=index,
                    rank=1,
                    score=0.8,
                )
            ],
            root=paths.root,
        )

        # Segment 3 gets a pick nobody is confident about -> a gap.
        segment.picks.append(
            Pick(
                kind="image",
                path=relative,
                rank=1,
                confidence=0.2 if index == 3 else 0.8,
                slug=f"topic_{index}",
                use=True,
            )
        )
        paths.ensure_segment(segment)
        atomic_write_text(
            paths.segment_dir(segment) / "segment.json", segment.model_dump_json(indent=2)
        )
    return paths


def _segments(paths: ProjectPaths) -> list[Segment]:
    return [
        Segment.model_validate_json((d / "segment.json").read_text(encoding="utf-8"))
        for d in paths.existing_segment_dirs()
    ]


def _rows(paths: ProjectPaths) -> list[dict]:
    with open(paths.sources_csv, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# §21: sources.csv row count == selected/ file count
# ---------------------------------------------------------------------------


def test_sources_csv_has_exactly_one_row_per_selected_file(project, settings, context):
    """The §21 requirement, stated literally."""
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_reports(result, segments, context, project, settings)

    files = [p.name for p in project.selected_dir.iterdir() if p.is_file()]
    rows = _rows(project)
    assert len(rows) == len(files), (
        f"sources.csv has {len(rows)} row(s) but selected/ holds {len(files)} file(s)"
    )
    assert {r["file"] for r in rows} == set(files), "the rows do not name the actual files"


def test_the_count_still_matches_after_a_pick_changes(project, settings, context):
    segments = _segments(project)
    collect(segments, project, settings)

    # Promote the gap segment's pick by hand, as the review UI would.
    segments[2].picks[0].user_set = True
    result = collect(segments, project, settings)
    write_reports(result, segments, context, project, settings)

    files = [p.name for p in project.selected_dir.iterdir() if p.is_file()]
    assert len(_rows(project)) == len(files) == 3


def test_every_asset_gets_a_row_even_without_metadata(project, settings, context):
    """§13: no silent assets."""
    segments = _segments(project)
    # Remove the manifest so nothing is known about segment 1's image.
    project.segment_image_manifest(segments[0]).unlink()

    result = collect(segments, project, settings)
    write_reports(result, segments, context, project, settings)

    rows = _rows(project)
    files = [p.name for p in project.selected_dir.iterdir() if p.is_file()]
    assert len(rows) == len(files)
    orphaned = [r for r in rows if r["segment"] == "001"]
    assert orphaned, "the asset with no metadata still needs a row"
    assert orphaned[0]["license"] == "unknown", "§13: mark unknown rather than guess"


# ---------------------------------------------------------------------------
# sources.csv shape (§13)
# ---------------------------------------------------------------------------


def test_the_columns_match_section_13(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_sources_csv(result, segments, project)

    with open(project.sources_csv, encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == SOURCES_COLUMNS
    assert header == [
        "segment",
        "timecode",
        "file",
        "type",
        "source_url",
        "source_page",
        "channel_or_creator",
        "timestamp",
        "license",
        "license_url",
        "query",
    ]


def test_rows_carry_real_attribution(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_sources_csv(result, segments, project)

    row = _rows(project)[0]
    assert row["source_page"] == "https://commons.wikimedia.org/wiki/File:Pic"
    assert row["channel_or_creator"] == "A Contributor"
    assert row["license"] == "CC BY-SA 4.0"
    assert row["license_url"].startswith("https://creativecommons.org")
    assert row["query"] == "Darth Baras SWTOR"
    assert row["type"] == "image"


def test_a_licence_is_never_invented(project, settings, context):
    """§13: mark license unknown rather than guessing."""
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_sources_csv(result, segments, project)
    for row in _rows(project):
        assert row["license"], "an empty licence cell is worse than 'unknown'"


def test_rows_are_in_chronological_order(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_sources_csv(result, segments, project)
    segment_numbers = [r["segment"] for r in _rows(project)]
    assert segment_numbers == sorted(segment_numbers)


def test_the_csv_is_utf8_and_excel_safe(project, settings, context):
    segments = _segments(project)
    segments[0].narration = 'Ünïcödé, with a comma and "quotes"'
    result = collect(segments, project, settings)
    write_sources_csv(result, segments, project)
    text = project.sources_csv.read_text(encoding="utf-8")
    assert text.count("\r") == 0, "mixed line endings break some importers"
    assert _rows(project), "the file should still parse"


# ---------------------------------------------------------------------------
# shotlist.md (§6)
# ---------------------------------------------------------------------------


def test_the_shotlist_lists_gaps(project, settings, context):
    """§6: no confident pick -> listed as a gap in shotlist.md."""
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_shotlist(result, segments, project, settings)

    text = project.shotlist_md.read_text(encoding="utf-8")
    assert "## Gaps" in text
    assert "1 gap(s)" in text
    assert "003" in text, "the gap segment should be named"
    assert "GAP" in text


def test_the_shotlist_says_so_when_there_are_no_gaps(project, settings, context):
    segments = _segments(project)
    for segment in segments:
        segment.picks[0].confidence = 0.9
    result = collect(segments, project, settings)
    write_shotlist(result, segments, project, settings)

    text = project.shotlist_md.read_text(encoding="utf-8")
    assert "0 gap(s)" in text
    assert "## Gaps" not in text


def test_the_shotlist_names_every_selected_file(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_shotlist(result, segments, project, settings)

    text = project.shotlist_md.read_text(encoding="utf-8")
    for path in project.selected_dir.iterdir():
        assert path.name in text, f"{path.name} is missing from the shot list"


def test_the_shotlist_flags_medium_confidence(project, settings, context):
    segments = _segments(project)
    for segment in segments:
        segment.picks[0].confidence = 0.7
    result = collect(segments, project, settings)
    write_shotlist(result, segments, project, settings)
    assert "Flagged" in project.shotlist_md.read_text(encoding="utf-8")


def test_a_pipe_in_the_narration_does_not_break_the_table(project, settings, context):
    segments = _segments(project)
    segments[2].narration = "A narration with a | pipe in it"
    result = collect(segments, project, settings)
    write_shotlist(result, segments, project, settings)
    text = project.shotlist_md.read_text(encoding="utf-8")
    assert "\\|" in text, "an unescaped pipe would split the table cell"


# ---------------------------------------------------------------------------
# research_report.md
# ---------------------------------------------------------------------------


def test_the_research_report_explains_the_corrections(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_research_report(result, segments, context, project, settings)

    text = project.research_report.read_text(encoding="utf-8")
    assert "Barriss" in text and "Darth Baras" in text
    assert "Sith Warrior + Dark Council context." in text, "the reason must be shown"
    assert "0.93" in text
    assert "domain_pack_exact" in text


def test_the_research_report_distinguishes_applied_from_not(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_research_report(result, segments, context, project, settings)

    text = project.research_report.read_text(encoding="utf-8")
    assert "were **not** applied" in text, "§10.3's sub-threshold behaviour should be explained"
    assert "search queries only" in text, "§10.2 should be stated"


def test_the_research_report_states_the_subject_and_packs(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_research_report(result, segments, context, project, settings)

    text = project.research_report.read_text(encoding="utf-8")
    assert "Star Wars: The Old Republic" in text
    assert "swtor" in text
    assert "SWTOR" in text


def test_the_research_report_reports_degraded_segments(project, settings, context):
    segments = _segments(project)
    segments[1].status = "degraded"
    segments[1].notes.append("ddgs failed on query 'x'")
    result = collect(segments, project, settings)
    write_research_report(result, segments, context, project, settings)

    text = project.research_report.read_text(encoding="utf-8")
    assert "Degraded" in text
    assert "ddgs failed" in text


def test_the_research_report_explains_the_bands(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    write_research_report(result, segments, context, project, settings)
    text = project.research_report.read_text(encoding="utf-8")
    assert "0.85" in text and "0.60" in text
    assert "review UI is copied regardless" in text


# ---------------------------------------------------------------------------
# All three together
# ---------------------------------------------------------------------------


def test_write_reports_creates_all_three_files(project, settings, context):
    segments = _segments(project)
    result = collect(segments, project, settings)
    rows = write_reports(result, segments, context, project, settings)

    for path in (project.sources_csv, project.shotlist_md, project.research_report):
        assert path.exists(), f"{path.name} was not written"
        assert path.stat().st_size > 0, f"{path.name} is empty"
    assert rows == len([p for p in project.selected_dir.iterdir() if p.is_file()])


def test_reports_survive_an_empty_project(sandbox, settings, context):
    paths = ProjectPaths(settings.project_dir("empty"))
    paths.ensure_base()
    result = collect([], paths, settings)
    rows = write_reports(result, [], context, paths, settings)

    assert rows == 0
    assert paths.sources_csv.exists()
    assert "0 gap(s)" in paths.shotlist_md.read_text(encoding="utf-8")


def test_reports_are_utf8(project, settings, context):
    segments = _segments(project)
    segments[0].narration = "Ünïcödé — em dash and ellipsis…"
    result = collect(segments, project, settings)
    write_reports(result, segments, context, project, settings)
    for path in (project.sources_csv, project.shotlist_md, project.research_report):
        path.read_text(encoding="utf-8")  # would raise on a bad encoding
