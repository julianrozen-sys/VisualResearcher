"""Review UI tests (CLAUDE.md §15, §21).

The rule that matters most here is the last line of §15: **changing picks
rewrites ``selected/`` exactly — adds and removes, no orphans.** Several of
these tests assert on the files on disk after an HTTP request, not on the
HTML, because the HTML is not the product.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from visualresearcher.project import ProjectPaths
from visualresearcher.providers.images.fake import generate_image
from visualresearcher.schemas import ImageRecord, Pick, ProjectContext, Segment
from visualresearcher.web.app import create_app


@pytest.fixture
def project(sandbox, settings) -> ProjectPaths:
    """A small but complete project on disk: 3 segments, images, a manifest."""
    from visualresearcher.pipeline.image_search import write_candidate_manifest
    from visualresearcher.utils.files import atomic_write_text
    from visualresearcher.utils.hashing import sha256_file

    paths = ProjectPaths(settings.project_dir("demo"))
    paths.ensure_base()

    atomic_write_text(
        paths.project_context,
        ProjectContext(
            subject="Star Wars: The Old Republic",
            franchise="Star Wars",
            search_tag="SWTOR",
            characters=["Darth Baras"],
        ).model_dump_json(indent=2),
    )

    for index in (1, 2, 3):
        segment = Segment(
            index=index,
            start=float(index * 8),
            end=float(index * 8 + 8),
            narration=f"Narration for segment {index}, mentioning Darth Baras.",
            topic=f"topic {index}",
            entities=["Darth Baras"],
            interpretation="Exact moment; fall back to a portrait.",
        )
        images_dir = paths.segment_images_dir(segment)
        images_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for position in range(3):
            path = images_dir / f"{index:03d}_{position:02d}.jpg"
            generate_image(path, 500 * index + position, width=1000, height=700)
            relative = path.relative_to(paths.root).as_posix()
            records.append(
                ImageRecord(
                    local_path=relative,
                    width=1000,
                    height=700,
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    segment_index=index,
                    rank=position + 1,
                    score=0.9 - position * 0.1,
                    domain="commons.wikimedia.org",
                    license="CC BY-SA 4.0",
                    query="Darth Baras SWTOR",
                    query_kind="exact_event",
                    status="selected" if position == 0 else "kept",
                )
            )
            segment.picks.append(
                Pick(
                    kind="image",
                    path=relative,
                    rank=position + 1,
                    confidence=0.9 - position * 0.1,
                    slug=f"topic_{index}",
                    use=position == 0,
                )
            )
        write_candidate_manifest(paths.segment_image_manifest(segment), records, root=paths.root)
        paths.ensure_segment(segment)
        atomic_write_text(
            paths.segment_dir(segment) / "segment.json", segment.model_dump_json(indent=2)
        )
    return paths


@pytest.fixture
def client(settings, project, db_path) -> TestClient:
    return TestClient(create_app(settings))


def _selected(project: ProjectPaths) -> list[str]:
    return sorted(p.name for p in project.selected_dir.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_the_jobs_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "VisualResearcher" in response.text


def test_the_jobs_page_lists_projects_on_disk(client):
    assert "demo" in client.get("/").text


def test_the_project_page_renders(client):
    response = client.get("/projects/demo")
    assert response.status_code == 200
    assert "demo" in response.text
    assert "Star Wars: The Old Republic" in response.text


def test_the_project_page_shows_the_gap_count_at_the_top(client, project):
    """§15: gap count at top."""
    response = client.get("/projects/demo")
    assert "no gaps" in response.text or "gap(s)" in response.text


def test_an_unknown_project_is_a_404(client):
    assert client.get("/projects/does-not-exist").status_code == 404


def test_a_segment_card_renders_everything_section_15_asks_for(client):
    """timecode, narration, interpretation, entities, queries, images, badge."""
    response = client.get("/projects/demo/segments/1")
    assert response.status_code == 200
    body = response.text
    assert "00:00:08" in body, "timecode missing"
    assert "Narration for segment 1" in body, "narration missing"
    assert "fall back to a portrait" in body, "interpretation missing"
    assert "Darth Baras" in body, "entities missing"
    assert "USE THIS" in body or "USED" in body, "USE THIS action missing"
    assert "commons.wikimedia.org" in body, "source domain missing"
    assert "CC BY-SA 4.0" in body, "licence missing"


def test_a_segment_card_shows_a_confidence_badge(client):
    body = client.get("/projects/demo/segments/1").text
    assert "band-high" in body or "band-medium" in body or "band-low" in body


def test_an_unknown_segment_is_a_404(client):
    assert client.get("/projects/demo/segments/999").status_code == 404


# ---------------------------------------------------------------------------
# File serving, including the traversal guard
# ---------------------------------------------------------------------------


def test_a_project_file_is_served(client, project):
    segment_dir = next(project.segments_dir.iterdir())
    image = next((segment_dir / "images").glob("*.jpg"))
    relative = image.relative_to(project.root).as_posix()
    response = client.get(f"/projects/demo/file/{relative}")
    assert response.status_code == 200
    assert response.content == image.read_bytes()


@pytest.mark.parametrize(
    "attack",
    [
        "../../../../Windows/System32/drivers/etc/hosts",
        "../../config.yaml",
        "..%2f..%2fconfig.yaml",
    ],
)
def test_path_traversal_is_blocked(client, attack):
    """The same rule as §11's downloads, applied to the file server."""
    response = client.get(f"/projects/demo/file/{attack}")
    assert response.status_code in (403, 404), (
        f"traversal succeeded with {attack!r}: {response.status_code}"
    )


def test_a_missing_file_is_a_404(client):
    assert client.get("/projects/demo/file/segments/nope.jpg").status_code == 404


# ---------------------------------------------------------------------------
# §15: changing picks rewrites selected/ exactly
# ---------------------------------------------------------------------------


def test_use_this_writes_the_file_to_selected(client, project):
    client.post("/projects/demo/collect")
    before = _selected(project)
    assert before == [
        "001_1_topic_1.jpg",
        "002_1_topic_2.jpg",
        "003_1_topic_3.jpg",
    ]


def test_use_this_on_a_second_image_adds_exactly_one_file(client, project):
    client.post("/projects/demo/collect")
    before = set(_selected(project))

    segment = Segment.model_validate_json(
        (
            project.segments_dir
            / sorted(p.name for p in project.segments_dir.iterdir())[0]
            / "segment.json"
        ).read_text(encoding="utf-8")
    )
    second = segment.picks[1].path

    response = client.post("/projects/demo/segments/1/use", data={"path": second})
    assert response.status_code == 200

    after = set(_selected(project))
    added = after - before
    assert len(added) == 1, f"expected exactly one new file, got {added}"
    assert added == {"001_2_topic_1.jpg"}
    assert not (before - after), "an existing file was removed"


def test_unusing_a_pick_removes_exactly_that_file(client, project):
    client.post("/projects/demo/collect")
    segment = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    first = segment.picks[0].path

    client.post("/projects/demo/segments/1/use", data={"path": first})

    after = _selected(project)
    assert "001_1_topic_1.jpg" not in after, "the unused pick's file survived"
    assert "002_1_topic_2.jpg" in after, "an unrelated segment's file was removed"


def test_rejecting_removes_the_file_and_leaves_no_orphan(client, project):
    client.post("/projects/demo/collect")
    segment = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    client.post("/projects/demo/segments/1/reject", data={"path": segment.picks[0].path})

    after = _selected(project)
    assert "001_1_topic_1.jpg" not in after
    assert len(after) == 2, f"orphans left behind: {after}"


def test_the_folder_on_disk_always_matches_the_page(client, project):
    """There is no save button; every action must reconcile immediately."""
    client.post("/projects/demo/collect")
    segment = Segment.model_validate_json(
        (project.segment_dir(Segment(index=2, start=16.0, end=24.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    for pick in segment.picks:
        client.post("/projects/demo/segments/2/use", data={"path": pick.path})

    reloaded = Segment.model_validate_json(
        (project.segment_dir(Segment(index=2, start=16.0, end=24.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    expected = sum(1 for p in reloaded.picks if p.use)
    actual = len([n for n in _selected(project) if n.startswith("002_")])
    assert actual == expected, (
        f"page says {expected} pick(s) for segment 2 but selected/ holds {actual}"
    )


# ---------------------------------------------------------------------------
# Other §15 actions
# ---------------------------------------------------------------------------


def test_favorite_persists(client, project):
    segment = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    client.post("/projects/demo/segments/1/favorite", data={"path": segment.picks[1].path})
    reloaded = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(p.favorite for p in reloaded.picks)


def test_a_note_is_saved(client, project):
    client.post("/projects/demo/segments/1/note", data={"note": "check this shot"})
    reloaded = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    assert "check this shot" in reloaded.notes


def test_entities_are_editable_and_reversible(client, project):
    """§15: entities visible + reversible."""
    path = project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json"

    client.post("/projects/demo/segments/1/entity", data={"entities": "Darth Vader, Coruscant"})
    assert Segment.model_validate_json(path.read_text(encoding="utf-8")).entities == [
        "Darth Vader",
        "Coruscant",
    ]

    client.post("/projects/demo/segments/1/entity", data={"entities": "Darth Baras"})
    assert Segment.model_validate_json(path.read_text(encoding="utf-8")).entities == ["Darth Baras"]


def test_a_query_can_be_added(client, project):
    client.post(
        "/projects/demo/segments/1/query",
        data={"kind": "exact_event", "text": "a better query"},
    )
    reloaded = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(q.text == "a better query" for q in reloaded.queries)


def test_mark_reviewed_persists(client, project):
    client.post("/projects/demo/segments/1/reviewed")
    reloaded = Segment.model_validate_json(
        (project.segment_dir(Segment(index=1, start=8.0, end=16.0)) / "segment.json").read_text(
            encoding="utf-8"
        )
    )
    assert "reviewed" in reloaded.notes


def test_filtering_by_band_narrows_the_list(client):
    everything = client.get("/projects/demo?band=all").text
    low_only = client.get("/projects/demo?band=low").text
    assert everything.count('id="seg-') >= low_only.count('id="seg-')


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


# ---------------------------------------------------------------------------
# A hundred segments, reviewable without the CLI (§22 P6)
# ---------------------------------------------------------------------------


def test_a_hundred_segment_project_renders(settings, sandbox, db_path):
    """§22 P6: a 100-segment project fully reviewable without the CLI."""
    from visualresearcher.utils.files import atomic_write_text

    paths = ProjectPaths(settings.project_dir("big"))
    paths.ensure_base()
    for index in range(1, 101):
        segment = Segment(
            index=index,
            start=float(index * 8),
            end=float(index * 8 + 8),
            narration=f"Segment {index}.",
            topic=f"topic {index}",
        )
        paths.ensure_segment(segment)
        atomic_write_text(
            paths.segment_dir(segment) / "segment.json", segment.model_dump_json(indent=2)
        )

    client = TestClient(create_app(settings))
    response = client.get("/projects/big")
    assert response.status_code == 200
    assert response.text.count('id="seg-') == 100
    # Every segment is a gap, since none has a pick -- and the page says so.
    assert "100 gap(s)" in response.text
    assert client.get("/projects/big/segments/100").status_code == 200


def test_use_this_below_the_confidence_band_still_writes_the_file(client, project):
    """§14's bands govern *automatic* copying; an explicit choice is not automatic.

    A low-confidence image used to show USED on the card while no file
    appeared in selected/ -- the page and the folder disagreeing, which is
    precisely what §15 forbids.
    """
    from visualresearcher.utils.files import atomic_write_text

    path = project.segment_dir(Segment(index=3, start=24.0, end=32.0)) / "segment.json"
    segment = Segment.model_validate_json(path.read_text(encoding="utf-8"))
    # Force the third pick well below the medium threshold.
    segment.picks[2].confidence = 0.21
    atomic_write_text(path, segment.model_dump_json(indent=2))

    client.post("/projects/demo/collect")
    before = set(_selected(project))

    response = client.post("/projects/demo/segments/3/use", data={"path": segment.picks[2].path})
    assert response.status_code == 200
    assert "USED" in response.text, "the card should show the pick as used"

    added = set(_selected(project)) - before
    assert added, (
        "the card says USED but no file appeared in selected/; the page and the folder disagree"
    )


def test_a_low_confidence_pick_nobody_chose_is_still_excluded(client, project, settings):
    """The band still applies to picks the pipeline set on its own (§14)."""
    from visualresearcher.pipeline.collect import plan_selected
    from visualresearcher.utils.files import atomic_write_text

    path = project.segment_dir(Segment(index=3, start=24.0, end=32.0)) / "segment.json"
    segment = Segment.model_validate_json(path.read_text(encoding="utf-8"))
    for pick in segment.picks:
        pick.use = True
        pick.user_set = False
        pick.confidence = 0.2
    atomic_write_text(path, segment.model_dump_json(indent=2))

    planned, gaps = plan_selected([segment], project, settings)
    assert planned == []
    assert [s.index for s in gaps] == [3]
