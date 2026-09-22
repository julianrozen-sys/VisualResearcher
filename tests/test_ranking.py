"""Ranking, diversity and contact sheets (CLAUDE.md §11, §21).

§21 asks that ranking is deterministic and that diversity is enforced. The
diversity test here is written against the failure §11 names by name: eight
near-identical portraits. It builds a pool where the highest-scoring images
are all the same picture, and requires the selection not to take them.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from visualresearcher.pipeline.rank import (
    DIVERSITY_STRENGTH,
    SegmentHistory,
    build_prompt,
    rank_segment,
    score_records,
    select_diverse,
    sharpness,
    source_tier,
    watermark_penalty,
    write_contact_sheet,
)
from visualresearcher.providers.embedding.base import cosine
from visualresearcher.providers.embedding.fake import FakeEmbeddingProvider
from visualresearcher.providers.images.fake import generate_image
from visualresearcher.schemas import ImageRecord, ProjectContext, Segment


@pytest.fixture
def context() -> ProjectContext:
    return ProjectContext(
        subject="Star Wars: The Old Republic",
        franchise="Star Wars",
        search_tag="SWTOR",
        characters=["Darth Baras", "Darth Vowrawn"],
        places=["Korriban"],
    )


@pytest.fixture
def segment() -> Segment:
    return Segment(
        index=31,
        start=229.0,
        end=237.0,
        narration="Baras is unmasked before the Dark Council.",
        topic="Baras unmasked before the Dark Council",
        entities=["Darth Baras", "Dark Council"],
        event="the unmasking",
        location="Korriban",
        interpretation="Exact cutscene moment; fall back to portrait plus chamber wide.",
    )


def _make_record(path: Path, sandbox: Path, **kwargs) -> ImageRecord:
    from PIL import Image

    from visualresearcher.pipeline.dedupe import compute_hashes
    from visualresearcher.utils.hashing import sha256_file

    with Image.open(path) as image:
        width, height = image.size
    phash, dhash = compute_hashes(path)
    base = {
        "local_path": str(path),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "phash": phash,
        "dhash": dhash,
        "domain": "commons.wikimedia.org",
        "query": "Darth Baras Dark Council",
        "query_kind": "exact_event",
    }
    base.update(kwargs)
    return ImageRecord(**base)


@pytest.fixture
def distinct_records(sandbox) -> list[ImageRecord]:
    records = []
    for i in range(12):
        path = sandbox / f"img_{i:02d}.jpg"
        generate_image(path, 2000 + i * 211, width=1200 + (i % 3) * 200, height=800)
        records.append(_make_record(path, sandbox))
    return records


# ---------------------------------------------------------------------------
# Prompt (§11: interpretation + entities + event + location)
# ---------------------------------------------------------------------------


def test_the_prompt_contains_every_part_section_11_names(segment, context):
    prompt = build_prompt(segment, context)
    assert segment.interpretation in prompt
    assert "Darth Baras" in prompt
    assert segment.event in prompt
    assert segment.location in prompt


def test_the_prompt_falls_back_to_narration(context):
    bare = Segment(index=1, start=0.0, end=8.0, narration="Something happened here.")
    assert "Something happened here." in build_prompt(bare, context)


def test_the_prompt_is_bounded():
    huge = Segment(index=1, start=0.0, end=8.0, interpretation="x " * 2000)
    assert len(build_prompt(huge, ProjectContext())) <= 600


# ---------------------------------------------------------------------------
# Individual components
# ---------------------------------------------------------------------------


def test_source_tier_prefers_wikimedia_over_a_link_aggregator():
    assert source_tier("commons.wikimedia.org") > source_tier("fandom.com")
    assert source_tier("fandom.com") > source_tier("i.imgur.com")
    assert 0.0 < source_tier("something-unknown.example") <= 1.0


def test_sharpness_separates_a_blurred_copy_from_a_sharp_one(sandbox):
    from PIL import Image, ImageFilter

    sharp = sandbox / "sharp.jpg"
    generate_image(sharp, 4242, width=1200, height=800, quality=95)
    blurred = sandbox / "blurred.jpg"
    with Image.open(sharp) as source:
        source.filter(ImageFilter.GaussianBlur(6)).save(blurred, "JPEG", quality=95)

    assert sharpness(sharp) > sharpness(blurred), (
        "a Gaussian-blurred copy must score lower on Laplacian variance"
    )


def test_sharpness_of_an_unreadable_file_is_zero(sandbox):
    broken = sandbox / "broken.jpg"
    broken.write_bytes(b"nope")
    assert sharpness(broken) == 0.0


def test_watermark_penalty_fires_on_a_heavy_border(sandbox):
    """§11: border edge-density, no OCR dependency."""
    from PIL import Image, ImageDraw

    plain = sandbox / "plain.jpg"
    generate_image(plain, 555, width=1200, height=800)

    banded = sandbox / "banded.jpg"
    with Image.open(plain) as source:
        copy = source.copy()
        draw = ImageDraw.Draw(copy)
        # A busy striped band top and bottom, as a site banner would be.
        for y in range(0, 70, 4):
            draw.line([(0, y), (copy.width, y)], fill=(255, 255, 255), width=2)
            draw.line(
                [(0, copy.height - y), (copy.width, copy.height - y)],
                fill=(0, 0, 0),
                width=2,
            )
        copy.save(banded, "JPEG", quality=95)

    assert watermark_penalty(banded) > watermark_penalty(plain), (
        "a striped border band should raise the watermark penalty"
    )


def test_watermark_penalty_is_zero_for_an_ordinary_image(sandbox):
    plain = sandbox / "plain.jpg"
    generate_image(plain, 777, width=1200, height=800)
    assert 0.0 <= watermark_penalty(plain) < 0.5


def test_watermark_penalty_of_an_unreadable_file_is_zero(sandbox):
    broken = sandbox / "b.jpg"
    broken.write_bytes(b"nope")
    assert watermark_penalty(broken) == 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_every_record_gets_a_score_and_a_full_breakdown(
    distinct_records, segment, context, settings
):
    score_records(distinct_records, segment, context, settings, embedder=FakeEmbeddingProvider())
    for record in distinct_records:
        assert record.score >= 0.0
        for key in (
            "clip",
            "entity",
            "exact_event",
            "source",
            "resolution",
            "sharpness",
            "watermark",
            "repetition",
        ):
            assert key in record.score_breakdown, f"{key} missing from the breakdown"


def test_records_come_back_sorted_by_score(distinct_records, segment, context, settings):
    score_records(distinct_records, segment, context, settings)
    scores = [r.score for r in distinct_records]
    assert scores == sorted(scores, reverse=True)


def test_scoring_is_deterministic(distinct_records, segment, context, settings, sandbox):
    """§11: deterministic under a fixed seed."""
    import copy

    first = copy.deepcopy(distinct_records)
    second = copy.deepcopy(distinct_records)
    score_records(first, segment, context, settings, embedder=FakeEmbeddingProvider())
    score_records(second, segment, context, settings, embedder=FakeEmbeddingProvider())
    assert [(r.local_path, r.score) for r in first] == [(r.local_path, r.score) for r in second]


def test_scoring_does_not_depend_on_input_order(distinct_records, segment, context, settings):
    import copy

    forward = copy.deepcopy(distinct_records)
    backward = list(reversed(copy.deepcopy(distinct_records)))
    score_records(forward, segment, context, settings)
    score_records(backward, segment, context, settings)
    assert [r.local_path for r in forward] == [r.local_path for r in backward], (
        "ranking must not depend on the order records arrived in"
    )


def test_an_exact_event_query_outranks_a_fallback_all_else_equal(
    sandbox, segment, context, settings
):
    path = sandbox / "same.jpg"
    generate_image(path, 31337, width=1200, height=800)
    exact = _make_record(path, sandbox, query_kind="exact_event", local_path=str(path))
    fallback = _make_record(path, sandbox, query_kind="fallback", local_path=str(path))
    score_records([exact, fallback], segment, context, settings)
    assert exact.score > fallback.score


def test_a_better_source_outranks_a_worse_one_all_else_equal(sandbox, segment, context, settings):
    path = sandbox / "same.jpg"
    generate_image(path, 999, width=1200, height=800)
    good = _make_record(path, sandbox, domain="commons.wikimedia.org")
    poor = _make_record(path, sandbox, domain="i.imgur.com")
    score_records([good, poor], segment, context, settings)
    assert good.score > poor.score


def test_the_clip_weight_is_dropped_when_the_embedder_is_not_meaningful(
    distinct_records, segment, context, settings
):
    """Scoring 45% of the result on an arbitrary number would be worse than not."""
    import copy

    fake = FakeEmbeddingProvider()
    assert fake.meaningful is False

    with_fake = copy.deepcopy(distinct_records)
    without = copy.deepcopy(distinct_records)
    score_records(with_fake, segment, context, settings, embedder=fake)
    score_records(without, segment, context, settings, embedder=None)

    assert [r.local_path for r in with_fake] == [r.local_path for r in without], (
        "a non-meaningful embedder must not change the ranking"
    )


def test_an_embedder_that_raises_does_not_fail_the_segment(
    distinct_records, segment, context, settings
):
    class _Exploding(FakeEmbeddingProvider):
        meaningful = True

        def embed_text(self, texts):
            raise RuntimeError("model exploded")

    score_records(distinct_records, segment, context, settings, embedder=_Exploding())
    assert all(r.score >= 0 for r in distinct_records)
    assert any("without CLIP" in n for n in segment.notes)


# ---------------------------------------------------------------------------
# Diversity (§11) -- the rule this phase exists for
# ---------------------------------------------------------------------------


def test_eight_near_identical_images_are_not_all_selected(sandbox, segment, context, settings):
    """§11 names this failure explicitly: "never eight near-identical portraits"."""
    from PIL import Image

    base = sandbox / "portrait.jpg"
    generate_image(base, 8888, width=1000, height=1400)

    records: list[ImageRecord] = []
    # Ten slight variations of one picture, each with a high score.
    for i in range(10):
        path = sandbox / f"portrait_{i}.jpg"
        with Image.open(base) as source:
            source.resize((source.width - i * 6, source.height - i * 6), Image.LANCZOS).save(
                path, "JPEG", quality=90 - i
            )
        records.append(
            _make_record(path, sandbox, query_kind="exact_event", domain="commons.wikimedia.org")
        )
    # Ten genuinely different pictures, every one scored lower than the
    # portraits. There have to be at least `keep` distinct pictures available,
    # or "no near-identical pairs" would be arithmetically impossible and the
    # test would be measuring the pool rather than the selection.
    for i in range(10):
        path = sandbox / f"other_{i}.jpg"
        generate_image(path, 6000 + i * 383, width=1600, height=900)
        records.append(_make_record(path, sandbox, query_kind="fallback", domain="i.imgur.com"))

    score_records(records, segment, context, settings)
    # The portraits really are the top-scoring candidates, so a plain top-8
    # would return eight near-identical images.
    assert sum(1 for r in records[:8] if "portrait" in r.local_path) >= 7

    chosen = select_diverse(records, 8)

    from visualresearcher.pipeline.dedupe import hamming

    near_identical = [
        (Path(a.local_path).name, Path(b.local_path).name)
        for i, a in enumerate(chosen)
        for b in chosen[i + 1 :]
        if hamming(a.phash, b.phash) <= 6
    ]
    assert len(near_identical) <= 1, (
        f"{len(near_identical)} near-identical pairs in the final {len(chosen)}: "
        f"{near_identical}; selection is not enforcing diversity"
    )
    portraits = sum(1 for r in chosen if "portrait" in r.local_path)
    assert portraits <= 2, f"{portraits} of the 8 picks are the same portrait"


def test_selection_prefers_a_different_picture_over_a_second_copy(
    sandbox, segment, context, settings
):
    from PIL import Image

    base = sandbox / "a.jpg"
    generate_image(base, 1234, width=1200, height=800)
    twin = sandbox / "a_twin.jpg"
    with Image.open(base) as source:
        source.save(twin, "JPEG", quality=70)
    other = sandbox / "b.jpg"
    generate_image(other, 5678, width=1200, height=800)

    records = [
        _make_record(base, sandbox, query_kind="exact_event"),
        _make_record(twin, sandbox, query_kind="exact_event"),
        _make_record(other, sandbox, query_kind="fallback", domain="i.imgur.com"),
    ]
    score_records(records, segment, context, settings)
    chosen = select_diverse(records, 2)
    paths = {Path(r.local_path).name for r in chosen}
    assert "b.jpg" in paths, (
        "the different picture should beat a near-duplicate even at a lower score"
    )


def test_selection_respects_keep_per_segment(distinct_records, segment, context, settings):
    score_records(distinct_records, segment, context, settings)
    assert len(select_diverse(distinct_records, 8)) == 8
    assert len(select_diverse(distinct_records, 6)) == 6
    assert len(select_diverse(distinct_records, 0)) == 0


def test_selection_cannot_return_more_than_it_was_given(sandbox, segment, context, settings):
    path = sandbox / "only.jpg"
    generate_image(path, 42)
    records = [_make_record(path, sandbox)]
    score_records(records, segment, context, settings)
    assert len(select_diverse(records, 8)) == 1


def test_selection_assigns_ranks_in_order(distinct_records, segment, context, settings):
    score_records(distinct_records, segment, context, settings)
    chosen = select_diverse(distinct_records, 6)
    assert [r.rank for r in chosen] == [1, 2, 3, 4, 5, 6]
    assert all(r.status == "selected" for r in chosen)


def test_selection_is_deterministic(distinct_records, segment, context, settings):
    import copy

    first = copy.deepcopy(distinct_records)
    second = copy.deepcopy(distinct_records)
    score_records(first, segment, context, settings)
    score_records(second, segment, context, settings)
    assert [r.local_path for r in select_diverse(first, 8)] == [
        r.local_path for r in select_diverse(second, 8)
    ]


# ---------------------------------------------------------------------------
# Cross-segment repetition (§11)
# ---------------------------------------------------------------------------


def test_history_penalises_an_image_used_in_a_recent_segment(sandbox):
    path = sandbox / "repeat.jpg"
    generate_image(path, 24680)
    record = _make_record(path, sandbox)

    history = SegmentHistory()
    assert history.repetition(record) == 0.0, "nothing seen yet"

    history.add([record])
    assert history.repetition(record) > 0.5, "an immediate repeat should score high"


def test_history_forgets_beyond_its_window(sandbox):
    path = sandbox / "old.jpg"
    generate_image(path, 13579)
    record = _make_record(path, sandbox)

    history = SegmentHistory(window=2)
    history.add([record])
    other = sandbox / "other.jpg"
    generate_image(other, 97531)
    for _ in range(3):
        history.add([_make_record(other, sandbox)])
    assert history.repetition(record) == 0.0, "the window should have dropped it"


def test_a_repeated_image_loses_rank_to_a_fresh_one(sandbox, segment, context, settings):
    repeated = sandbox / "repeated.jpg"
    generate_image(repeated, 111, width=1400, height=900)
    fresh = sandbox / "fresh.jpg"
    generate_image(fresh, 222, width=1400, height=900)

    history = SegmentHistory()
    history.add([_make_record(repeated, sandbox)])

    records = [
        _make_record(repeated, sandbox, query_kind="exact_event"),
        _make_record(fresh, sandbox, query_kind="exact_event"),
    ]
    score_records(records, segment, context, settings, history=history)
    assert Path(records[0].local_path).name == "fresh.jpg", (
        "an image reused from the previous segment should drop below a fresh one"
    )


# ---------------------------------------------------------------------------
# Whole-segment ranking
# ---------------------------------------------------------------------------


def test_rank_segment_returns_keep_per_segment(distinct_records, segment, context, settings):
    chosen = rank_segment(distinct_records, segment, context, settings)
    assert len(chosen) == settings.images.keep_per_segment


def test_rank_segment_with_no_records_notes_it(segment, context, settings):
    assert rank_segment([], segment, context, settings) == []
    assert any("no images" in n for n in segment.notes)


def test_rank_segment_updates_the_history(distinct_records, segment, context, settings):
    history = SegmentHistory()
    rank_segment(distinct_records, segment, context, settings, history=history)
    assert history.entries, "ranking should record what it picked"


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


def test_the_fake_embedder_groups_similar_images(sandbox):
    from PIL import Image

    base = sandbox / "a.jpg"
    generate_image(base, 321, width=1200, height=800)
    similar = sandbox / "a2.jpg"
    with Image.open(base) as source:
        source.resize((600, 400), Image.LANCZOS).save(similar, "JPEG", quality=80)
    different = sandbox / "b.jpg"
    generate_image(different, 654, width=1200, height=800)

    provider = FakeEmbeddingProvider()
    vectors = provider.embed_images([base, similar, different])
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2]), (
        "a resized copy should embed closer than an unrelated image"
    )


def test_the_fake_embedder_is_deterministic(sandbox):
    path = sandbox / "a.jpg"
    generate_image(path, 852)
    assert FakeEmbeddingProvider().embed_images([path]) == (
        FakeEmbeddingProvider().embed_images([path])
    )
    assert FakeEmbeddingProvider().embed_text(["hello"]) == (
        FakeEmbeddingProvider().embed_text(["hello"])
    )


def test_the_fake_embedder_returns_a_zero_vector_for_an_unreadable_file(sandbox):
    broken = sandbox / "broken.jpg"
    broken.write_bytes(b"nope")
    vector = FakeEmbeddingProvider().embed_images([broken])[0]
    assert len(vector) == FakeEmbeddingProvider.dimensions
    assert not any(vector)


def test_the_fake_embedder_declares_its_similarity_meaningless():
    """This flag is what stops the offline run pretending to a CLIP judgement."""
    assert FakeEmbeddingProvider().meaningful is False


def test_vectors_are_normalised(sandbox):
    path = sandbox / "a.jpg"
    generate_image(path, 963)
    vector = FakeEmbeddingProvider().embed_images([path])[0]
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-6


def test_the_real_clip_provider_reports_its_own_readiness():
    """``doctor`` must be able to ask without importing torch."""
    from visualresearcher.providers.embedding.openclip import OpenClipEmbeddingProvider

    availability = OpenClipEmbeddingProvider().availability()
    assert isinstance(availability.ok, bool)
    if not availability.ok:
        assert availability.missing, "an unavailable provider must say how to fix it"


# ---------------------------------------------------------------------------
# Contact sheet (§6)
# ---------------------------------------------------------------------------


def test_a_contact_sheet_is_written(sandbox, distinct_records, segment, context, settings):
    chosen = rank_segment(distinct_records, segment, context, settings)
    sheet = sandbox / "contact_sheet.jpg"
    assert write_contact_sheet(chosen, sheet) == sheet
    assert sheet.exists() and sheet.stat().st_size > 0

    from PIL import Image

    with Image.open(sheet) as image:
        assert image.width > 0 and image.height > 0


def test_no_contact_sheet_is_written_for_an_empty_segment(sandbox):
    assert write_contact_sheet([], sandbox / "none.jpg") is None
    assert not (sandbox / "none.jpg").exists()


def test_a_missing_image_does_not_break_the_contact_sheet(sandbox, distinct_records):
    distinct_records[0].local_path = str(sandbox / "deleted.jpg")
    for rank, record in enumerate(distinct_records[:4], start=1):
        record.rank = rank
    sheet = sandbox / "sheet.jpg"
    assert write_contact_sheet(distinct_records[:4], sheet) == sheet
    assert sheet.exists()


def test_diversity_strength_is_strong_enough_to_matter():
    assert DIVERSITY_STRENGTH > 0.5, (
        "a weak diversity penalty cannot outweigh a score gap, which is how "
        "eight near-identical portraits get through"
    )
