"""Image search, download validation and dedupe (CLAUDE.md §11, §21).

§21 asks for: >=15 candidates/segment; oversize, wrong-type and undecodable
rejected; filename sanitised; path traversal blocked; dedupe catching exact,
resized, cropped and recompressed copies.

These tests use real generated image bytes rather than mocks, because the
rules being tested are about bytes: a mock that returns "this is a valid JPEG"
would pass a validator that does not work.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from visualresearcher.pipeline.dedupe import (
    DEFAULT_DISTANCE,
    compute_hashes,
    dedupe_records,
    hamming,
)
from visualresearcher.pipeline.download import (
    ALLOWED_CONTENT_TYPES,
    RejectReason,
    download_candidate,
    download_candidates,
    validate_image_bytes,
)
from visualresearcher.pipeline.image_search import (
    collect_providers,
    search_segment,
    write_candidate_manifest,
)
from visualresearcher.providers.base import Availability, ProviderError
from visualresearcher.providers.images.base import ImageCandidate, ImageSearchProvider
from visualresearcher.providers.images.fake import FakeImageSearchProvider, generate_image
from visualresearcher.schemas import ImageRecord, Query, Segment


@pytest.fixture
def images_dir(sandbox) -> Path:
    path = sandbox / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def provider(sandbox) -> FakeImageSearchProvider:
    return FakeImageSearchProvider(cache_dir=sandbox / "fake_images")


def _jpeg_bytes(width=1200, height=800, seed=1, quality=88) -> bytes:
    from PIL import Image

    tmp = io.BytesIO()
    path = Path(__file__).parent / ".scratch.jpg"
    generate_image(path, seed, width=width, height=height, quality=quality)
    with Image.open(path) as image:
        image.save(tmp, "JPEG", quality=quality)
    path.unlink(missing_ok=True)
    return tmp.getvalue()


def _candidate(url: str, **kwargs) -> ImageCandidate:
    base = {"image_url": url, "provider": "test", "query": "test query"}
    base.update(kwargs)
    return ImageCandidate(**base)


# ---------------------------------------------------------------------------
# Validation (§11)
# ---------------------------------------------------------------------------


def test_a_good_jpeg_validates(settings):
    ok, reason, detail, (width, height), ext = validate_image_bytes(_jpeg_bytes(), settings)
    assert ok, f"{reason}: {detail}"
    assert (width, height) == (1200, 800)
    assert ext == ".jpg"


def test_oversize_is_rejected(settings):
    settings.images.max_bytes = 1000
    ok, reason, _, _, _ = validate_image_bytes(_jpeg_bytes(), settings)
    assert not ok
    assert reason == RejectReason.TOO_LARGE


def test_an_image_below_min_width_is_rejected(settings):
    ok, reason, detail, _, _ = validate_image_bytes(_jpeg_bytes(width=320, height=200), settings)
    assert not ok
    assert reason == RejectReason.TOO_SMALL
    assert "320px" in detail


def test_undecodable_bytes_are_rejected(settings):
    ok, reason, _, _, _ = validate_image_bytes(b"\xff\xd8\xff\xe0 not a jpeg", settings)
    assert not ok
    assert reason == RejectReason.UNDECODABLE


def test_empty_bytes_are_rejected(settings):
    ok, reason, _, _, _ = validate_image_bytes(b"", settings)
    assert not ok
    assert reason == RejectReason.EMPTY


@pytest.mark.parametrize(
    "payload",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800"></svg>',
        b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
        b'   <svg width="10"></svg>',
    ],
)
def test_svg_is_rejected(settings, payload):
    """§11 rejects SVG outright -- it is script-capable markup, not a raster."""
    ok, reason, _, _, _ = validate_image_bytes(payload, settings)
    assert not ok
    assert reason == RejectReason.SVG


def test_html_masquerading_as_an_image_is_rejected(settings):
    ok, reason, _, _, _ = validate_image_bytes(b"<!DOCTYPE html><html>404</html>", settings)
    assert not ok
    assert reason in {RejectReason.UNDECODABLE, RejectReason.BAD_CONTENT_TYPE}


def test_the_content_type_allowlist_excludes_svg():
    assert "image/svg+xml" not in ALLOWED_CONTENT_TYPES
    assert "image/jpeg" in ALLOWED_CONTENT_TYPES


# ---------------------------------------------------------------------------
# Download: filenames and path safety (§11)
# ---------------------------------------------------------------------------


def test_the_saved_filename_ignores_the_remote_name(settings, images_dir, sandbox):
    """§11: never trust a remote filename."""
    source = sandbox / "evil name; rm -rf.jpg"
    generate_image(source, 7)
    outcome = download_candidate(
        _candidate(source.resolve().as_uri()),
        images_dir,
        settings,
        index=3,
        segment_index=31,
    )
    assert outcome.ok, outcome.detail
    name = Path(outcome.record.local_path).name
    assert name == "031_03.jpg", f"filename came from the remote name: {name}"


@pytest.mark.parametrize(
    "hostile_name",
    ["../../escape.jpg", "..\\..\\escape.jpg", "con.jpg", "....//....//escape.jpg"],
)
def test_a_hostile_remote_filename_cannot_escape_the_images_dir(
    settings, images_dir, sandbox, hostile_name
):
    """§11: block path traversal."""
    source = sandbox / "legit.jpg"
    generate_image(source, 11)
    candidate = _candidate(source.resolve().as_uri(), title=hostile_name)
    outcome = download_candidate(candidate, images_dir, settings, index=0, segment_index=1)
    assert outcome.ok, outcome.detail
    written = Path(outcome.record.local_path).resolve()
    assert written.parent == images_dir.resolve(), (
        f"file escaped to {written.parent}, expected {images_dir.resolve()}"
    )
    assert not (sandbox / "escape.jpg").exists()


def test_a_url_ending_in_svg_is_refused_before_download(settings, images_dir, sandbox):
    svg = sandbox / "vector.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    outcome = download_candidate(
        _candidate(svg.resolve().as_uri()), images_dir, settings, index=0, segment_index=1
    )
    assert not outcome.ok
    assert outcome.reason == RejectReason.SVG


def test_a_missing_file_is_a_transport_rejection_not_a_crash(settings, images_dir, sandbox):
    outcome = download_candidate(
        _candidate((sandbox / "nope.jpg").resolve().as_uri()),
        images_dir,
        settings,
        index=0,
        segment_index=1,
    )
    assert not outcome.ok
    assert outcome.reason == RejectReason.TRANSPORT


def test_an_oversize_file_is_refused_before_it_is_written(settings, images_dir, sandbox):
    big = sandbox / "big.jpg"
    generate_image(big, 3, width=1600, height=1200)
    settings.images.max_bytes = 500
    outcome = download_candidate(
        _candidate(big.resolve().as_uri()), images_dir, settings, index=0, segment_index=1
    )
    assert not outcome.ok
    assert list(images_dir.iterdir()) == [], "nothing should have been written"


def test_no_partial_file_is_left_behind(settings, images_dir, sandbox):
    """§12.6's rule, applied to images: write to .tmp/, move on success."""
    source = sandbox / "ok.jpg"
    generate_image(source, 5)
    download_candidate(
        _candidate(source.resolve().as_uri()), images_dir, settings, index=0, segment_index=1
    )
    assert not list(images_dir.glob("*.part"))
    assert not list(settings.tmp_dir.glob("*.part"))


def test_a_downloaded_record_carries_the_full_metadata(settings, images_dir, sandbox):
    """§7's image record: every field populated, licence never invented."""
    source = sandbox / "meta.jpg"
    generate_image(source, 9, width=1400, height=900)
    candidate = _candidate(
        source.resolve().as_uri(),
        source_page="https://commons.wikimedia.org/wiki/File:X",
        creator="A Person",
        license="CC BY-SA 4.0",
        license_url="https://example.invalid/licence",
        query_kind="exact_event",
    )
    outcome = download_candidate(candidate, images_dir, settings, index=1, segment_index=31)
    record = outcome.record
    assert record.width == 1400 and record.height == 900
    assert record.aspect_ratio == pytest.approx(1400 / 900, rel=1e-3)
    assert len(record.sha256) == 64
    assert record.bytes > 0
    assert record.segment_index == 31
    assert record.provider == "test"
    assert record.query_kind == "exact_event"
    assert record.creator == "A Person"
    assert record.license == "CC BY-SA 4.0"
    assert record.domain == "commons.wikimedia.org"


def test_a_missing_licence_is_recorded_as_unknown_not_guessed(settings, images_dir, sandbox):
    """§13: mark licence unknown rather than guessing."""
    source = sandbox / "nolicence.jpg"
    generate_image(source, 13)
    outcome = download_candidate(
        _candidate(source.resolve().as_uri(), license=""),
        images_dir,
        settings,
        index=0,
        segment_index=1,
    )
    assert outcome.record.license == "unknown"


def test_the_same_url_is_not_downloaded_twice(settings, images_dir, sandbox):
    """A cache-busting query string does not make it a different image."""
    source = sandbox / "same.jpg"
    generate_image(source, 17)
    url = source.resolve().as_uri()
    records, rejected = download_candidates(
        [_candidate(url), _candidate(url + "?cache=1"), _candidate(url)],
        images_dir,
        settings,
        segment_index=1,
    )
    assert len(records) == 1, f"fetched the same image {len(records)} times"
    assert len([r for r in rejected if r.reason == RejectReason.DUPLICATE_URL]) == 2


def test_rejections_are_returned_not_swallowed(settings, images_dir, sandbox):
    """§11: record discards, never delete silently."""
    good = sandbox / "good.jpg"
    generate_image(good, 19)
    bad = sandbox / "bad.jpg"
    bad.write_bytes(b"not an image")

    records, rejected = download_candidates(
        [_candidate(good.resolve().as_uri()), _candidate(bad.resolve().as_uri())],
        images_dir,
        settings,
        segment_index=1,
    )
    assert len(records) == 1
    assert len(rejected) == 1
    assert rejected[0].reason == RejectReason.UNDECODABLE
    assert rejected[0].detail, "a rejection must say why"


def test_the_disk_guard_stops_downloading(settings, images_dir):
    from visualresearcher.utils.disk import DiskSpaceError, free_gb

    settings.output.min_free_gb = free_gb(images_dir) + 10_000
    with pytest.raises(DiskSpaceError, match="image download"):
        download_candidates([], images_dir, settings, segment_index=1)


# ---------------------------------------------------------------------------
# Dedupe (§11, §21)
# ---------------------------------------------------------------------------


def _record(path: Path, **kwargs) -> ImageRecord:
    from PIL import Image

    from visualresearcher.utils.hashing import sha256_file

    with Image.open(path) as image:
        width, height = image.size
    base = {
        "local_path": str(path),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    base.update(kwargs)
    return ImageRecord(**base)


def test_hamming_distance_basics():
    assert hamming("ffff", "ffff") == 0
    assert hamming("0000", "0001") == 1
    assert hamming("", "ffff") == 999, "a missing hash must never look like a match"
    assert hamming("ff", "ffff") == 999


def test_dedupe_catches_an_exact_duplicate(sandbox):
    a = sandbox / "a.jpg"
    generate_image(a, 21)
    b = sandbox / "b.jpg"
    b.write_bytes(a.read_bytes())

    result = dedupe_records([_record(a), _record(b)])
    assert len(result.kept) == 1
    assert len(result.duplicates) == 1
    assert "SHA256" in result.duplicates[0].notes


def test_dedupe_catches_a_resized_copy(sandbox):
    from PIL import Image

    a = sandbox / "orig.jpg"
    generate_image(a, 23, width=1600, height=1200)
    b = sandbox / "resized.jpg"
    with Image.open(a) as source:
        source.resize((800, 600), Image.LANCZOS).save(b, "JPEG", quality=85)

    result = dedupe_records([_record(a), _record(b)])
    assert len(result.kept) == 1, "a resize is the same picture"
    assert result.kept[0].width == 1600, "the larger copy should survive"


def test_dedupe_catches_a_cropped_copy(sandbox):
    from PIL import Image

    a = sandbox / "orig.jpg"
    generate_image(a, 29, width=1400, height=1000)
    b = sandbox / "cropped.jpg"
    with Image.open(a) as source:
        inset_x, inset_y = int(source.width * 0.04), int(source.height * 0.04)
        source.crop((inset_x, inset_y, source.width - inset_x, source.height - inset_y)).save(
            b, "JPEG", quality=88
        )

    result = dedupe_records([_record(a), _record(b)])
    assert len(result.kept) == 1, "a small crop is the same picture"


def test_dedupe_catches_a_recompressed_copy(sandbox):
    from PIL import Image

    a = sandbox / "orig.jpg"
    generate_image(a, 31, width=1200, height=800, quality=95)
    b = sandbox / "recompressed.jpg"
    with Image.open(a) as source:
        source.save(b, "JPEG", quality=30)

    assert a.read_bytes() != b.read_bytes(), "the bytes must differ or SHA256 would catch it"
    result = dedupe_records([_record(a), _record(b)])
    assert len(result.kept) == 1


def test_dedupe_keeps_genuinely_different_images(sandbox):
    paths = []
    for i in range(6):
        path = sandbox / f"distinct_{i}.jpg"
        generate_image(path, 500 + i * 97, width=1200, height=800)
        paths.append(path)

    result = dedupe_records([_record(p) for p in paths])
    assert len(result.kept) == 6, (
        f"six different pictures collapsed to {len(result.kept)}; the distance "
        "threshold is too loose"
    )


def test_dedupe_records_what_a_duplicate_duplicates(sandbox):
    a = sandbox / "a.jpg"
    generate_image(a, 37)
    b = sandbox / "b.jpg"
    b.write_bytes(a.read_bytes())

    result = dedupe_records([_record(a), _record(b)])
    duplicate = result.duplicates[0]
    assert duplicate.status == "duplicate"
    assert Path(result.kept[0].local_path).name in duplicate.notes
    assert result.mapping[duplicate.local_path] == result.kept[0].local_path


def test_dedupe_never_deletes_a_file(sandbox):
    """§2.8: never destroy user data."""
    a = sandbox / "a.jpg"
    generate_image(a, 41)
    b = sandbox / "b.jpg"
    b.write_bytes(a.read_bytes())

    dedupe_records([_record(a), _record(b)])
    assert a.exists() and b.exists(), "dedupe must not delete anything"


def test_dedupe_is_deterministic_regardless_of_input_order(sandbox):
    paths = []
    for i in range(5):
        path = sandbox / f"d_{i}.jpg"
        generate_image(path, 700 + i * 53, width=1000 + i * 100, height=700)
        paths.append(path)
    duplicate = sandbox / "d_dup.jpg"
    duplicate.write_bytes(paths[0].read_bytes())
    paths.append(duplicate)

    forward = dedupe_records([_record(p) for p in paths])
    backward = dedupe_records([_record(p) for p in reversed(paths)])
    assert sorted(r.local_path for r in forward.kept) == sorted(
        r.local_path for r in backward.kept
    ), "dedupe survivors must not depend on input order"


def test_an_unhashable_file_does_not_match_everything(sandbox):
    broken = sandbox / "broken.jpg"
    broken.write_bytes(b"not an image at all")
    good = sandbox / "good.jpg"
    generate_image(good, 43)

    assert compute_hashes(broken) == ("", "")
    records = [
        ImageRecord(local_path=str(broken), sha256="a" * 64, width=1, height=1),
        _record(good),
    ]
    result = dedupe_records(records)
    assert len(result.kept) == 2, "an unhashable file must not collapse into another"


def test_the_configured_distance_is_actually_used(sandbox):
    """A wide threshold must collapse images a narrow one keeps apart."""
    paths = []
    for i in range(4):
        path = sandbox / f"t_{i}.jpg"
        generate_image(path, 900 + i * 131, width=1200, height=800)
        paths.append(path)
    records = [_record(p) for p in paths]

    tight = dedupe_records([_record(p) for p in paths], distance=0)
    loose = dedupe_records([_record(p) for p in paths], distance=64)

    assert len(tight.kept) == len(records), "distinct images must survive a strict threshold"
    assert len(loose.kept) == 1, (
        "a maximal threshold should collapse everything; if it does not, the "
        "distance parameter is being ignored"
    )
    assert DEFAULT_DISTANCE == 6


def test_a_clean_downscale_matches_even_at_distance_zero(sandbox):
    """Worth pinning: pHash is scale-invariant by design.

    A LANCZOS downscale of a clean source can produce a byte-identical
    perceptual hash, so distance 0 is not the same as "exact bytes only".
    """
    from PIL import Image

    a = sandbox / "a.jpg"
    generate_image(a, 47, width=1400, height=1000)
    b = sandbox / "b.jpg"
    with Image.open(a) as source:
        source.resize((700, 500), Image.LANCZOS).save(b, "JPEG", quality=80)

    assert len(dedupe_records([_record(a), _record(b)], distance=0).kept) == 1


# ---------------------------------------------------------------------------
# Search orchestration
# ---------------------------------------------------------------------------


def _segment_with_queries(n: int = 5) -> Segment:
    return Segment(
        index=1,
        start=0.0,
        end=8.0,
        narration="Darth Baras before the Dark Council.",
        queries=[
            Query(kind="exact_event", text=f"query number {i} baras dark council") for i in range(n)
        ],
    )


def test_search_returns_candidates_from_every_query(provider, settings):
    segment = _segment_with_queries(4)
    candidates, notes = search_segment(segment, [provider], settings)
    assert notes == []
    assert len(candidates) >= 20
    assert len({c.query for c in candidates}) == 4, "every query should contribute"


def test_search_interleaves_so_one_query_cannot_eat_the_budget(provider, settings):
    segment = _segment_with_queries(3)
    candidates, _ = search_segment(segment, [provider], settings)
    first_five = [c.query for c in candidates[:3]]
    assert len(set(first_five)) == 3, (
        "the first results should come from different queries, not all from the first"
    )


def test_a_failing_provider_does_not_stop_the_segment(provider, settings):
    class _Broken(ImageSearchProvider):
        name = "broken"

        def availability(self):
            return Availability.available("broken")

        def search(self, query, *, limit=30, **kwargs):
            raise ProviderError("upstream is down")

    segment = _segment_with_queries(2)
    candidates, notes = search_segment(segment, [_Broken(), provider], settings)
    assert candidates, "the working provider's results must survive"
    assert notes, "the failure must be recorded, not hidden"
    assert any("broken" in n for n in notes)


def test_a_provider_raising_an_unexpected_type_is_still_contained(provider, settings):
    class _Exploding(ImageSearchProvider):
        name = "exploding"

        def availability(self):
            return Availability.available("exploding")

        def search(self, query, *, limit=30, **kwargs):
            raise ZeroDivisionError("not even a ProviderError")

    segment = _segment_with_queries(2)
    candidates, notes = search_segment(segment, [_Exploding(), provider], settings)
    assert candidates
    assert any("exploding" in n for n in notes)


def test_a_segment_with_no_queries_is_reported(provider, settings):
    segment = Segment(index=1, start=0.0, end=8.0)
    candidates, notes = search_segment(segment, [provider], settings)
    assert candidates == []
    assert notes


def test_no_providers_at_all_is_reported(settings):
    candidates, notes = search_segment(_segment_with_queries(), [], settings)
    assert candidates == []
    assert notes


def test_offline_resolves_every_configured_provider_to_the_fake(settings):
    providers = collect_providers(settings)
    assert len(providers) == 1, "VR_OFFLINE should collapse to a single fake"
    assert providers[0].is_fake


# ---------------------------------------------------------------------------
# The offline fake itself
# ---------------------------------------------------------------------------


def test_the_fake_produces_real_decodable_images(provider, settings):
    candidates = provider.search("darth baras dark council", limit=30)
    good = [c for c in candidates if "note" not in c.extra or not c.extra["note"]]
    assert good
    for candidate in good[:5]:
        path = Path(candidate.image_url.replace("file:///", ""))
        assert path.exists()
        ok, reason, detail, _, _ = validate_image_bytes(path.read_bytes(), settings)
        assert ok, f"{path.name}: {reason} {detail}"


def test_the_fake_is_deterministic(sandbox):
    a = FakeImageSearchProvider(cache_dir=sandbox / "c1").search("same query", limit=20)
    b = FakeImageSearchProvider(cache_dir=sandbox / "c1").search("same query", limit=20)
    assert [c.image_url for c in a] == [c.image_url for c in b]


def test_the_fake_includes_files_that_must_be_rejected(provider):
    notes = {c.extra.get("note", "") for c in provider.search("test query", limit=30)}
    assert {"too small", "undecodable", "svg"} <= notes, (
        "the offline run should exercise the rejection paths too"
    )


def test_the_fake_includes_near_duplicates_for_dedupe_to_find(provider):
    notes = {c.extra.get("note", "") for c in provider.search("test query", limit=30)}
    assert {"exact duplicate", "resized", "cropped", "recompressed"} <= notes


# ---------------------------------------------------------------------------
# Manifest (§11)
# ---------------------------------------------------------------------------


def test_the_manifest_records_discards_as_well_as_survivors(sandbox, settings, images_dir):
    good = sandbox / "g.jpg"
    generate_image(good, 53)
    bad = sandbox / "b.jpg"
    bad.write_bytes(b"nope")

    records, rejected = download_candidates(
        [_candidate(good.resolve().as_uri()), _candidate(bad.resolve().as_uri())],
        images_dir,
        settings,
        segment_index=1,
    )
    result = dedupe_records(records)
    manifest = sandbox / "manifest.json"
    write_candidate_manifest(manifest, result.kept, rejected=rejected, duplicates=result.duplicates)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["counts"]["kept"] == 1
    assert payload["counts"]["rejected"] == 1
    assert payload["rejected"][0]["reason"] == RejectReason.UNDECODABLE
    assert payload["rejected"][0]["detail"]
