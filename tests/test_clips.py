"""Video search, timestamps and sectioned clip download (CLAUDE.md §12, §21).

§21 asks for four things here:

* ``--download-sections`` range matches the located timestamp,
* the full-download guard refuses long sources,
* no partial file is left in ``clips/``,
* the cache prevents a re-download.

The range test is made against a **real trimmed mp4** measured with ffprobe,
not against a mocked return value -- otherwise it would pass whatever the
span arithmetic did.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from visualresearcher.pipeline.clips import (
    DOWNLOADABLE_CLASSES,
    MIN_CONFIDENCE,
    ClipCache,
    download_clip_for_segment,
    span_for,
)
from visualresearcher.pipeline.timestamps import (
    METHOD_CONFIDENCE,
    from_captions,
    from_chapters,
    from_description,
    from_metadata,
    from_similarity,
    locate_timestamps,
    parse_description_timestamps,
)
from visualresearcher.pipeline.video_search import classify_hit, search_videos
from visualresearcher.providers.video.base import SectionRequest, VideoHit
from visualresearcher.providers.video.fake import (
    SOURCE_DURATION,
    FakeVideoProvider,
    probe_duration,
)
from visualresearcher.providers.video.ytdlp import (
    build_search_command,
    build_section_command,
)
from visualresearcher.schemas import (
    Classification,
    Query,
    Segment,
    TimestampCandidate,
    YouTubeRecord,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for the clip tests"
)


@pytest.fixture
def provider(sandbox) -> FakeVideoProvider:
    return FakeVideoProvider(cache_dir=sandbox / "fake_videos")


@pytest.fixture
def segment() -> Segment:
    return Segment(
        index=31,
        start=229.0,
        end=237.0,
        narration="Darth Baras is unmasked before the Dark Council.",
        topic="Darth Baras unmasked",
        entities=["Darth Baras", "Dark Council"],
        event="the unmasking",
        location="Korriban",
        queries=[
            Query(kind="youtube", text="Darth Baras Dark Council SWTOR cutscene"),
        ],
    )


def _hit(**kwargs) -> VideoHit:
    base = {
        "video_id": "abc12345678",
        "url": "https://www.youtube.com/watch?v=abc12345678",
        "title": "Darth Baras Dark Council cutscene",
        "channel": "Test Channel",
        "duration": 180.0,
    }
    base.update(kwargs)
    return VideoHit(**base)


# ---------------------------------------------------------------------------
# The yt-dlp commands (§12.1, §12.4)
# ---------------------------------------------------------------------------


def test_search_uses_ytsearch_with_no_key():
    """§12.1: ytsearchN:query -- no key, no quota."""
    command = build_search_command("darth baras cutscene", 5)
    assert command[0] == "yt-dlp"
    assert command[1] == "ytsearch5:darth baras cutscene"
    assert not any("key" in part.lower() for part in command)


def test_the_section_command_matches_section_12_4():
    request = SectionRequest(
        video_id="abc",
        url="https://www.youtube.com/watch?v=abc",
        start_s=102.0,
        end_s=118.5,
        destination=Path("out"),
        max_height=1080,
    )
    command = build_section_command(request, "out/%(id)s.%(ext)s")
    joined = " ".join(command)

    assert "--download-sections" in command
    assert "*00:01:42.00-00:01:58.50" in command, (
        f"the span is not encoded as yt-dlp expects: {joined}"
    )
    assert "--force-keyframes-at-cuts" in command
    assert "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" in command
    assert "-o" in command


def test_the_section_command_honours_max_height():
    request = SectionRequest(
        video_id="a", url="u", start_s=0, end_s=5, destination=Path("o"), max_height=720
    )
    assert "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" in build_section_command(
        request, "o"
    )


def test_the_section_command_never_asks_for_the_whole_video():
    request = SectionRequest(video_id="a", url="u", start_s=10, end_s=20, destination=Path("o"))
    command = build_section_command(request, "o")
    assert "--download-sections" in command, (
        "without this flag yt-dlp fetches the entire video, which is the exact "
        "failure §12.4 exists to prevent"
    )


# ---------------------------------------------------------------------------
# Timestamp location (§12.3)
# ---------------------------------------------------------------------------


def test_the_method_order_matches_section_12_3():
    """captions > chapters > description > metadata > similarity."""
    order = ["captions", "chapters", "description", "metadata", "similarity"]
    confidences = [METHOD_CONFIDENCE[m] for m in order]
    assert confidences == sorted(confidences, reverse=True)


def test_captions_locate_the_moment(segment):
    hit = _hit(
        captions=[
            {"start": 0.0, "end": 4.0, "text": "unrelated opening"},
            {"start": 62.0, "end": 66.0, "text": "Darth Baras stands before the Dark Council"},
        ]
    )
    candidate = from_captions(hit, segment)
    assert candidate is not None
    assert candidate.start_s == 62.0
    assert candidate.method == "captions"
    assert "Darth Baras" in candidate.evidence
    assert candidate.confidence > 0


def test_chapters_locate_the_moment(segment):
    hit = _hit(
        chapters=[
            {"start_time": 0.0, "end_time": 40.0, "title": "Intro"},
            {"start_time": 40.0, "end_time": 90.0, "title": "Darth Baras unmasked"},
        ]
    )
    candidate = from_chapters(hit, segment)
    assert candidate is not None
    assert candidate.start_s == 40.0
    assert candidate.method == "chapters"


def test_description_timestamps_are_parsed():
    entries = parse_description_timestamps(
        "Chapters:\n0:00 Intro\n1:42 Darth Baras unmasked\n1:02:03 Much later\nnot a timestamp"
    )
    assert (0.0, "Intro") in entries
    assert (102.0, "Darth Baras unmasked") in entries
    assert (3723.0, "Much later") in entries
    assert len(entries) == 3


def test_description_locates_the_moment(segment):
    hit = _hit(description="0:00 Intro\n1:42 Darth Baras before the Dark Council\n")
    candidate = from_description(hit, segment)
    assert candidate is not None
    assert candidate.start_s == 102.0
    assert candidate.method == "description"


def test_metadata_only_fires_for_a_short_video(segment):
    assert from_metadata(_hit(duration=120.0), segment) is not None
    assert from_metadata(_hit(duration=2400.0), segment) is None, (
        "'somewhere near the start' is not a defensible answer for a 40-minute video"
    )


def test_similarity_is_the_lowest_confidence_fallback(segment):
    candidate = from_similarity(_hit(duration=3600.0), segment)
    assert candidate is not None
    assert candidate.method == "similarity"
    assert candidate.confidence <= METHOD_CONFIDENCE["similarity"]
    assert "no positional evidence" in candidate.evidence


def test_every_candidate_states_its_method_and_evidence(segment):
    hit = _hit(
        captions=[{"start": 20.0, "end": 24.0, "text": "Darth Baras and the Dark Council"}],
        chapters=[{"start_time": 18.0, "end_time": 30.0, "title": "Darth Baras"}],
        description="0:22 Darth Baras unmasked",
    )
    candidates = locate_timestamps(hit, segment)
    assert len(candidates) >= 3
    for candidate in candidates:
        assert candidate.method
        assert candidate.evidence.strip(), f"{candidate.method} gave no evidence"
        assert 0.0 <= candidate.confidence <= 1.0


def test_candidates_come_back_best_first(segment):
    hit = _hit(
        captions=[{"start": 20.0, "end": 24.0, "text": "Darth Baras and the Dark Council"}],
        chapters=[{"start_time": 18.0, "end_time": 30.0, "title": "Darth Baras"}],
        description="0:22 Darth Baras unmasked",
    )
    candidates = locate_timestamps(hit, segment)
    assert [c.confidence for c in candidates] == sorted(
        (c.confidence for c in candidates), reverse=True
    )
    assert candidates[0].method == "captions", "captions are the strongest evidence (§12.3)"


def test_a_timestamp_past_the_end_is_clamped(segment):
    hit = _hit(duration=100.0, description="9:99 Darth Baras unmasked\n5:00 Dark Council")
    for candidate in locate_timestamps(hit, segment):
        assert candidate.start_s <= 100.0
        assert candidate.end_s <= 100.0


def test_a_video_with_no_evidence_yields_nothing_positional(segment):
    assert from_captions(_hit(), segment) is None
    assert from_chapters(_hit(), segment) is None
    assert from_description(_hit(description="just a video"), segment) is None


# ---------------------------------------------------------------------------
# Classification (§12.2) -- every one with a stated reason
# ---------------------------------------------------------------------------


def test_every_classification_states_a_reason(segment):
    for title in (
        "Darth Baras Dark Council cutscene",
        "Let's Play SWTOR part 41 Darth Baras",
        "Darth Baras explained - video essay",
        "completely unrelated cooking video",
    ):
        classification, reason, _ = classify_hit(_hit(title=title), segment)
        assert classification in set(Classification)
        assert len(reason.strip()) > 20, f"{title!r} got a thin reason: {reason!r}"


def test_a_short_named_cutscene_is_exact_scene(segment):
    classification, reason, _ = classify_hit(
        _hit(title="Darth Baras confronts the Dark Council - full cutscene", duration=180.0),
        segment,
    )
    assert classification == Classification.EXACT_SCENE
    assert "cutscene" in reason.lower() or "scene" in reason.lower()


def test_a_long_compilation_is_only_likely_exact(segment):
    classification, _, _ = classify_hit(
        _hit(title="All SWTOR cutscenes - Darth Baras and the Dark Council", duration=7200.0),
        segment,
    )
    assert classification == Classification.LIKELY_EXACT_SCENE, (
        "a two-hour compilation is not the clip; the moment still has to be found"
    )


def test_a_playthrough_is_related_footage(segment):
    classification, reason, _ = classify_hit(
        _hit(title="SWTOR Let's Play part 41 - Darth Baras", duration=2400.0), segment
    )
    assert classification == Classification.RELATED_FOOTAGE
    assert "chapter" in reason.lower() or "playthrough" in reason.lower()


def test_commentary_is_contextual(segment):
    classification, reason, _ = classify_hit(
        _hit(title="Darth Baras explained - lore analysis", duration=900.0), segment
    )
    assert classification == Classification.CONTEXTUAL
    assert "commentary" in reason.lower()


def test_an_unrelated_video_is_contextual(segment):
    classification, reason, relevance = classify_hit(
        _hit(title="How to bake sourdough bread", description="flour and water"), segment
    )
    assert classification == Classification.CONTEXTUAL
    assert relevance == 0.0
    assert "no term" in reason.lower()


# ---------------------------------------------------------------------------
# Search orchestration
# ---------------------------------------------------------------------------


def test_search_returns_classified_records_with_timestamps(provider, segment, settings):
    records, notes = search_videos(segment, provider, settings, limit=5)
    assert notes == []
    assert len(records) == 5
    for record in records:
        assert record.video_id and record.title
        assert record.reason.strip()
        assert record.classification in set(Classification)


def test_the_clickable_link_is_stored_regardless_of_download(provider, segment, settings):
    """§12.7: store the &t= URL whether or not the clip downloaded."""
    records, _ = search_videos(segment, provider, settings, limit=3)
    with_timestamps = [r for r in records if r.best_timestamp()]
    assert with_timestamps, "the fake plants a locatable moment in every video"
    for record in with_timestamps:
        assert "t=" in record.url, f"no &t= in {record.url}"
        assert record.downloaded_clip_path == "", "nothing has been downloaded yet"


def test_results_are_ordered_best_first(provider, segment, settings):
    records, _ = search_videos(segment, provider, settings, limit=5)
    order = {
        Classification.EXACT_SCENE: 0,
        Classification.LIKELY_EXACT_SCENE: 1,
        Classification.RELATED_FOOTAGE: 2,
        Classification.CONTEXTUAL: 3,
    }
    ranks = [order[r.classification] for r in records]
    assert ranks == sorted(ranks)


def test_a_failing_video_provider_does_not_kill_the_segment(segment, settings):
    from visualresearcher.providers.base import Availability, ProviderError
    from visualresearcher.providers.video.base import VideoProvider

    class _Broken(VideoProvider):
        name = "broken"

        def availability(self):
            return Availability.available("broken")

        def search(self, query, *, limit=5, **kwargs):
            raise ProviderError("youtube is down")

        def fetch_section(self, request):
            raise NotImplementedError

    records, notes = search_videos(segment, _Broken(), settings)
    assert records == []
    assert notes and "youtube is down" in notes[0]


# ---------------------------------------------------------------------------
# §21: the download-sections range matches the located timestamp
# ---------------------------------------------------------------------------


def _record_at(start: float, end: float, confidence: float = 0.9, **kwargs) -> YouTubeRecord:
    base = {
        "video_id": "testvideo01",
        "url": "https://www.youtube.com/watch?v=testvideo01",
        "title": "Darth Baras cutscene",
        "duration": SOURCE_DURATION,
        "classification": Classification.EXACT_SCENE,
        "relevance": 0.9,
        "timestamp_candidates": [
            TimestampCandidate(
                start_s=start,
                end_s=end,
                method="captions",
                evidence="test",
                confidence=confidence,
            )
        ],
    }
    base.update(kwargs)
    return YouTubeRecord(**base)


def test_the_downloaded_clip_matches_the_located_span(provider, segment, settings, sandbox):
    """§21: the --download-sections range matches the located timestamp.

    Measured with ffprobe on the actual file, so the span arithmetic is what
    is being tested rather than a mock's return value.
    """
    settings.clips.padding_s = 3.0
    clips_dir = sandbox / "clips"
    record = _record_at(60.0, 68.0)

    outcomes = download_clip_for_segment(
        segment,
        [record],
        provider,
        settings,
        clips_dir=clips_dir,
        cache=ClipCache(sandbox / "clipcache"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes and outcomes[0].ok, outcomes[0].detail if outcomes else "no outcome"

    expected_start, expected_end = span_for(record, segment, settings)
    expected_duration = expected_end - expected_start

    actual = probe_duration(outcomes[0].path)
    assert abs(actual - expected_duration) < 1.0, (
        f"clip is {actual:.2f}s but the located span is {expected_duration:.2f}s "
        f"({expected_start:.2f}-{expected_end:.2f})"
    )


def test_padding_is_applied_on_both_sides(segment, settings):
    """§12.4: pad ±clip_padding_s."""
    settings.clips.padding_s = 3.0
    start, end = span_for(_record_at(60.0, 68.0), segment, settings)
    assert start == pytest.approx(57.0)
    assert end >= 71.0


def test_padding_cannot_produce_a_negative_start(segment, settings):
    settings.clips.padding_s = 5.0
    start, _ = span_for(_record_at(1.0, 4.0), segment, settings)
    assert start >= 0.0


def test_the_span_is_clamped_to_the_video(segment, settings):
    settings.clips.padding_s = 10.0
    start, end = span_for(_record_at(170.0, 178.0, duration=180.0), segment, settings)
    assert end <= 180.0
    assert start < end


def test_a_low_confidence_timestamp_yields_no_span(segment, settings):
    """§6: never fabricate a placeholder. A guess is not a pick."""
    assert span_for(_record_at(60.0, 68.0, confidence=0.2), segment, settings) is None
    assert MIN_CONFIDENCE > 0.3


def test_a_low_confidence_record_is_reported_not_silently_skipped(
    provider, segment, settings, sandbox
):
    outcomes = download_clip_for_segment(
        segment,
        [_record_at(60.0, 68.0, confidence=0.1)],
        provider,
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes and not outcomes[0].ok
    assert outcomes[0].reason == "low_confidence"
    assert "clickable link is kept" in outcomes[0].detail


# ---------------------------------------------------------------------------
# §21: the full-download guard refuses long sources
# ---------------------------------------------------------------------------


def test_the_full_download_guard_refuses_a_long_source(segment, settings, sandbox):
    from visualresearcher.providers.base import Availability
    from visualresearcher.providers.video.base import SectionResult, VideoProvider

    class _NoRanged(VideoProvider):
        name = "no-ranged"

        def availability(self):
            return Availability.available("cannot do ranged fetches")

        def search(self, query, *, limit=5, **kwargs):
            return []

        def fetch_section(self, request):
            return SectionResult(False, reason="unsupported", detail="no ranged support")

    settings.clips.max_full_download_minutes = 15
    # A 90-minute source: far over the limit.
    record = _record_at(60.0, 68.0, duration=5400.0)

    outcomes = download_clip_for_segment(
        segment,
        [record],
        _NoRanged(),
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes and not outcomes[0].ok
    assert "refused" in outcomes[0].detail
    assert "90.0 min" in outcomes[0].detail
    assert not list((sandbox / "clips").glob("*")) if (sandbox / "clips").exists() else True


def test_the_full_download_guard_permits_a_short_source(segment, settings, sandbox):
    from visualresearcher.providers.base import Availability
    from visualresearcher.providers.video.base import SectionResult, VideoProvider

    class _NoRanged(VideoProvider):
        name = "no-ranged"

        def availability(self):
            return Availability.available("cannot do ranged fetches")

        def search(self, query, *, limit=5, **kwargs):
            return []

        def fetch_section(self, request):
            return SectionResult(False, reason="unsupported", detail="no ranged support")

    settings.clips.max_full_download_minutes = 15
    record = _record_at(60.0, 68.0, duration=300.0)  # 5 minutes

    outcomes = download_clip_for_segment(
        segment,
        [record],
        _NoRanged(),
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes
    assert "permitted" in outcomes[0].detail
    assert "refused" not in outcomes[0].detail


# ---------------------------------------------------------------------------
# §21: no partial file in clips/
# ---------------------------------------------------------------------------


def test_a_failed_download_leaves_nothing_in_clips(segment, settings, sandbox):
    from visualresearcher.providers.base import Availability
    from visualresearcher.providers.video.base import SectionResult, VideoProvider

    class _Failing(VideoProvider):
        name = "failing"

        def availability(self):
            return Availability.available("always fails")

        def search(self, query, *, limit=5, **kwargs):
            return []

        def fetch_section(self, request):
            # Write a partial file where a careless implementation would,
            # then fail. clips/ must still be clean afterwards.
            staging = Path(request.tmp_dir)
            staging.mkdir(parents=True, exist_ok=True)
            (staging / f"{request.video_id}.mp4.part").write_bytes(b"half a video")
            return SectionResult(False, reason="download_failed", detail="interrupted")

    clips_dir = sandbox / "clips"
    download_clip_for_segment(
        segment,
        [_record_at(60.0, 68.0, duration=600.0)],
        _Failing(),
        settings,
        clips_dir=clips_dir,
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    leftovers = list(clips_dir.glob("*")) if clips_dir.exists() else []
    assert leftovers == [], f"clips/ holds {leftovers} after a failed download"


def test_a_successful_download_leaves_no_staging_file(provider, segment, settings, sandbox):
    clips_dir = sandbox / "clips"
    tmp = sandbox / "tmp"
    download_clip_for_segment(
        segment,
        [_record_at(40.0, 48.0)],
        provider,
        settings,
        clips_dir=clips_dir,
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=tmp,
        project_root=sandbox,
    )
    assert not list(clips_dir.glob("*.part"))
    assert not list(tmp.glob("*.part"))
    assert list(clips_dir.glob("*.mp4")), "the finished clip should be there"


# ---------------------------------------------------------------------------
# §21: the cache prevents a re-download
# ---------------------------------------------------------------------------


def test_the_cache_prevents_a_second_download(provider, segment, settings, sandbox):
    cache = ClipCache(sandbox / "clipcache")

    calls = {"n": 0}
    original = provider.fetch_section

    def counting(request):
        calls["n"] += 1
        return original(request)

    provider.fetch_section = counting  # type: ignore[method-assign]

    for run in range(2):
        outcomes = download_clip_for_segment(
            segment,
            [_record_at(50.0, 58.0)],
            provider,
            settings,
            clips_dir=sandbox / f"clips{run}",
            cache=cache,
            tmp_dir=sandbox / "tmp",
            project_root=sandbox,
        )
        assert outcomes[0].ok, outcomes[0].detail

    assert calls["n"] == 1, f"the clip was fetched {calls['n']} times; the cache did not hold"
    assert outcomes[0].cached is True


def test_the_cache_key_is_video_id_plus_span():
    """§12.6: cache by (video_id, start, end)."""
    a = ClipCache.key("vid", 10.0, 20.0)
    assert a == ClipCache.key("vid", 10.0, 20.0)
    assert a == ClipCache.key("vid", 10.04, 20.04), "sub-0.1s jitter is the same clip"
    assert a != ClipCache.key("vid", 11.0, 20.0)
    assert a != ClipCache.key("other", 10.0, 20.0)


def test_a_stale_cache_entry_is_forgotten(sandbox):
    cache = ClipCache(sandbox / "cc")
    fake_clip = sandbox / "clip.mp4"
    fake_clip.write_bytes(b"data")
    stored = cache.put("vid", 1.0, 2.0, fake_clip)
    assert cache.get("vid", 1.0, 2.0) is not None

    stored.unlink()
    assert cache.get("vid", 1.0, 2.0) is None, "a cache entry whose file is gone must not be used"


def test_a_corrupt_cache_index_is_survivable(sandbox):
    root = sandbox / "cc"
    root.mkdir(parents=True)
    (root / "index.json").write_text("{not json", encoding="utf-8")
    cache = ClipCache(root)
    assert cache.get("vid", 1.0, 2.0) is None


# ---------------------------------------------------------------------------
# Limits (§12.5)
# ---------------------------------------------------------------------------


def test_max_per_segment_is_honoured(provider, segment, settings, sandbox):
    settings.clips.max_per_segment = 1
    records = [_record_at(40.0 + i * 10, 48.0 + i * 10, video_id=f"vid{i}") for i in range(3)]
    outcomes = download_clip_for_segment(
        segment,
        records,
        provider,
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert sum(1 for o in outcomes if o.ok) == 1


def test_no_clips_disables_the_stage(provider, segment, settings, sandbox):
    settings.clips.enabled = False
    assert (
        download_clip_for_segment(
            segment,
            [_record_at(40.0, 48.0)],
            provider,
            settings,
            clips_dir=sandbox / "clips",
            cache=ClipCache(sandbox / "cc"),
            tmp_dir=sandbox / "tmp",
            project_root=sandbox,
        )
        == []
    )


def test_contextual_results_are_not_downloaded(provider, segment, settings, sandbox):
    record = _record_at(40.0, 48.0, classification=Classification.CONTEXTUAL)
    outcomes = download_clip_for_segment(
        segment,
        [record],
        provider,
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes == []
    assert Classification.CONTEXTUAL not in DOWNLOADABLE_CLASSES


def test_the_project_budget_stops_downloading(provider, segment, settings, sandbox):
    """§12.5: max project GB."""
    # Put real bytes in the project and set the budget below them, rather than
    # relying on a degenerate zero budget.
    (sandbox / "already_used.bin").write_bytes(b"0" * 2_000_000)
    settings.clips.max_project_gb = 1_500_000 / 1024**3

    outcomes = download_clip_for_segment(
        segment,
        [_record_at(40.0, 48.0)],
        provider,
        settings,
        clips_dir=sandbox / "clips",
        cache=ClipCache(sandbox / "cc"),
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert outcomes and not outcomes[0].ok
    assert outcomes[0].reason == "project_budget"


# ---------------------------------------------------------------------------
# The fake provider itself
# ---------------------------------------------------------------------------


def test_the_fake_produces_a_playable_trimmed_video(provider, sandbox):
    request = SectionRequest(
        video_id="synthetic01",
        url="https://www.youtube.com/watch?v=synthetic01",
        start_s=30.0,
        end_s=38.0,
        destination=sandbox / "out" / "clip",
        tmp_dir=sandbox / "tmp",
    )
    (sandbox / "out").mkdir(parents=True, exist_ok=True)
    result = provider.fetch_section(request)
    assert result.ok, result.detail
    assert result.path.exists()
    assert abs(probe_duration(result.path) - 8.0) < 0.6


def test_the_fake_refuses_a_span_past_the_end(provider, sandbox):
    (sandbox / "out").mkdir(parents=True, exist_ok=True)
    result = provider.fetch_section(
        SectionRequest(
            video_id="synthetic02",
            url="u",
            start_s=SOURCE_DURATION + 10,
            end_s=SOURCE_DURATION + 18,
            destination=sandbox / "out" / "clip",
            tmp_dir=sandbox / "tmp",
        )
    )
    assert not result.ok


def test_the_fake_plants_locatable_evidence(provider, segment):
    hits = provider.search("Darth Baras Dark Council SWTOR cutscene", limit=3)
    assert hits
    for hit in hits:
        assert hit.captions and hit.chapters
        assert hit.duration == SOURCE_DURATION


def test_the_fake_is_deterministic(sandbox):
    a = FakeVideoProvider(cache_dir=sandbox / "v").search("same query", limit=3)
    b = FakeVideoProvider(cache_dir=sandbox / "v").search("same query", limit=3)
    assert [h.video_id for h in a] == [h.video_id for h in b]


def test_a_cached_clip_lands_under_the_same_name_as_a_fresh_one(
    provider, segment, settings, sandbox
):
    """Two runs of the same project must produce the same files on disk.

    A cached clip used to be copied in under the cache's own filename, so a
    re-run produced a differently-named file for the same clip and broke the
    resume-identity guarantee.
    """
    cache = ClipCache(sandbox / "cc")

    first = download_clip_for_segment(
        segment,
        [_record_at(50.0, 58.0)],
        provider,
        settings,
        clips_dir=sandbox / "clips_a",
        cache=cache,
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    second = download_clip_for_segment(
        segment,
        [_record_at(50.0, 58.0)],
        provider,
        settings,
        clips_dir=sandbox / "clips_b",
        cache=cache,
        tmp_dir=sandbox / "tmp",
        project_root=sandbox,
    )
    assert first[0].ok and second[0].ok
    assert second[0].cached is True
    assert first[0].path.name == second[0].path.name, (
        f"fresh download named it {first[0].path.name} but the cached copy "
        f"named it {second[0].path.name}"
    )


def test_clip_confidence_is_not_the_timestamp_confidence():
    """§14's band answers a different question than §12.3's timestamp.

    A clip used to clear the download gate and then be excluded from
    selected/ by its band, so the bandwidth was spent and the user never saw
    the file.
    """
    from visualresearcher.pipeline.clips import clip_confidence

    exact = _record_at(60.0, 68.0, confidence=0.46)
    assert exact.classification == Classification.EXACT_SCENE
    band = clip_confidence(exact)
    assert band > 0.46, "the classification should carry most of the weight"
    assert band >= 0.60, "an EXACT_SCENE clip should reach selected/ (§14 MEDIUM)"


def test_clip_confidence_ranks_by_classification():
    from visualresearcher.pipeline.clips import clip_confidence

    def at(classification):
        return clip_confidence(_record_at(60.0, 68.0, classification=classification))

    assert (
        at(Classification.EXACT_SCENE)
        > at(Classification.LIKELY_EXACT_SCENE)
        > at(Classification.RELATED_FOOTAGE)
        > at(Classification.CONTEXTUAL)
    )


def test_an_uncertain_timestamp_discounts_but_does_not_dominate():
    from visualresearcher.pipeline.clips import clip_confidence

    sure = clip_confidence(_record_at(60.0, 68.0, confidence=1.0))
    unsure = clip_confidence(_record_at(60.0, 68.0, confidence=0.1))
    assert sure > unsure
    assert unsure > 0.4, "a well-classified clip should not be sunk by a fuzzy timestamp"
