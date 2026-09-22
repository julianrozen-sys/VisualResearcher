"""robots.txt and rate limiting (CLAUDE.md §13).

§13: "Respect robots.txt and rate limits; prefer APIs over HTML scraping."

The rule that needs a test more than the others is the failure direction: a
robots.txt we *cannot read* must mean allowed. Treating an unreachable or
malformed file as a prohibition would make the tool refuse most of the web
over a transient error, and the symptom would look like "image search stopped
working" rather than "robots.txt timed out".
"""

from __future__ import annotations

import threading
import time

from visualresearcher.utils.politeness import (
    DEFAULT_DELAY_S,
    HostRateLimiter,
    Politeness,
    RobotsCache,
)


class _StubRobots(RobotsCache):
    """A RobotsCache with the network replaced by a canned rules table."""

    def __init__(self, rules: dict[str, str] | None = None, *, fail: bool = False):
        super().__init__()
        self.rules = rules or {}
        self.fail = fail
        self.fetches = 0

    def _parser(self, scheme: str, host: str):
        import urllib.robotparser

        self.fetches += 1
        if self.fail or host not in self.rules:
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(self.rules[host].splitlines())
        return parser


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def test_a_disallowed_path_is_refused():
    robots = _StubRobots({"example.invalid": "User-agent: *\nDisallow: /private/"})
    assert robots.allowed("https://example.invalid/private/secret.jpg") is False
    assert robots.allowed("https://example.invalid/public/ok.jpg") is True


def test_an_unreadable_robots_means_allowed():
    """A timeout is not a prohibition."""
    robots = _StubRobots(fail=True)
    assert robots.allowed("https://example.invalid/anything.jpg") is True


def test_a_host_with_no_rules_is_allowed():
    assert _StubRobots().allowed("https://unknown.invalid/x.jpg") is True


def test_a_blanket_disallow_is_obeyed():
    robots = _StubRobots({"closed.invalid": "User-agent: *\nDisallow: /"})
    assert robots.allowed("https://closed.invalid/anything.jpg") is False


def test_file_urls_skip_robots_entirely():
    """The offline fake must not be consulted or throttled."""
    robots = _StubRobots({"closed.invalid": "User-agent: *\nDisallow: /"})
    assert robots.allowed("file:///D:/some/local/image.jpg") is True
    assert robots.fetches == 0, "a file:// URL should not trigger a robots fetch"


def test_a_crawl_delay_is_read():
    robots = _StubRobots({"slow.invalid": "User-agent: *\nCrawl-delay: 5\nDisallow:"})
    assert robots.crawl_delay("https://slow.invalid/x.jpg") == 5.0


def test_a_host_without_a_crawl_delay_gets_the_default():
    robots = _StubRobots({"plain.invalid": "User-agent: *\nDisallow: /private/"})
    assert robots.crawl_delay("https://plain.invalid/x.jpg") == DEFAULT_DELAY_S


def test_robots_answers_are_cached():
    robots = RobotsCache()
    robots._parsers["cached.invalid"] = (time.monotonic(), None)
    before = len(robots._parsers)
    robots.allowed("https://cached.invalid/a.jpg")
    robots.allowed("https://cached.invalid/b.jpg")
    assert len(robots._parsers) == before, "a cached host should not be re-fetched"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_requests_to_one_host_are_spaced_out():
    limiter = HostRateLimiter(default_delay=0.05)
    started = time.monotonic()
    for _ in range(3):
        limiter.wait("https://one.invalid/x")
    elapsed = time.monotonic() - started
    assert elapsed >= 0.08, f"three requests took {elapsed:.3f}s; the limiter is not holding"


def test_different_hosts_do_not_slow_each_other_down():
    """§13 politeness is something the remote server experiences."""
    limiter = HostRateLimiter(default_delay=0.2)
    limiter.wait("https://a.invalid/x")
    started = time.monotonic()
    limiter.wait("https://b.invalid/x")
    assert time.monotonic() - started < 0.1, (
        "a request to one host was delayed by a request to a different host"
    )


def test_file_urls_are_not_throttled():
    limiter = HostRateLimiter(default_delay=1.0)
    started = time.monotonic()
    for _ in range(5):
        limiter.wait("file:///D:/x.jpg")
    assert time.monotonic() - started < 0.2


def test_the_limiter_is_thread_safe():
    limiter = HostRateLimiter(default_delay=0.01)
    errors = []

    def hammer():
        try:
            for _ in range(20):
                limiter.wait("https://busy.invalid/x")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors


# ---------------------------------------------------------------------------
# Together, and wired into downloads
# ---------------------------------------------------------------------------


def test_politeness_combines_both():
    politeness = Politeness(
        robots=_StubRobots({"x.invalid": "User-agent: *\nDisallow: /no/"}),
        limiter=HostRateLimiter(default_delay=0.01),
    )
    assert politeness.allowed("https://x.invalid/no/a.jpg") is False
    assert politeness.allowed("https://x.invalid/yes/a.jpg") is True
    politeness.before_request("https://x.invalid/yes/a.jpg")


def test_robots_checking_can_be_turned_off():
    politeness = Politeness(
        robots=_StubRobots({"x.invalid": "User-agent: *\nDisallow: /"}),
        check_robots=False,
    )
    assert politeness.allowed("https://x.invalid/anything.jpg") is True


def test_the_downloader_refuses_a_disallowed_url(settings, sandbox, monkeypatch):
    """§13, wired all the way through."""
    from visualresearcher.pipeline import download as download_mod
    from visualresearcher.providers.images.base import ImageCandidate

    monkeypatch.setattr(
        download_mod,
        "POLITENESS",
        Politeness(robots=_StubRobots({"blocked.invalid": "User-agent: *\nDisallow: /"})),
    )
    images = sandbox / "images"
    images.mkdir(parents=True, exist_ok=True)

    outcome = download_mod.download_candidate(
        ImageCandidate(image_url="https://blocked.invalid/a.jpg", provider="test"),
        images,
        settings,
        index=0,
        segment_index=1,
    )
    assert outcome.ok is False
    assert outcome.reason == download_mod.RejectReason.DISALLOWED
    assert "robots.txt" in outcome.detail
    assert list(images.iterdir()) == [], "nothing may be written for a refused URL"


def test_the_offline_fake_is_never_throttled(settings, sandbox):
    """A file:// download must not consult robots.txt or sleep."""
    from visualresearcher.pipeline.download import download_candidate
    from visualresearcher.providers.images.base import ImageCandidate
    from visualresearcher.providers.images.fake import generate_image

    source = sandbox / "local.jpg"
    generate_image(source, 4321)
    images = sandbox / "images"
    images.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    outcome = download_candidate(
        ImageCandidate(image_url=source.resolve().as_uri(), provider="fake"),
        images,
        settings,
        index=0,
        segment_index=1,
    )
    assert outcome.ok, outcome.detail
    assert time.monotonic() - started < 2.0, "the offline path was throttled"


# ---------------------------------------------------------------------------
# Provider searches, not just downloads (found by a live 429)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """The three bits of an httpx response the provider actually reads."""

    def __init__(self, status: int, payload: dict | None = None, retry_after: str | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = {"retry-after": retry_after} if retry_after else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None  # type: ignore[arg-type]
            )

    def json(self):
        return self._payload


_ONE_HIT = {
    "query": {
        "pages": {
            "1": {
                "title": "File:Sith.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/a/Sith.jpg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Sith.jpg",
                        "width": 1600,
                        "height": 900,
                        "mime": "image/jpeg",
                        "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 2.0"}},
                    }
                ],
            }
        }
    }
}


def _provider_and_calls(monkeypatch, responses):
    """A Wikimedia provider whose HTTP layer replays ``responses``."""
    from visualresearcher.providers.images import wikimedia as mod

    calls: list[float] = []
    queue = list(responses)

    def fake_get(url, **kwargs):
        calls.append(time.monotonic())
        return queue.pop(0)

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda s: calls.append(-s))
    return mod.WikimediaImageProvider(), calls


def test_a_429_is_retried_rather_than_losing_the_segment(monkeypatch):
    """The live failure: one 429 must not cost a segment its provider."""
    provider, calls = _provider_and_calls(
        monkeypatch, [_FakeResponse(429), _FakeResponse(200, _ONE_HIT)]
    )
    results = provider.search("sith lord", limit=5)
    assert len(results) == 1, "a retryable 429 was treated as a dead end"
    assert results[0].license == "CC BY-SA 2.0"


def test_a_retry_after_header_is_obeyed(monkeypatch):
    """The server's own number beats our backoff."""
    provider, calls = _provider_and_calls(
        monkeypatch,
        [_FakeResponse(429, retry_after="7"), _FakeResponse(200, _ONE_HIT)],
    )
    provider.search("sith lord", limit=5)
    slept = [-c for c in calls if c < 0]
    assert 7.0 in slept, f"Retry-After: 7 was not waited out; slept {slept}"


def test_a_persistent_429_gives_up_instead_of_looping(monkeypatch):
    from visualresearcher.providers.base import ProviderError
    from visualresearcher.providers.images import wikimedia as mod

    attempts = mod.RETRY_ON_429 + 1
    provider, _ = _provider_and_calls(monkeypatch, [_FakeResponse(429)] * attempts)
    try:
        provider.search("sith lord", limit=5)
    except ProviderError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("a permanently rate-limited host should raise, not return")


def test_a_search_passes_through_the_rate_limiter(monkeypatch):
    """§13 applies to searches, not only to downloads."""
    from visualresearcher.providers.images import wikimedia as mod

    seen: list[str] = []
    monkeypatch.setattr(
        mod.POLITENESS, "before_request", lambda url: seen.append(url) or 0.0
    )
    provider, _ = _provider_and_calls(monkeypatch, [_FakeResponse(200, _ONE_HIT)])
    provider.search("sith lord", limit=5)
    assert seen == [mod.API], "the search did not go through the politeness gate"


def test_searches_and_downloads_share_one_gate():
    """Two gates would let each politely double the real request rate."""
    from visualresearcher.pipeline import download as download_mod
    from visualresearcher.providers.images import wikimedia as wikimedia_mod
    from visualresearcher.utils.politeness import SHARED

    assert wikimedia_mod.POLITENESS is SHARED
    assert download_mod.POLITENESS is SHARED


def test_a_garbled_retry_after_falls_back_to_our_own_backoff(monkeypatch):
    """A header we cannot parse is still a 429; it must not crash the stage."""
    provider, calls = _provider_and_calls(
        monkeypatch,
        [_FakeResponse(429, retry_after="in a bit"), _FakeResponse(200, _ONE_HIT)],
    )
    assert len(provider.search("sith lord", limit=5)) == 1
    assert any(c < 0 for c in calls), "an unparseable Retry-After skipped the backoff"



# ---------------------------------------------------------------------------
# Fetching robots.txt (found by the first live run)
#
# `RobotFileParser.read()` fetches with urllib, which sends
# `User-Agent: Python-urllib/3.x`. Wikimedia and Cloudflare reject that, and
# read() converts a 401/403 into `disallow_all = True` -- refusing every URL
# on the host without having parsed one rule. Live, that meant
# upload.wikimedia.org came back disallowed while its robots.txt was happily
# allowing us, and the whole download stage would have rejected everything.
# ---------------------------------------------------------------------------


class _RobotsResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


def _robots_serving(monkeypatch, response, *, record=None):
    """A RobotsCache whose robots.txt fetch returns ``response``."""
    from visualresearcher.utils import politeness as mod

    def fake_get(url, **kwargs):
        if record is not None:
            record.append((url, kwargs.get("headers", {})))
        return response

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    return mod.RobotsCache()


def test_a_403_on_robots_means_allowed_not_forbidden(monkeypatch):
    """The live bug. A host refusing our robots request is not a prohibition."""
    robots = _robots_serving(monkeypatch, _RobotsResponse(403))
    assert robots.allowed("https://upload.wikimedia.org/a/Sith.jpg") is True


def test_a_404_on_robots_means_allowed(monkeypatch):
    robots = _robots_serving(monkeypatch, _RobotsResponse(404))
    assert robots.allowed("https://nowhere.invalid/a.jpg") is True


def test_a_500_on_robots_means_allowed(monkeypatch):
    robots = _robots_serving(monkeypatch, _RobotsResponse(500))
    assert robots.allowed("https://broken.invalid/a.jpg") is True


def test_a_robots_we_can_read_is_still_obeyed(monkeypatch):
    """The safety property must survive the fix."""
    robots = _robots_serving(
        monkeypatch, _RobotsResponse(200, "User-agent: *\nDisallow: /private/\n")
    )
    assert robots.allowed("https://real.invalid/private/x.jpg") is False
    assert robots.allowed("https://real.invalid/public/x.jpg") is True


def test_the_robots_fetch_identifies_us(monkeypatch):
    """The root cause: we were blocked for not saying who we are."""
    from visualresearcher.utils.politeness import USER_AGENT

    seen: list = []
    robots = _robots_serving(monkeypatch, _RobotsResponse(200, "User-agent: *\nDisallow:\n"),
                             record=seen)
    robots.allowed("https://commons.wikimedia.org/x.jpg")
    assert seen, "no robots.txt request was made at all"
    url, headers = seen[0]
    assert url == "https://commons.wikimedia.org/robots.txt"
    assert headers.get("User-Agent") == USER_AGENT, (
        f"robots.txt was fetched as {headers.get('User-Agent')!r}; "
        "an unidentified fetch is what got us blocked live"
    )


def test_a_network_error_fetching_robots_means_allowed(monkeypatch):
    import httpx

    from visualresearcher.utils import politeness as mod

    def boom(url, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(mod.httpx, "get", boom)
    assert mod.RobotsCache().allowed("https://unreachable.invalid/a.jpg") is True


# ---------------------------------------------------------------------------
# A 429 must change the rate, not just delay the next identical attempt
#
# The first live run retried correctly and still spent 10.6 minutes of its
# first 32 segments sitting in `time.sleep`, across 14 separate refusals,
# because every retry went back to the same interval that had just been
# refused. Retrying handles the symptom; only backing off handles the cause.
# ---------------------------------------------------------------------------


def test_a_429_widens_the_gate_for_the_rest_of_the_run(monkeypatch):
    from visualresearcher.providers.images import wikimedia as mod

    widened: list[str] = []
    monkeypatch.setattr(mod.POLITENESS, "rate_limited", lambda url: widened.append(url) or 2.0)
    provider, _ = _provider_and_calls(
        monkeypatch, [_FakeResponse(429), _FakeResponse(200, _ONE_HIT)]
    )
    provider.search("sith lord", limit=5)
    assert widened == [mod.API], "a 429 did not slow the host down; we retry into the same wall"


def test_the_penalty_compounds_and_is_capped():
    from visualresearcher.utils.politeness import MAX_PENALTY, HostRateLimiter

    limiter = HostRateLimiter()
    url = "https://commons.wikimedia.org/w/api.php"
    assert limiter.penalize(url) == 2.0
    assert limiter.penalize(url) == 4.0
    for _ in range(10):
        limiter.penalize(url)
    assert limiter.penalize(url) == MAX_PENALTY, "an unbounded penalty would stall the run"


def test_the_penalty_is_per_host():
    """One rude host must not slow every other provider down."""
    from visualresearcher.utils.politeness import HostRateLimiter

    limiter = HostRateLimiter()
    limiter.penalize("https://commons.wikimedia.org/w/api.php")
    limiter.penalize("https://commons.wikimedia.org/w/api.php")
    slow = limiter.wait("https://commons.wikimedia.org/w/api.php")
    fast = limiter.wait("https://duckduckgo.com/")
    assert slow == 0.0 and fast == 0.0  # first call to each host never waits
    assert limiter._penalty.get("duckduckgo.com") is None


def test_the_rate_limit_log_line_names_the_status_code(monkeypatch, caplog):
    """The old line said only "rate-limited", so a filter watching for 429 saw
    nothing -- which is how a run was reported as having zero rate limits
    while it sat through fourteen of them."""
    import logging

    provider, _ = _provider_and_calls(
        monkeypatch, [_FakeResponse(429), _FakeResponse(200, _ONE_HIT)]
    )
    with caplog.at_level(logging.WARNING):
        provider.search("sith lord", limit=5)
    assert "429" in caplog.text, f"the retry log does not mention 429: {caplog.text!r}"
