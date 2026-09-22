"""robots.txt and rate limiting (CLAUDE.md §13).

§13 asks for two things when fetching from the open web: respect robots.txt,
and respect rate limits. Both are about being a guest on somebody else's
server, and both fail in the same direction if you get them wrong — you get
blocked, and then the tool stops working for reasons that look like bugs.

Two decisions worth stating:

* **A robots.txt we cannot read means allowed.** A 404, a timeout or a
  malformed file is not a prohibition; treating it as one would make the tool
  refuse most of the web over a transient error. A file we *can* read and that
  says no is obeyed.
* **The rate limiter is per host, not global.** Politeness is something the
  remote server experiences; slowing down requests to Wikimedia because we
  just fetched from imgur helps nobody.

``file://`` URLs skip both, so the offline fake is not throttled.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .. import __version__
from ..logging_setup import get_logger

__all__ = [
    "SHARED",
    "RobotsCache",
    "HostRateLimiter",
    "Politeness",
    "USER_AGENT",
    "DEFAULT_DELAY_S",
    "MAX_PENALTY",
    "ROBOTS_TTL_S",
]

log = get_logger("utils.politeness")

#: Wikimedia's User-Agent policy asks API clients to identify themselves with a
#: contact address, and throttles generic clients harder. The first live run ate
#: 14 refusals with 44-47s Retry-After values behind a bare
#: "VisualResearcher/0.1.0".
#:
#: The address is personal data, so it is never hardcoded here: it comes from
#: VR_CONTACT, which belongs in the gitignored .env.local. Unset is fine -- the
#: UA still says what we are, just without a way to reach us.
_CONTACT = os.environ.get("VR_CONTACT", "").strip()
USER_AGENT = (
    f"VisualResearcher/{__version__} (narration b-roll research tool; {_CONTACT})"
    if _CONTACT
    else f"VisualResearcher/{__version__} (narration b-roll research tool)"
)

#: How long a robots.txt answer is trusted before being fetched again.
ROBOTS_TTL_S = 3600.0

#: Minimum gap between requests to the same host, when robots.txt sets none.
DEFAULT_DELAY_S = 0.5

#: Ceiling on the adaptive penalty, so a host that 429s us forever settles at
#: a slow-but-moving rate instead of stalling the run completely.
MAX_PENALTY = 16.0


class RobotsCache:
    """Fetches and caches robots.txt per host."""

    def __init__(self, *, timeout: float = 10.0, user_agent: str = USER_AGENT) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self._parsers: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
        self._lock = threading.Lock()

    def _parser(self, scheme: str, host: str):
        now = time.monotonic()
        with self._lock:
            cached = self._parsers.get(host)
            if cached and now - cached[0] < ROBOTS_TTL_S:
                return cached[1]

        parser = self._fetch(scheme, host)

        with self._lock:
            self._parsers[host] = (now, parser)
        return parser

    def _fetch(self, scheme: str, host: str):
        """Fetch and parse robots.txt, or return None for "no opinion".

        We do the HTTP ourselves rather than call ``RobotFileParser.read()``,
        for one reason found on the first live run: ``read()`` uses urllib,
        which sends ``User-Agent: Python-urllib/3.x``. Wikimedia and Cloudflare
        both reject that outright, and ``read()`` turns a 401/403 into
        ``disallow_all = True`` -- so *every* URL on the host comes back
        refused, having never parsed a single rule.

        That is the exact inversion of this module's rule. Worse, it is
        self-inflicted: we were blocked for not identifying ourselves, which is
        what ``USER_AGENT`` exists to do. The symptom would have been
        "every image rejected as robots-disallowed" with a perfectly healthy
        robots.txt sitting there allowing us.

        So: send a real User-Agent, obey a robots.txt we actually receive, and
        treat anything else -- 403, 404, timeout, garbage -- as no opinion.
        """
        url = f"{scheme}://{host}/robots.txt"
        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - unreachable means "no opinion"
            log.debug("robots.txt unreachable for %s (%s); treating as allowed", host, exc)
            return None

        if response.status_code != 200:
            # Includes 403 "we do not like your client" and 404 "no robots.txt".
            # Neither is a prohibition we can read.
            log.debug(
                "robots.txt for %s returned HTTP %d; treating as allowed",
                host,
                response.status_code,
            )
            return None

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(url)
        try:
            parser.parse(response.text.splitlines())
        except Exception as exc:  # noqa: BLE001
            log.debug("robots.txt unparseable for %s (%s); treating as allowed", host, exc)
            return None
        return parser

    def allowed(self, url: str) -> bool:
        """True when ``url`` may be fetched. Unknown means allowed."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return True  # file:// and friends are ours to read

        parser = self._parser(parsed.scheme, parsed.netloc)
        if parser is None:
            return True
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:  # noqa: BLE001
            return True

    def crawl_delay(self, url: str) -> float:
        """The host's requested delay, or the default."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return 0.0
        parser = self._parser(parsed.scheme, parsed.netloc)
        if parser is None:
            return DEFAULT_DELAY_S
        try:
            delay = parser.crawl_delay(self.user_agent)
        except Exception:  # noqa: BLE001
            delay = None
        return float(delay) if delay else DEFAULT_DELAY_S


@dataclass
class HostRateLimiter:
    """One minimum interval per host, shared across threads."""

    default_delay: float = DEFAULT_DELAY_S
    _next_allowed: dict[str, float] = field(default_factory=dict, init=False)
    _penalty: dict[str, float] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def penalize(self, url: str) -> float:
        """Slow this host down for the rest of the run. Returns the new factor.

        Found on the first live run: retrying a 429 works, but retrying alone
        never stops us asking too fast. 32 segments cost 14 rate-limit retries
        and 10.6 minutes of pure waiting, because every retry went back to the
        same interval that had just been refused. A refusal is information --
        the host telling us our rate is wrong -- so it has to change the rate,
        not just delay the next identical attempt.
        """
        parsed = urlparse(url)
        if not parsed.netloc:
            return 1.0
        with self._lock:
            factor = min(self._penalty.get(parsed.netloc, 1.0) * 2.0, MAX_PENALTY)
            self._penalty[parsed.netloc] = factor
        return factor

    def wait(self, url: str, delay: float | None = None) -> float:
        """Block until this host may be called again. Returns the wait."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return 0.0

        host = parsed.netloc
        gap = self.default_delay if delay is None else delay
        gap *= self._penalty.get(host, 1.0)
        with self._lock:
            now = time.monotonic()
            earliest = self._next_allowed.get(host, 0.0)
            sleep_for = max(0.0, earliest - now)
            self._next_allowed[host] = max(now, earliest) + gap
        if sleep_for:
            time.sleep(sleep_for)
        return sleep_for


@dataclass
class Politeness:
    """robots.txt plus rate limiting, as one thing to pass around."""

    robots: RobotsCache = field(default_factory=RobotsCache)
    limiter: HostRateLimiter = field(default_factory=HostRateLimiter)
    #: Set False to skip robots.txt, e.g. for a host you own.
    check_robots: bool = True

    def allowed(self, url: str) -> bool:
        return self.robots.allowed(url) if self.check_robots else True

    def rate_limited(self, url: str) -> float:
        """Call after a 429. Widens this host's interval for the rest of the run."""
        return self.limiter.penalize(url)

    def before_request(self, url: str) -> float:
        """Rate-limit this host, using its own crawl-delay when it sets one."""
        delay = self.robots.crawl_delay(url) if self.check_robots else None
        return self.limiter.wait(url, delay)


#: The process-wide politeness gate.
#:
#: One instance, shared by provider *searches* and image *downloads* alike.
#: Having a separate limiter per call site defeats the point: a remote host
#: counts all our requests together, and Wikimedia returned 429 to a search
#: while the downloader was politely pacing itself against the same domain.
SHARED = Politeness()
