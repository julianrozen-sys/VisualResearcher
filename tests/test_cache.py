"""Provider cache tests (CLAUDE.md §20, §21).

§21 names one requirement here: **cache TTL correct**. §20 sets it at 30 days.

The second thing worth protecting is that a cache miss is never an error. A
cache that can fail a job is worse than no cache, so every failure path has to
degrade to "do the work again".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from visualresearcher.cache.store import DEFAULT_TTL_DAYS, CacheStore
from visualresearcher.db import session_scope
from visualresearcher.models import CacheEntry


@pytest.fixture
def cache(db_path) -> CacheStore:
    return CacheStore(db_path)


def _age(db_path, key: str, days: float) -> None:
    """Backdate an entry's expiry, as if it had been written ``days`` ago."""
    with session_scope(db_path) as session:
        entry = session.get(CacheEntry, key)
        entry.expires_at = datetime.now(UTC) - timedelta(days=days)
        session.add(entry)


# ---------------------------------------------------------------------------
# §21: the TTL is correct
# ---------------------------------------------------------------------------


def test_the_default_ttl_is_thirty_days():
    """§20 states it exactly."""
    assert DEFAULT_TTL_DAYS == 30


def test_a_fresh_entry_is_a_hit(cache):
    cache.set("ddgs", "search", {"q": "baras"}, ["a", "b"])
    assert cache.get("ddgs", "search", {"q": "baras"}) == ["a", "b"]
    assert cache.hits == 1


def test_an_expired_entry_is_a_miss(cache, db_path):
    key = cache.set("ddgs", "search", {"q": "baras"}, ["a"])
    _age(db_path, key, days=1)
    assert cache.get("ddgs", "search", {"q": "baras"}) is None


def test_an_entry_just_inside_the_ttl_still_hits(cache, db_path):
    key = cache.set("ddgs", "search", {"q": "x"}, ["a"])
    with session_scope(db_path) as session:
        entry = session.get(CacheEntry, key)
        entry.expires_at = datetime.now(UTC) + timedelta(minutes=1)
        session.add(entry)
    assert cache.get("ddgs", "search", {"q": "x"}) == ["a"]


def test_the_ttl_is_configurable(db_path):
    short = CacheStore(db_path, ttl_days=0)
    short.set("p", "e", {}, "value")
    # A zero-day TTL expires immediately.
    assert short.get("p", "e", {}) is None


def test_a_per_call_ttl_overrides_the_default(cache, db_path):
    key = cache.set("p", "e", {}, "value", ttl_days=90)
    with session_scope(db_path) as session:
        entry = session.get(CacheEntry, key)
        remaining = entry.expires_at.replace(tzinfo=UTC) - datetime.now(UTC)
    assert remaining > timedelta(days=80)


def test_an_expired_entry_is_removed_on_read(cache, db_path):
    """A stale row must not be re-checked on every future lookup."""
    key = cache.set("p", "e", {}, "value")
    _age(db_path, key, days=1)
    cache.get("p", "e", {})

    with session_scope(db_path) as session:
        assert session.get(CacheEntry, key) is None


def test_purge_removes_only_the_expired(cache, db_path):
    stale = cache.set("p", "old", {}, "value")
    cache.set("p", "new", {}, "value")
    _age(db_path, stale, days=1)

    assert cache.purge_expired() == 1
    assert cache.get("p", "new", {}) == "value"
    assert cache.get("p", "old", {}) is None


# ---------------------------------------------------------------------------
# Keys (§20)
# ---------------------------------------------------------------------------


def test_the_key_is_provider_endpoint_and_params(cache):
    base = CacheStore.key("ddgs", "search", {"q": "a"})
    assert base != CacheStore.key("wikimedia", "search", {"q": "a"})
    assert base != CacheStore.key("ddgs", "images", {"q": "a"})
    assert base != CacheStore.key("ddgs", "search", {"q": "b"})


def test_parameter_order_does_not_change_the_key(cache):
    cache.set("ddgs", "search", {"q": "a", "n": 30}, ["hit"])
    assert cache.get("ddgs", "search", {"n": 30, "q": "a"}) == ["hit"], (
        "dict ordering must not create a second cache entry"
    )


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        ["a", "b", "c"],
        {"nested": {"deep": [1, 2, 3]}},
        "a plain string",
        42,
        3.14,
        True,
        [],
        {},
    ],
)
def test_values_round_trip(cache, value):
    cache.set("p", "e", {"v": str(value)}, value)
    assert cache.get("p", "e", {"v": str(value)}) == value


def test_an_unserialisable_value_is_skipped_not_raised(cache):
    class NotJson:
        pass

    key = cache.set("p", "e", {}, {"obj": NotJson()})
    assert isinstance(key, str), "it must still return a key"
    # It simply is not cached; the caller does the work again.


def test_overwriting_replaces_the_value(cache):
    cache.set("p", "e", {}, "first")
    cache.set("p", "e", {}, "second")
    assert cache.get("p", "e", {}) == "second"


# ---------------------------------------------------------------------------
# A cache must never fail a job
# ---------------------------------------------------------------------------


def test_a_miss_on_an_unknown_key_is_none(cache):
    assert cache.get("nobody", "nothing", {}) is None
    assert cache.misses == 1


def test_an_unusable_database_is_a_miss_not_a_crash(tmp_path):
    broken = CacheStore(tmp_path / "nope" / "cannot" / "exist.db")
    # Reading must degrade to a miss rather than raising.
    assert broken.get("p", "e", {}) is None


def test_stats_report_what_is_there(cache, db_path):
    stale = cache.set("p", "a", {}, 1)
    cache.set("p", "b", {}, 2)
    _age(db_path, stale, days=1)

    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["expired"] == 1
    assert stats["live"] == 1


def test_clear_empties_the_cache(cache):
    cache.set("p", "a", {}, 1)
    cache.set("q", "b", {}, 2)
    assert cache.clear() == 2
    assert cache.get("p", "a", {}) is None


def test_clear_can_target_one_provider(cache):
    cache.set("p", "a", {}, 1)
    cache.set("q", "b", {}, 2)
    assert cache.clear(provider="p") == 1
    assert cache.get("p", "a", {}) is None
    assert cache.get("q", "b", {}) == 2


def test_hits_are_counted_on_the_row(cache, db_path):
    key = cache.set("p", "e", {}, "value")
    for _ in range(3):
        cache.get("p", "e", {})
    with session_scope(db_path) as session:
        assert session.get(CacheEntry, key).hits == 3
