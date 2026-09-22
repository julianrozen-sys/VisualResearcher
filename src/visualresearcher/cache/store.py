"""The provider response cache (CLAUDE.md §20).

Keyed by ``sha256(provider + endpoint + normalized_params)``, with a 30-day
TTL. What it holds is listed in §20: searches, transcripts, entity
resolutions, LLM outputs, image hashes, CLIP embeddings, YouTube metadata and
clip spans.

Two properties matter more than speed:

* **an expired entry is not a hit.** A 30-day-old image search is stale; the
  page has probably changed and the licence may have. Expiry is checked on
  read rather than swept on a timer, so an entry cannot be served after its
  TTL even if nothing has cleaned up.
* **a cache miss is never an error.** Every failure path here — unreadable
  row, undecodable JSON, a database that will not open — returns "miss" and
  lets the caller do the work. A cache that can fail a job is worse than no
  cache.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlmodel import select

from ..db import session_scope
from ..logging_setup import get_logger
from ..models import CacheEntry, utcnow
from ..utils.hashing import cache_key

__all__ = ["CacheStore", "DEFAULT_TTL_DAYS"]

log = get_logger("cache.store")

#: §20: "TTL 30 days".
DEFAULT_TTL_DAYS = 30


class CacheStore:
    """Read-through cache over the ``cache_entries`` table."""

    def __init__(self, db_path: Path, *, ttl_days: int = DEFAULT_TTL_DAYS) -> None:
        self.db_path = Path(db_path)
        self.ttl = timedelta(days=ttl_days)
        self.hits = 0
        self.misses = 0

    # -- keys --------------------------------------------------------------

    @staticmethod
    def key(provider: str, endpoint: str, params: Any) -> str:
        return cache_key(provider, endpoint, params)

    # -- read --------------------------------------------------------------

    def get(self, provider: str, endpoint: str, params: Any) -> Any | None:
        """The cached value, or None on a miss or an expired entry."""
        key = self.key(provider, endpoint, params)
        try:
            with session_scope(self.db_path) as session:
                entry = session.get(CacheEntry, key)
                if entry is None:
                    self.misses += 1
                    return None

                if self._expired(entry):
                    # Drop it rather than leaving a stale row to be re-checked
                    # on every future lookup.
                    session.delete(entry)
                    self.misses += 1
                    log.debug("cache: %s/%s expired", provider, endpoint)
                    return None

                entry.hits += 1
                session.add(entry)
                value = entry.value
                self.hits += 1
                return value.get("v") if isinstance(value, dict) and "v" in value else value
        except Exception as exc:  # noqa: BLE001 - a broken cache is still a miss
            log.debug("cache read failed for %s/%s: %s", provider, endpoint, exc)
            self.misses += 1
            return None

    def _expired(self, entry: CacheEntry) -> bool:
        expires = entry.expires_at
        if expires is None:
            return False
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires <= datetime.now(UTC)

    # -- write -------------------------------------------------------------

    def set(
        self,
        provider: str,
        endpoint: str,
        params: Any,
        value: Any,
        *,
        ttl_days: int | None = None,
    ) -> str:
        """Store ``value``. Returns the key. Never raises."""
        key = self.key(provider, endpoint, params)
        ttl = timedelta(days=ttl_days) if ttl_days is not None else self.ttl
        # Wrapped so a non-dict value (a list of results, say) round-trips.
        payload = value if isinstance(value, dict) else {"v": value}
        try:
            json.dumps(payload, default=str)
        except (TypeError, ValueError) as exc:
            log.debug("cache: %s/%s is not serialisable (%s); not caching", provider, endpoint, exc)
            return key

        try:
            with session_scope(self.db_path) as session:
                entry = session.get(CacheEntry, key)
                now = utcnow()
                if entry is None:
                    entry = CacheEntry(key=key, provider=provider, endpoint=endpoint)
                entry.value = payload
                entry.created_at = now
                entry.expires_at = now + ttl
                session.add(entry)
        except Exception as exc:  # noqa: BLE001
            log.debug("cache write failed for %s/%s: %s", provider, endpoint, exc)
        return key

    # -- maintenance -------------------------------------------------------

    def purge_expired(self) -> int:
        """Delete every expired row. Returns how many went."""
        removed = 0
        with session_scope(self.db_path) as session:
            for entry in session.exec(select(CacheEntry)).all():
                if self._expired(entry):
                    session.delete(entry)
                    removed += 1
        if removed:
            log.info("cache: purged %d expired entr(y/ies)", removed)
        return removed

    def clear(self, provider: str | None = None) -> int:
        """Empty the cache, or just one provider's part of it."""
        with session_scope(self.db_path) as session:
            rows = session.exec(select(CacheEntry)).all()
            removed = 0
            for entry in rows:
                if provider is None or entry.provider == provider:
                    session.delete(entry)
                    removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        with session_scope(self.db_path) as session:
            rows = session.exec(select(CacheEntry)).all()
            expired = sum(1 for r in rows if self._expired(r))
            total = len(rows)
        return {
            "entries": total,
            "expired": expired,
            "live": total - expired,
            "hits": self.hits,
            "misses": self.misses,
        }
