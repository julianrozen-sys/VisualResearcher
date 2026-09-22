"""Content hashing. Dedupe (§11), cache keys (§20), and clip identity (§12)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["sha256_bytes", "sha256_file", "cache_key", "normalize_params"]

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_params(params: Any) -> str:
    """Stable JSON for cache keys: sorted keys, no incidental whitespace.

    Two calls that differ only in dict ordering must produce one cache entry.
    """
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def cache_key(provider: str, endpoint: str, params: Any) -> str:
    """``sha256(provider + endpoint + normalized_params)`` (CLAUDE.md §20)."""
    payload = f"{provider}\x00{endpoint}\x00{normalize_params(params)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
