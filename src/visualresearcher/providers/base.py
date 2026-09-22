"""Provider contract (CLAUDE.md §2.7).

Providers do I/O and nothing else: no scoring, no file layout decisions, no
pipeline logic. A provider that cannot work says so through ``availability()``
instead of raising at import time, because ``doctor`` has to be able to report
on a provider that is not installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["Availability", "Provider", "ProviderError", "ProviderUnavailable"]


class ProviderError(RuntimeError):
    """A provider failed a call. Never fatal to a job (§8) — log and degrade."""


class ProviderUnavailable(ProviderError):
    """The provider cannot run at all here (missing dependency, missing key)."""


@dataclass(frozen=True)
class Availability:
    """What ``doctor`` prints for one provider."""

    ok: bool
    detail: str = ""
    #: Things the user could install or set to make this work.
    missing: tuple[str, ...] = field(default_factory=tuple)
    #: True when this provider needs no network and no credentials.
    offline_safe: bool = False

    @classmethod
    def available(cls, detail: str = "", *, offline_safe: bool = False) -> Availability:
        return cls(ok=True, detail=detail, offline_safe=offline_safe)

    @classmethod
    def unavailable(cls, detail: str, *missing: str) -> Availability:
        return cls(ok=False, detail=detail, missing=tuple(missing))


class Provider(ABC):
    """Base for every provider. ``name`` is what config and logs refer to."""

    #: Stable identifier used in config, cache keys, and records.
    name: str = "unnamed"
    #: Which interface this implements: transcription, images, video, llm, notify.
    kind: str = "unknown"
    #: True when the provider is a fixture-backed stand-in (§2.9).
    is_fake: bool = False

    @abstractmethod
    def availability(self) -> Availability:
        """Cheap, non-throwing readiness check. Must not hit the network."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} kind={self.kind!r}>"
