"""Provider lookup.

``VR_OFFLINE=1`` is enforced here rather than in each provider, so there is one
place that decides whether real I/O is permitted and no pipeline stage can
accidentally bypass it.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import offline_mode
from ..logging_setup import get_logger
from .base import Provider, ProviderUnavailable

__all__ = ["register", "get_provider", "list_providers", "providers_of_kind", "resolve"]

log = get_logger("providers")

_REGISTRY: dict[tuple[str, str], Callable[..., Provider]] = {}
#: Fake to fall back to when offline, per kind.
_FAKES: dict[str, str] = {}


def register(
    kind: str, name: str, factory: Callable[..., Provider], *, is_fake: bool = False
) -> None:
    _REGISTRY[(kind, name)] = factory
    if is_fake:
        _FAKES.setdefault(kind, name)


def list_providers(kind: str | None = None) -> list[tuple[str, str]]:
    keys = sorted(_REGISTRY)
    return [k for k in keys if kind is None or k[0] == kind]


def providers_of_kind(kind: str, **kwargs) -> list[Provider]:
    return [_REGISTRY[(k, n)](**kwargs) for (k, n) in list_providers(kind)]


def get_provider(kind: str, name: str, **kwargs) -> Provider:
    factory = _REGISTRY.get((kind, name))
    if factory is None:
        known = ", ".join(n for (k, n) in list_providers(kind)) or "none registered"
        raise ProviderUnavailable(f"no {kind} provider named {name!r} (have: {known})")
    return factory(**kwargs)


def resolve(kind: str, name: str, **kwargs) -> Provider:
    """Pick a usable provider for ``kind``, honouring offline mode.

    Order: the offline fake when ``VR_OFFLINE=1``; otherwise ``name``; otherwise
    the registered fake with a warning. A job never dies because a provider is
    missing (§2.9).
    """
    if offline_mode():
        fake = _FAKES.get(kind)
        if fake:
            if name != fake:
                log.info("VR_OFFLINE=1: using fake %s provider %r instead of %r", kind, fake, name)
            return _REGISTRY[(kind, fake)](**kwargs)
        raise ProviderUnavailable(f"VR_OFFLINE=1 but no fake {kind} provider is registered")

    try:
        provider = get_provider(kind, name, **kwargs)
    except ProviderUnavailable as exc:
        log.warning("%s", exc)
    else:
        status = provider.availability()
        if status.ok:
            return provider
        log.warning("%s provider %r unavailable: %s", kind, name, status.detail)

    fake = _FAKES.get(kind)
    if not fake:
        raise ProviderUnavailable(f"no usable {kind} provider and no fake to fall back to")
    log.warning("falling back to fake %s provider %r", kind, fake)
    return _REGISTRY[(kind, fake)](**kwargs)
