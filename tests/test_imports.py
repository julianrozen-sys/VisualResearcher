"""Every module must import cleanly.

This exists because a bad escape sequence in a docstring inside a
lazily-imported module went unnoticed until a test happened to touch it. A
module that cannot be imported is a broken module whether or not anything
imports it today.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

import pytest

import visualresearcher

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(visualresearcher.__path__, prefix="visualresearcher.")
)


def test_module_discovery_found_the_package():
    assert len(MODULES) > 15, f"expected the whole package, found {MODULES}"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_without_error_or_warning(module_name):
    with warnings.catch_warnings():
        # A SyntaxWarning (e.g. an invalid escape sequence) must fail the test,
        # not scroll past in the output.
        warnings.simplefilter("error", SyntaxWarning)
        importlib.import_module(module_name)


def test_cli_exposes_every_command_named_in_section_17():
    """§17 lists the CLI surface; all of it must at least exist."""
    from visualresearcher.cli import app

    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    names |= {g.name for g in app.registered_groups}
    expected = {
        "run",
        "serve",
        "worker",
        "watch",
        "status",
        "resume",
        "rerun",
        "collect",
        "open",
        "doctor",
        "cache",
    }
    missing = expected - names
    assert not missing, f"CLI is missing commands from §17: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------

#: Every keyword the pipeline passes to `resolve()` anywhere, pooled. A
#: provider is constructed by name, so each one must tolerate the whole set --
#: including a fake standing in for a real provider with different options.
_PIPELINE_KWARGS = {
    "model": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "device": "cpu",
    "compute_type": "int8",
    "seed": 1729,
    "min_width": 800,
    "timeout": 20.0,
}


def test_every_registered_provider_accepts_the_pipeline_kwargs():
    """A fake that cannot be constructed like its real counterpart is not a fake.

    This exists because `FakeEmbeddingProvider` took no arguments while the
    registry passed it four, so `VR_OFFLINE=1` failed at the ranking stage --
    the one path that is supposed to always work.
    """
    import visualresearcher.providers  # noqa: F401  (registers everything)
    from visualresearcher.providers.registry import get_provider, list_providers

    failures = []
    for kind, name in list_providers():
        try:
            get_provider(kind, name, **_PIPELINE_KWARGS)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{kind}/{name}: {type(exc).__name__}: {exc}")
    assert not failures, "providers that cannot be constructed:\n  " + "\n  ".join(failures)


def test_every_registered_provider_reports_availability_without_raising():
    """`doctor` calls this on everything, including things that are not installed."""
    import visualresearcher.providers  # noqa: F401
    from visualresearcher.providers.base import Availability
    from visualresearcher.providers.registry import get_provider, list_providers

    for kind, name in list_providers():
        provider = get_provider(kind, name, **_PIPELINE_KWARGS)
        availability = provider.availability()
        assert isinstance(availability, Availability), f"{kind}/{name}"
        if not availability.ok:
            assert availability.detail, f"{kind}/{name} is unavailable but says nothing"


def test_every_provider_kind_has_a_fake():
    """§2.9: VR_OFFLINE=1 must run the whole pipeline with zero credentials."""
    import visualresearcher.providers  # noqa: F401
    from visualresearcher.providers.registry import _FAKES, list_providers

    kinds = {kind for kind, _ in list_providers()}
    missing = kinds - set(_FAKES)
    assert not missing, f"these provider kinds have no offline fake: {sorted(missing)}"
