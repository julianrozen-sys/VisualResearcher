"""Notification tests (CLAUDE.md §22 P8).

The rule these exist to protect: **a notification must never fail a job.** The
whole point of the watcher is zero manual steps, and a run that crashes on the
last line because a toast could not be shown has failed at exactly the moment
it had succeeded.
"""

from __future__ import annotations

import pytest

from visualresearcher.providers.notify.desktop import (
    ConsoleNotifyProvider,
    DesktopNotifyProvider,
)


def test_the_console_provider_always_works():
    provider = ConsoleNotifyProvider()
    assert provider.availability().ok is True
    assert provider.notify("title", "message") is True
    assert provider.sent == [("title", "message")]


def test_the_console_provider_is_offline_safe():
    assert ConsoleNotifyProvider().availability().offline_safe is True


def test_the_desktop_provider_reports_its_readiness():
    availability = DesktopNotifyProvider().availability()
    assert isinstance(availability.ok, bool)
    if not availability.ok:
        assert availability.missing or availability.detail


def test_a_failing_toast_returns_false_rather_than_raising(monkeypatch, tmp_path):
    """The one behaviour that matters: it must not raise."""
    import subprocess

    def explode(*args, **kwargs):
        raise OSError("no shell here")

    monkeypatch.setattr(subprocess, "run", explode)
    provider = DesktopNotifyProvider()
    result = provider.notify("title", "message", path=tmp_path)
    assert result in (True, False), "notify must return a bool, not raise"


def test_a_nonzero_toast_exit_is_not_an_error(monkeypatch):
    import subprocess

    class _Result:
        returncode = 1
        stderr = "toast refused"
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
    import os

    if os.name != "nt":
        pytest.skip("the toast path only runs on Windows")
    assert DesktopNotifyProvider().notify("t", "m") is False


def test_the_console_line_happens_even_when_the_toast_fails(monkeypatch):
    """A watched terminal beats a toast that focus assist may have eaten."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    provider = DesktopNotifyProvider()
    provider.notify("headline", "body")
    assert provider._console.sent == [("headline", "body")]


def test_long_text_is_truncated_not_rejected(monkeypatch):
    captured = {}

    import subprocess

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def capture(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        return _Result()

    monkeypatch.setattr(subprocess, "run", capture)
    import os

    if os.name != "nt":
        pytest.skip("the toast path only runs on Windows")

    DesktopNotifyProvider().notify("T" * 500, "B" * 2000)
    assert len(captured["VR_TOAST_TITLE"]) <= 120
    assert len(captured["VR_TOAST_BODY"]) <= 300


def test_both_providers_are_registered():
    import visualresearcher.providers  # noqa: F401
    from visualresearcher.providers.registry import list_providers

    names = {name for kind, name in list_providers("notify")}
    assert names == {"console", "desktop"}


def test_offline_resolves_to_the_console_provider(monkeypatch):
    """§2.9: VR_OFFLINE must not try to raise a desktop toast."""
    monkeypatch.setenv("VR_OFFLINE", "1")
    from visualresearcher.providers.registry import resolve

    assert resolve("notify", "desktop").name == "console"
