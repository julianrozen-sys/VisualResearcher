"""Desktop notifications.

Windows first (CLAUDE.md §2.12), via PowerShell's toast API, with a plain
console fallback everywhere else and whenever the toast fails.

No new dependency: the notification libraries on PyPI either pull in a GUI
toolkit or shell out to exactly this, and §2.10 asks for a logged reason
before adding anything. The reason here would have been "to avoid writing
fifteen lines of PowerShell", which is not a reason.

Delivery is best-effort by design. A missed toast must never fail a job, so
every failure path returns False and logs at debug level.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ...logging_setup import console, get_logger
from ..base import Availability
from .base import NotifyProvider

__all__ = ["DesktopNotifyProvider", "ConsoleNotifyProvider"]

log = get_logger("providers.notify.desktop")

TIMEOUT = 20

#: Windows toast via the WinRT XML API, which needs no install. The long
#: lines are PowerShell type accelerators, not Python -- see the per-file
#: line-length exemption in pyproject.toml.
_TOAST_SCRIPT = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType=WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$texts = $template.GetElementsByTagName('text')
$texts.Item(0).AppendChild($template.CreateTextNode($env:VR_TOAST_TITLE)) | Out-Null
$texts.Item(1).AppendChild($template.CreateTextNode($env:VR_TOAST_BODY)) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:VR_TOAST_APP).Show($toast)
"""


class ConsoleNotifyProvider(NotifyProvider):
    """Always available. Prints to the terminal and returns True."""

    name = "console"
    is_fake = True

    def __init__(self, **_ignored) -> None:
        self.sent: list[tuple[str, str]] = []

    def availability(self) -> Availability:
        return Availability.available("prints to the terminal", offline_safe=True)

    def notify(
        self, title: str, message: str, *, path: Path | None = None, urgency: str = "normal"
    ) -> bool:
        self.sent.append((title, message))
        colour = "red" if urgency == "critical" else "green"
        console.print(f"[{colour}]{title}[/{colour}] {message}")
        if path:
            console.print(f"  {path}")
        return True


class DesktopNotifyProvider(NotifyProvider):
    name = "desktop"

    def __init__(self, *, app_id: str = "VisualResearcher", **_ignored) -> None:
        self.app_id = app_id
        self._console = ConsoleNotifyProvider()

    def availability(self) -> Availability:
        if os.name != "nt":
            return Availability.available("falls back to the console on this platform")
        if not shutil.which("powershell") and not shutil.which("powershell.exe"):
            return Availability.unavailable(
                "powershell is not on PATH, so toasts are unavailable",
                "the console fallback still works",
            )
        return Availability.available("Windows toast via PowerShell")

    def notify(
        self, title: str, message: str, *, path: Path | None = None, urgency: str = "normal"
    ) -> bool:
        # The console line always happens: a terminal that is being watched is
        # more reliable than a toast that may be suppressed by focus assist.
        self._console.notify(title, message, path=path, urgency=urgency)

        if os.name != "nt":
            return True
        try:
            body = message if not path else f"{message}\n{path}"
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST_SCRIPT],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                env={
                    **os.environ,
                    "VR_TOAST_TITLE": title[:120],
                    "VR_TOAST_BODY": body[:300],
                    "VR_TOAST_APP": self.app_id,
                },
            )
            if result.returncode != 0:
                log.debug("toast failed: %s", (result.stderr or "").strip()[:200])
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - a missed toast is never fatal
            log.debug("toast unavailable: %s", exc)
            return False
