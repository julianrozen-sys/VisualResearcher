"""Notification provider interface (CLAUDE.md §22 P8).

The point of the watcher is that a run needs **zero manual steps**. That only
works if something tells the user it finished, because otherwise they have to
poll a folder, which is a manual step wearing a disguise.

A notification failing must never fail a job. Every implementation returns
False rather than raising.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from ..base import Provider

__all__ = ["NotifyProvider"]


class NotifyProvider(Provider):
    kind = "notify"

    @abstractmethod
    def notify(
        self,
        title: str,
        message: str,
        *,
        path: Path | None = None,
        urgency: str = "normal",
    ) -> bool:
        """Show a notification. Returns True when it was delivered.

        Args:
            title: short headline.
            message: one or two lines of detail.
            path: a folder worth opening, if the platform can attach one.
            urgency: ``normal`` or ``critical``.

        Never raises. A desktop that cannot show a toast is not a job failure.
        """
