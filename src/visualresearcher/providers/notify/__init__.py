"""Notification providers. Importing this package registers them."""

from ..registry import register
from .base import NotifyProvider
from .desktop import ConsoleNotifyProvider, DesktopNotifyProvider

register("notify", "console", ConsoleNotifyProvider, is_fake=True)
register("notify", "desktop", DesktopNotifyProvider)

__all__ = ["NotifyProvider", "ConsoleNotifyProvider", "DesktopNotifyProvider"]
