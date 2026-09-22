"""Provider layer. Importing this package registers every known provider."""

from . import (
    embedding,  # noqa: F401
    images,  # noqa: F401
    llm,  # noqa: F401
    notify,  # noqa: F401
    transcription,  # noqa: F401
    video,  # noqa: F401
)
from .base import Availability, Provider, ProviderError, ProviderUnavailable
from .registry import get_provider, list_providers, providers_of_kind, register, resolve

__all__ = [
    "Availability",
    "Provider",
    "ProviderError",
    "ProviderUnavailable",
    "get_provider",
    "list_providers",
    "providers_of_kind",
    "register",
    "resolve",
]
