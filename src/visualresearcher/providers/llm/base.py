"""LLM provider interface (CLAUDE.md §2.7).

Providers are I/O only. They do not build prompts, do not interpret answers,
and do not know what a segment is -- the pipeline owns all of that and hands
down a finished prompt.

``task`` is the one concession to that rule: a short machine-readable label
naming what is being asked. Real providers use it only for logging and cache
keys; the fake uses it to pick a fixture without having to parse English.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..base import Provider

__all__ = ["LLMProvider", "LLMResponse"]


class LLMResponse(dict):
    """A JSON object from the model, with the usage metadata kept alongside.

    Subclasses ``dict`` so callers can treat it as the parsed payload while
    ``usage`` and ``model`` stay available for the cost report (§22 P8).
    """

    def __init__(self, payload: dict, *, model: str = "", usage: dict | None = None):
        super().__init__(payload)
        self.model = model
        self.usage = usage or {}


class LLMProvider(Provider):
    kind = "llm"

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        task: str,
        schema_hint: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Return a JSON object for the given prompt.

        Args:
            system: the system prompt.
            user: the user prompt, already assembled by the pipeline.
            task: machine-readable label, e.g. ``"project_context"``.
            schema_hint: optional shape description passed to the model.
            max_tokens: output budget.

        Raises:
            ProviderError: on transport failure or unparseable output. The
                caller degrades the segment rather than failing the job (§8).
        """
