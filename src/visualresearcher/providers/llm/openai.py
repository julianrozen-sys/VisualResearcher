"""OpenAI LLM provider (CLAUDE.md §4: "LLM: OpenAI behind an interface").

Talks to the REST API over ``httpx`` rather than the vendor SDK. §2.7 requires
pipeline logic to go through a provider interface; keeping the HTTP call here
means one small, auditable surface and one dependency fewer.

Never constructed under ``VR_OFFLINE=1`` -- the registry substitutes the fake
before this class is reached.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ...logging_setup import get_logger
from ..base import Availability, ProviderError
from .base import LLMProvider, LLMResponse

__all__ = ["OpenAILLMProvider"]

log = get_logger("providers.llm.openai")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 3,
        **_ignored,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def _api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "").strip()

    def availability(self) -> Availability:
        if not self._api_key():
            return Availability.unavailable(
                "OPENAI_API_KEY is not set",
                "set OPENAI_API_KEY in your environment or .env.local",
            )
        return Availability.available(f"model={self.model}")

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        task: str,
        schema_hint: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        key = self._api_key()
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")

        if schema_hint:
            user = f"{user}\n\nReturn JSON matching this shape:\n{json.dumps(schema_hint)}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            # Deterministic output matters: §11 requires ranking to be
            # reproducible, and re-running a project must not churn results.
            "temperature": 0,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("openai %s: transport error on attempt %d: %s", task, attempt, exc)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = ProviderError(
                    f"HTTP {response.status_code} from OpenAI: {response.text[:200]}"
                )
                log.warning("openai %s: retryable HTTP %d", task, response.status_code)
                continue
            if response.status_code != 200:
                raise ProviderError(
                    f"HTTP {response.status_code} from OpenAI: {response.text[:400]}"
                )

            body = response.json()
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise ProviderError(f"unexpected OpenAI response shape: {body}") from exc
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI returned non-JSON content: {content[:400]}") from exc

            usage = body.get("usage", {})
            log.debug(
                "openai %s: %s prompt + %s completion tokens",
                task,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
            )
            return LLMResponse(parsed, model=body.get("model", self.model), usage=usage)

        raise ProviderError(
            f"OpenAI request failed after {self.max_retries} attempts: {last_error}"
        )
