"""LLM providers. Importing this package registers them."""

from ..registry import register
from .base import LLMProvider, LLMResponse
from .fake import FakeLLMProvider
from .openai import OpenAILLMProvider

register("llm", "fake", FakeLLMProvider, is_fake=True)
register("llm", "openai", OpenAILLMProvider)

__all__ = ["LLMProvider", "LLMResponse", "FakeLLMProvider", "OpenAILLMProvider"]
