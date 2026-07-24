# src_ecg_1d/generation/llm_client.py
#
# Layer 4: LLM API wrapper.
#
# Abstracts the LLM behind a simple interface so the provider (Anthropic
# Claude / OpenAI GPT) can be swapped via configs.config_1d.LLM_PROVIDER
# without touching PromptBuilder or ExplanationGenerator.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from configs.config_1d import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE


class LLMClient(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        """Send prompt, return response text."""


class AnthropicClient(LLMClient):
    """Anthropic Claude API wrapper.

    Uses claude-sonnet-4-6 by default — this model accepts an explicit
    `temperature`, which the 4.7/4.8-tier and Sonnet 5 models reject
    outright. Low temperature keeps explanations reproducible for
    evaluation (see Layer 5).
    """

    def __init__(self, model: str = LLM_MODEL, api_key: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return next(block.text for block in response.content if block.type == "text")


class OpenAIClient(LLMClient):
    """OpenAI GPT wrapper — same interface, drop-in replacement."""

    def __init__(self, model: str = "gpt-4-turbo", api_key: Optional[str] = None):
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def build_llm_client(provider: str = "anthropic", **kwargs) -> LLMClient:
    """Factory: build the configured LLM client by provider name."""
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
