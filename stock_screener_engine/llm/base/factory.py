"""Factory helpers for constructing LLM clients from settings."""

from __future__ import annotations

import os

from stock_screener_engine.config.settings import LlmSettings
from stock_screener_engine.llm.base.llm_client import HeuristicLLMClient, LLMClient
from stock_screener_engine.llm.base.provider_adapters import AnthropicLLMClient, OpenAICompatibleLLMClient


def build_llm_client(settings: LlmSettings) -> LLMClient:
    provider = settings.provider.strip().lower()
    if provider == "heuristic":
        return HeuristicLLMClient(model_name=settings.model)

    key = os.getenv(settings.api_key_env, "").strip()
    if not key:
        raise ValueError(f"LLM provider '{provider}' requires env var {settings.api_key_env}")

    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleLLMClient(
            api_key=key,
            model_name=settings.model,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
        )

    if provider == "anthropic":
        return AnthropicLLMClient(
            api_key=key,
            model_name=settings.model,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.provider}")
