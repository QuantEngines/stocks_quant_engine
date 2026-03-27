"""Startup validation helpers for strict deployment readiness checks."""

from __future__ import annotations

import os

from stock_screener_engine.config.settings import AppSettings

_OPENAI_PROVIDER_ALIASES = {"openai", "openai-compatible", "openai_compatible"}


def validate_startup_settings(settings: AppSettings) -> None:
    """Fail fast for invalid startup configuration.

    Current strict checks:
    - If LLM extraction is enabled, provider must be supported.
    - Real providers must define and resolve a non-empty API key env var.
    """

    if not settings.llm.enabled:
        return

    provider = settings.llm.provider.strip().lower()
    if provider == "heuristic":
        return

    if provider not in _OPENAI_PROVIDER_ALIASES and provider != "anthropic":
        raise ValueError(
            "SSE_ENABLE_LLM_EXTRACTION=true but SSE_LLM_PROVIDER is unsupported: "
            f"{settings.llm.provider}"
        )

    key_env = settings.llm.api_key_env.strip()
    if not key_env:
        raise ValueError(
            "SSE_ENABLE_LLM_EXTRACTION=true but SSE_LLM_API_KEY_ENV is empty for provider "
            f"{settings.llm.provider}"
        )

    key_value = os.getenv(key_env, "").strip()
    if not key_value:
        raise ValueError(
            "SSE_ENABLE_LLM_EXTRACTION=true but required provider key is missing: "
            f"set env var {key_env} for provider {settings.llm.provider}"
        )

    if not settings.llm.base_url.strip():
        raise ValueError(
            "SSE_ENABLE_LLM_EXTRACTION=true but SSE_LLM_BASE_URL is empty for provider "
            f"{settings.llm.provider}"
        )
