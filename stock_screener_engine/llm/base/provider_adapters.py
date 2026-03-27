"""Provider adapters implementing LLMClient for OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from stock_screener_engine.llm.base.llm_client import LLMClient


class LLMProviderError(RuntimeError):
    """Raised when provider responses are malformed or transport fails."""


Transport = Callable[[str, dict[str, str], dict[str, object], int], dict[str, object]]


@dataclass
class OpenAICompatibleLLMClient(LLMClient):
    """Uses OpenAI-compatible chat-completions JSON mode for structured payloads."""

    api_key: str
    model_name: str
    base_url: str = "https://api.openai.com"
    timeout_seconds: int = 30
    transport: Transport | None = None

    def complete(self, task: str, prompt: str, payload: dict[str, object]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "payload": payload,
                            "instruction": "Return exactly one valid JSON object.",
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._send(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            body=body,
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("openai response missing choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return _parse_json_content(content)

    def _send(self, url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        tx = self.transport or _default_transport
        return tx(url, headers, body, self.timeout_seconds)


@dataclass
class AnthropicLLMClient(LLMClient):
    """Uses Anthropic messages API and parses text output as JSON."""

    api_key: str
    model_name: str
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: int = 30
    max_tokens: int = 800
    anthropic_version: str = "2023-06-01"
    transport: Transport | None = None

    def complete(self, task: str, prompt: str, payload: dict[str, object]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": f"{prompt}\nReturn exactly one valid JSON object and no markdown.",
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "payload": payload,
                        },
                        separators=(",", ":"),
                    ),
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        data = self._send(
            f"{self.base_url.rstrip('/')}/v1/messages",
            headers=headers,
            body=body,
        )
        chunks = data.get("content")
        if not isinstance(chunks, list) or not chunks:
            raise LLMProviderError("anthropic response missing content")
        first = chunks[0]
        if not isinstance(first, dict):
            raise LLMProviderError("anthropic response content item invalid")
        text = first.get("text", "")
        return _parse_json_content(text)

    def _send(self, url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        tx = self.transport or _default_transport
        return tx(url, headers, body, self.timeout_seconds)


def _parse_json_content(content: object) -> dict[str, object]:
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("provider returned empty response content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("provider returned non-JSON content") from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError("provider JSON content must be an object")
    return parsed


def _default_transport(
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    timeout_seconds: int,
) -> dict[str, object]:
    payload = json.dumps(body).encode("utf-8")
    req = Request(url=url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, OSError) as exc:
        raise LLMProviderError(f"provider transport failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("provider transport returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise LLMProviderError("provider transport returned non-object JSON")
    return parsed
