from __future__ import annotations

from stock_screener_engine.llm.base.provider_adapters import AnthropicLLMClient, OpenAICompatibleLLMClient


def test_openai_adapter_parses_structured_json_response() -> None:
    def fake_transport(url: str, headers: dict[str, str], body: dict[str, object], timeout: int) -> dict[str, object]:
        assert url.endswith("/v1/chat/completions")
        assert "Authorization" in headers
        assert body["response_format"] == {"type": "json_object"}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"event_type":"earnings_result","confidence":0.86}'
                    }
                }
            ]
        }

    client = OpenAICompatibleLLMClient(
        api_key="k",
        model_name="gpt-4.1-mini",
        transport=fake_transport,
    )
    payload = client.complete("event_extraction", "Extract event", {"title": "x", "text": "y"})
    assert payload["event_type"] == "earnings_result"
    assert payload["confidence"] == 0.86


def test_anthropic_adapter_parses_structured_json_response() -> None:
    def fake_transport(url: str, headers: dict[str, str], body: dict[str, object], timeout: int) -> dict[str, object]:
        assert url.endswith("/v1/messages")
        assert headers["anthropic-version"] == "2023-06-01"
        assert body["temperature"] == 0
        return {
            "content": [
                {
                    "type": "text",
                    "text": '{"category":"earnings_related","confidence":0.8}',
                }
            ]
        }

    client = AnthropicLLMClient(
        api_key="k",
        model_name="claude-3-5-sonnet",
        transport=fake_transport,
    )
    payload = client.complete("classification", "Classify", {"title": "x", "text": "y"})
    assert payload["category"] == "earnings_related"
    assert payload["confidence"] == 0.8
