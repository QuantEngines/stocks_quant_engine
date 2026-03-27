from __future__ import annotations

from urllib.error import URLError

from stock_screener_engine.data_sources.exchange.http_client import RetryingHTTPClient


class _FakeClient(RetryingHTTPClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def _get_json_with_retry(self, url: str, headers: dict[str, str]) -> dict:  # type: ignore[override]
        self.calls.append(url)
        if "primary" in url:
            raise URLError("primary down")
        return {"ok": True, "url": url}


def test_fallback_endpoint_used_when_primary_fails() -> None:
    c = _FakeClient()
    out = c.get_json(urls=["https://primary.example/api", "https://fallback.example/api"])
    assert out["ok"] is True
    assert any("primary" in call for call in c.calls)
    assert any("fallback" in call for call in c.calls)
