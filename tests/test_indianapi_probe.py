from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stock_screener_engine.data_sources.indianapi import IndianAPIClient, IndianAPIProbe


def test_indianapi_probe_summarizes_endpoint_coverage_without_raw_payloads() -> None:
    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        assert headers.get("x-api-key") == "test-key"
        if url.endswith("/industry_search"):
            return [{"id": "S0001", "commonName": "Reliance Industries", "exchangeCodeNsi": "RELIANCE"}]
        if url.endswith("/stock"):
            return {"id": "S0001", "tickerId": params["name"], "financials": {"Sales": 100}}
        if url.endswith("/historical_stats") and params["stats"] == "quarter_results":
            return {"Sales": {"Jun 2025": 100}, "Net Profit": {"Jun 2025": 10}}
        if url.endswith("/historical_stats") and params["stats"] == "shareholding_pattern_quarterly":
            return {"Promoters": {"Jun 2025": 50.0}, "FIIs": {"Jun 2025": 20.0}}
        if url.endswith("/historical_data"):
            return {"datasets": [{"metric": "Price", "values": [["2026-05-01", "100"]]}]}
        if url.endswith("/stock_target_price"):
            assert params["stock_id"] == "S0001"
            return {"priceTarget": {"Mean": 120}, "recommendation": {"Mean": 2.0}}
        if url.endswith("/stock_forecasts"):
            assert params["stock_id"] == "S0001"
            return {"EPS": {"FY2026": 10.0}}
        raise AssertionError(url)

    client = IndianAPIClient(api_key="test-key", fetch_fn=fake_fetch)
    report = IndianAPIProbe(client).run(
        symbols=["RELIANCE"],
        checks=["stock", "financials", "shareholding", "analyst", "forecasts", "history"],
    )

    assert report["passed"] is True
    assert report["coverage"]["search"]["coverage"] == 1.0
    assert report["coverage"]["financials"]["coverage"] == 1.0
    checks = report["symbol_reports"][0]["checks"]
    assert checks["stock"]["stock_id"] == "S0001"
    assert report["symbol_reports"][0]["resolved_stock_id"] == "S0001"
    assert "_payload" not in checks["financials"]["summary"]
    assert "Sales" not in checks["financials"]["summary"]


def test_indianapi_probe_treats_error_and_info_only_payloads_as_unusable() -> None:
    client = IndianAPIClient(fetch_fn=lambda *args, **kwargs: {"info": "No Data Available"})
    report = IndianAPIProbe(client).run(symbols=["RELIANCE"], checks=["corporate_actions"])

    result = report["symbol_reports"][0]["checks"]["corporate_actions"]

    assert result["ok"] is False
    assert "No Data Available" in result["error"]
    assert report["coverage"]["corporate_actions"]["coverage"] == 0.0


def test_indianapi_probe_runs_market_level_checks_once() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> list[dict[str, Any]]:
        calls.append(url)
        return [{"ticker": "RELIANCE.NS", "price": 100.0}]

    client = IndianAPIClient(fetch_fn=fake_fetch)
    report = IndianAPIProbe(client).run(symbols=["RELIANCE", "TCS"], checks=["nse_most_active"])

    assert calls == ["https://stock.indianapi.in/NSE_most_active"]
    assert report["coverage"]["nse_most_active"]["coverage"] == 1.0
    assert report["market_report"]["usable_sections"] == ["nse_most_active"]
