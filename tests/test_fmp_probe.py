from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from stock_screener_engine.data_sources.fmp import FMPClient, FMPProbe
from stock_screener_engine.data_sources.fmp.client import normalize_fmp_checks


def test_fmp_probe_resolves_indian_symbol_and_summarizes_coverage_without_raw_payloads() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> list[dict[str, Any]]:
        calls.append((url, dict(params)))
        assert params.get("apikey") == "test-key"
        if url.endswith("/search-symbol") or url.endswith("/search-name"):
            return [
                {
                    "symbol": "RELIANCE.NS",
                    "name": "Reliance Industries Limited",
                    "exchangeShortName": "NSE",
                    "currency": "INR",
                }
            ]
        if url.endswith("/income-statement"):
            assert params["symbol"] == "RELIANCE.NS"
            return [{"date": "2025-03-31", "revenue": 100, "netIncome": 10}]
        if url.endswith("/ratios"):
            assert params["symbol"] == "RELIANCE.NS"
            return [{"date": "2025-03-31", "returnOnEquity": 0.12}]
        if url.endswith("/historical-price-eod/full"):
            assert params["from"] == "2026-01-01"
            assert params["to"] == "2026-05-01"
            return [{"date": "2026-05-01", "close": 100.0, "volume": 1000000}]
        raise AssertionError(url)

    client = FMPClient(api_key="test-key", fetch_fn=fake_fetch)
    report = FMPProbe(
        client,
        price_start=date(2026, 1, 1),
        price_end=date(2026, 5, 1),
    ).run(
        symbols=["RELIANCE"],
        checks=["search", "income_statement", "ratios", "price_history"],
    )

    assert report["passed"] is True
    assert report["coverage"]["income_statement"]["coverage"] == 1.0
    assert report["coverage"]["income_statement"]["sample_resolved_symbols"] == ["RELIANCE.NS"]
    assert report["symbol_reports"][0]["resolved_symbol"] == "RELIANCE.NS"
    assert "_payload" not in report["symbol_reports"][0]["checks"]["income_statement"]["summary"]
    assert any(call[0].endswith("/income-statement") for call in calls)


def test_fmp_probe_treats_empty_and_error_payloads_as_unusable() -> None:
    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any] | list[Any]:
        if url.endswith("/search-symbol") or url.endswith("/search-name"):
            return []
        return {"Error Message": "Upgrade plan required"}

    client = FMPClient(fetch_fn=fake_fetch)
    report = FMPProbe(client).run(symbols=["TCS"], checks=["profile"])

    result = report["symbol_reports"][0]["checks"]["profile"]
    assert result["ok"] is False
    assert "Upgrade plan required" in result["error"]
    assert report["coverage"]["profile"]["coverage"] == 0.0


def test_fmp_probe_stops_after_rate_limit_without_trying_more_variants() -> None:
    attempted_symbols: list[str] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> list[dict[str, Any]]:
        attempted_symbols.append(str(params["symbol"]))
        raise RuntimeError("income_statement failed: HTTP Error 429: Too Many Requests")

    client = FMPClient(api_key="test-key", fetch_fn=fake_fetch)
    report = FMPProbe(client).run(
        symbols=["RELIANCE", "TCS"],
        checks=["income_statement", "balance_sheet"],
    )

    assert attempted_symbols == ["RELIANCE.NS"]
    first_symbol_checks = report["symbol_reports"][0]["checks"]
    assert first_symbol_checks["income_statement"]["terminal_error"] == "rate_limit"
    assert first_symbol_checks["balance_sheet"]["skipped"] is True
    second_symbol_checks = report["symbol_reports"][1]["checks"]
    assert second_symbol_checks["income_statement"]["skipped"] is True
    assert "rate limit" in report["coverage"]["balance_sheet"]["sample_errors"][0].lower()


def test_fmp_probe_exact_symbols_and_smoke_expansion() -> None:
    attempted_symbols: list[str] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> list[dict[str, Any]]:
        if url.endswith("/search-symbol") or url.endswith("/search-name"):
            return [{"symbol": "AAPL", "name": "Apple Inc.", "exchangeShortName": "NASDAQ"}]
        attempted_symbols.append(str(params["symbol"]))
        return [{"symbol": params["symbol"], "date": "2025-09-30", "revenue": 100}]

    client = FMPClient(api_key="test-key", fetch_fn=fake_fetch)
    report = FMPProbe(client, exact_symbols=True).run(
        symbols=["AAPL"],
        checks=["smoke", "income_statement"],
    )

    assert normalize_fmp_checks(["smoke", "income_statement"]) == ["search", "profile", "income_statement"]
    assert attempted_symbols == ["AAPL", "AAPL"]
    assert report["coverage"]["profile"]["coverage"] == 1.0
    assert report["coverage"]["income_statement"]["coverage"] == 1.0


def test_fmp_probe_stops_after_search_rate_limit() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> list[dict[str, Any]]:
        calls.append(url)
        if url.endswith("/search-symbol") or url.endswith("/search-name"):
            raise RuntimeError("search failed: HTTP Error 429: Too Many Requests")
        raise AssertionError("probe should not continue after search 429")

    client = FMPClient(api_key="test-key", fetch_fn=fake_fetch)
    report = FMPProbe(client, exact_symbols=True).run(
        symbols=["AAPL"],
        checks=["search", "profile", "income_statement"],
    )

    assert len(calls) == 2
    assert report["symbol_reports"][0]["checks"]["profile"]["skipped"] is True
    assert report["symbol_reports"][0]["checks"]["income_statement"]["skipped"] is True
