from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stock_screener_engine.data_sources.finedge import FinEdgeClient, FinEdgeProbe, FinEdgeSchemaInspector
from stock_screener_engine.data_sources.finedge.client import normalize_finedge_checks


def test_finedge_probe_summarizes_coverage_without_raw_payloads() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        calls.append((url, dict(params)))
        assert params.get("token") == "test-token"
        if url.endswith("/stock-symbols"):
            return {"symbols": [{"symbol": "ITC", "name": "ITC Ltd"}]}
        if url.endswith("/company-profile/ITC"):
            return {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG"}
        if url.endswith("/financials/ITC"):
            assert params["statement_type"] == "s"
            assert params["statement_code"] == "pl"
            assert params["period"] == "annual"
            return {"symbol": "ITC", "financials": [{"year": "FY2025", "income": 100}]}
        if url.endswith("/ratios/ITC"):
            return {"symbol": "ITC", "ratios": [{"year": "FY2025", "returnOnEquity": 0.2}]}
        raise AssertionError(url)

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeProbe(client).run(symbols=["ITC"], checks=["stock_symbols", "company_profile", "financials", "ratios"])

    assert report["passed"] is True
    assert report["coverage"]["stock_symbols"]["coverage"] == 1.0
    assert report["coverage"]["financials"]["coverage"] == 1.0
    checks = report["symbol_reports"][0]["checks"]
    assert "_payload" not in checks["financials"]["summary"]
    assert calls[0][0].endswith("/stock-symbols")


def test_finedge_probe_halts_after_rate_limit() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        calls.append(url)
        raise RuntimeError("financials failed: HTTP Error 429: Too Many Requests")

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeProbe(client).run(symbols=["ITC", "RELIANCE"], checks=["financials", "ratios"])

    assert calls == ["https://data.finedgeapi.com/api/v1/financials/ITC"]
    assert report["symbol_reports"][0]["checks"]["ratios"]["skipped"] is True
    assert report["symbol_reports"][1]["skipped"] is True


def test_finedge_corporate_actions_forwards_date_window() -> None:
    seen_params: dict[str, Any] = {}

    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        assert url.endswith("/corporate-actions/all")
        seen_params.update(dict(params))
        return {"data": [{"symbol": "ITC", "action": "dividend"}]}

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeProbe(
        client,
        from_date="2026-01-01",
        to_date="2026-05-28",
    ).run(symbols=["ITC"], checks=["corporate_actions"])

    assert report["coverage"]["corporate_actions"]["coverage"] == 1.0
    assert seen_params["symbol"] == "ITC"
    assert seen_params["from_date"] == "2026-01-01"
    assert seen_params["to_date"] == "2026-05-28"


def test_finedge_schema_inspector_sanitizes_values_and_profiles_records() -> None:
    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        assert url.endswith("/financials/ITC")
        return {
            "symbol": "ITC",
            "financials": [
                {
                    "period": "FY2025",
                    "revenue": 100,
                    "secret_commentary": "raw management sentence must not leak",
                }
            ],
        }

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeSchemaInspector(client).run(symbols=["ITC"], checks=["financials"])

    summary = report["symbol_reports"][0]["checks"]["financials"]["summary"]
    primary = summary["primary_record_set"]
    assert report["pipeline"] == "finedge_schema_inspection"
    assert primary["path"] == "$.financials"
    assert "revenue" in primary["fields"]
    assert "period" in primary["date_like_fields"]
    assert "revenue" in primary["numeric_like_fields"]
    serialized = str(report)
    assert "raw management sentence must not leak" not in serialized
    assert "FY2025" not in serialized
    assert "100" not in serialized


def test_finedge_check_aliases_expand_without_duplicates() -> None:
    assert normalize_finedge_checks(["smoke", "fundamentals", "financials"]) == [
        "stock_symbols",
        "company_profile",
        "financials",
        "ratios",
        "financial_metrics",
        "basic_financials",
    ]
