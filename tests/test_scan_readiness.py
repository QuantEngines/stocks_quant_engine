from __future__ import annotations

from stock_screener_engine.app import _compose_scan_readiness_report


def _coverage_report() -> dict[str, object]:
    return {
        "as_of": "2026-05-28",
        "start": "2021-05-28",
        "domains": [
            {"domain": "security_master", "label": "Security master", "coverage": 1.0, "missing_count": 0},
            {"domain": "daily_ohlcv", "label": "Daily OHLCV", "coverage": 1.0, "missing_count": 0},
            {"domain": "financials", "label": "Financials", "coverage": 0.06, "missing_count": 47},
            {"domain": "valuations", "label": "Valuations", "coverage": 0.06, "missing_count": 47},
            {"domain": "shareholding", "label": "Shareholding", "coverage": 0.06, "missing_count": 47},
        ],
        "gross_coverage": {"critical_domain_average": 0.61},
    }


def test_full_scan_readiness_allows_swing_when_long_term_blocks() -> None:
    report = _compose_scan_readiness_report(coverage_report=_coverage_report(), scan_mode="full")

    assert report["decision"] == "partial"
    assert report["passed"] is False
    assert report["signal_permissions"] == {"long_term": False, "swing": True}
    assert any(row["signal"] == "long_term" and row["domain"] == "financials" for row in report["console_rows"])
    assert any(row["signal"] == "swing" and row["severity"] == "pass" for row in report["console_rows"])


def test_daily_scan_readiness_blocks_long_term_only() -> None:
    report = _compose_scan_readiness_report(coverage_report=_coverage_report(), scan_mode="daily")

    assert report["decision"] == "block"
    assert report["signal_permissions"] == {"long_term": False}


def test_swing_scan_readiness_passes_with_market_data_only() -> None:
    report = _compose_scan_readiness_report(coverage_report=_coverage_report(), scan_mode="swing")

    assert report["decision"] == "pass"
    assert report["signal_permissions"] == {"swing": True}
