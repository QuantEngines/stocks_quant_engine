from __future__ import annotations

from stock_screener_engine.pipelines.coverage_gates import (
    CoverageGateEvaluator,
    FinEdgeOnboardingPlanner,
    build_data_readiness_report,
)


def _coverage_report(financials: float = 0.06) -> dict[str, object]:
    return {
        "as_of": "2026-05-28",
        "start": "2021-05-28",
        "domains": [
            {
                "domain": "security_master",
                "label": "Security master metadata",
                "coverage": 1.0,
                "missing_count": 0,
            },
            {
                "domain": "daily_ohlcv",
                "label": "Daily OHLCV history",
                "coverage": 1.0,
                "missing_count": 0,
            },
            {
                "domain": "financials",
                "label": "Point-in-time financial statements",
                "coverage": financials,
                "missing_count": 47,
                "entitlement_explanation": "Covered symbols match Basic entitlement.",
            },
            {"domain": "valuations", "label": "Valuations", "coverage": financials, "missing_count": 47},
            {"domain": "shareholding", "label": "Shareholding", "coverage": financials, "missing_count": 47},
            {"domain": "events_documents", "label": "Events", "coverage": 0.0, "missing_count": 50},
        ],
        "sources": [
            {
                "source_id": "finedge",
                "source": "FinEdge",
                "status": "primary_paid_candidate",
            }
        ],
        "entitlements": [
            {
                "source_id": "finedge",
                "display_name": "FinEdge",
                "plan_name": "Basic Free",
                "status": "basic_sandbox",
            }
        ],
        "gross_coverage": {"critical_domain_average": 0.61},
    }


def test_long_term_gate_blocks_when_factors_are_sparse() -> None:
    result = CoverageGateEvaluator().evaluate(_coverage_report(), mode="long-term-scan")

    assert result.passed is False
    assert result.decision == "block"
    assert {issue.domain for issue in result.issues} >= {"financials", "valuations", "shareholding"}
    assert "Covered symbols match Basic entitlement" in result.issues[0].message


def test_swing_gate_passes_with_market_data_only() -> None:
    result = CoverageGateEvaluator().evaluate(_coverage_report(), mode="swing_scan")

    assert result.passed is True
    assert result.decision == "pass"


def test_data_readiness_report_has_markdown_and_rows() -> None:
    report = build_data_readiness_report(coverage_report=_coverage_report(), mode="long_term_scan")

    assert report["decision"] == "block"
    assert report["console_rows"]
    assert "# Data Readiness Gate" in report["markdown"]


def test_finedge_onboarding_plan_generates_post_subscription_commands() -> None:
    coverage = _coverage_report()
    gate = build_data_readiness_report(coverage_report=coverage, mode="long_term_scan")

    report = FinEdgeOnboardingPlanner().build(
        coverage_report=coverage,
        gate_report=gate,
        universe_file="/tmp/nifty50.csv",
        as_of="2026-05-28",
        factor_root="/tmp/factors/finedge_paid_2026-05-28",
    )

    assert report["pipeline"] == "finedge_onboarding_plan"
    assert any("finedge-factor-export" in command for command in report["commands_after_subscription"])
    assert "# FinEdge Paid Onboarding Plan" in report["markdown"]
