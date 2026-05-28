from __future__ import annotations

from stock_screener_engine.config.settings import DataSourceEntitlementSettings
from stock_screener_engine.pipelines.source_priority import build_source_priority_report


def test_source_priority_marks_finedge_sandbox_domains() -> None:
    report = build_source_priority_report(
        entitlements=[
            DataSourceEntitlementSettings(
                source_id="finedge",
                display_name="FinEdge",
                role="fundamentals",
                status="basic_sandbox",
                plan_name="Basic Free",
                enabled=True,
                next_action="Upgrade paid plan.",
            )
        ]
    )

    rows = {row["domain"]: row for row in report["rows"]}

    assert rows["financials"]["status"] == "sandbox_ready"
    assert rows["financials"]["primary_statuses"][0]["plan_name"] == "Basic Free"
    assert "Data Source Priority Map" in report["markdown"]


def test_source_priority_flags_missing_primary_source() -> None:
    report = build_source_priority_report(entitlements=[])
    rows = {row["domain"]: row for row in report["rows"]}

    assert rows["analyst_estimates"]["status"] == "gap"
    assert any(row["domain"] == "analyst_estimates" for row in report["gaps"])
