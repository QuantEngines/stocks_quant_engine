from __future__ import annotations

from pathlib import Path

from stock_screener_engine.config.settings import DataSourceEntitlementSettings
from stock_screener_engine.pipelines.missing_data import build_missing_data_report, missing_data_rows_for_csv


def test_missing_data_report_removes_reusable_sibling_macro_variables(tmp_path: Path) -> None:
    macro_config = tmp_path / "macros_quant_engine" / "configs"
    macro_config.mkdir(parents=True)
    (macro_config / "macro_universe.yaml").write_text(
        """
macro_series:
  india_cpi: {}
  india_iip: {}
  pmi_manufacturing: {}
market_proxies:
  usd_inr: {}
  nifty_bank: {}
  nse_market_breadth: {}
""",
        encoding="utf-8",
    )

    rates_config = tmp_path / "rates_quant_engine" / "configs"
    rates_config.mkdir(parents=True)
    (rates_config / "data_sources.yaml").write_text(
        """
series:
  repo_rate: {}
  US_10Y: {}
  BRENT_CRUDE: {}
""",
        encoding="utf-8",
    )

    report = build_missing_data_report(quant_root=tmp_path, entitlements=[])
    removed_names = {
        row["name"]
        for row in report["removed_from_stock_specific_procurement"]
        if isinstance(row, dict)
    }
    still_missing = {
        row["name"]
        for row in report["still_missing"]
        if isinstance(row, dict)
    }

    assert "usd_inr" in removed_names
    assert "cpi_inflation" in removed_names
    assert "repo_rate" in removed_names
    assert "brent_crude" in removed_names
    assert "point_in_time_as_of_date" in still_missing
    assert "# Stock Engine Missing Data List" in report["markdown"]
    csv_rows = missing_data_rows_for_csv(report)
    usd_row = next(row for row in csv_rows if row["variable"] == "usd_inr")
    assert usd_row["status"] == "covered_upstream_not_wired"
    assert usd_row["definition"] == "USD/INR exchange rate."
    assert "macros_quant_engine" in str(usd_row["upstream_coverage"])


def test_missing_data_report_keeps_partial_variables_separate(tmp_path: Path) -> None:
    report = build_missing_data_report(
        quant_root=tmp_path,
        entitlements=[
            DataSourceEntitlementSettings(
                source_id="finedge",
                display_name="FinEdge",
                role="Fundamental data vendor",
                status="basic_sandbox",
                domains=["financials", "valuations"],
                next_action="Upgrade after sandbox validation.",
            )
        ],
        include_cross_engine=False,
    )
    partial_names = {
        row["name"]
        for row in report["partial_or_needs_validation"]
        if isinstance(row, dict)
    }
    source_ids = {
        row["source_id"]
        for row in report["source_summary"]
        if isinstance(row, dict)
    }

    assert "deposit_growth" in partial_names
    assert "official_delivery_percentage" in partial_names
    assert "usd_inr" in {row["name"] for row in report["still_missing"] if isinstance(row, dict)}
    assert "finedge" in source_ids
