from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from stock_screener_engine.data_sources.finedge import FinEdgeClient, FinEdgeFactorMapper


def test_finedge_factor_mapper_exports_financial_and_shareholding_csvs(tmp_path: Path) -> None:
    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        if url.endswith("/financials/ITC") and params["statement_code"] == "pl":
            return {
                "financials": [
                    {
                        "period_end": "2025-03-31",
                        "result_date": "2025-05-20",
                        "income": 1000,
                        "profitLossForPeriod": 120,
                        "financeCosts": 10,
                        "currentTax": 20,
                    }
                ]
            }
        if url.endswith("/financials/ITC") and params["statement_code"] == "bs":
            return {
                "financials": [
                    {
                        "period_end": "2025-03-31",
                        "assets": 2000,
                        "currentAssets": 700,
                        "currentLiabilities": 300,
                        "borrowingsCurrent": 50,
                        "borrowingsNoncurrent": 150,
                        "equity": 1000,
                    }
                ]
            }
        if url.endswith("/financials/ITC") and params["statement_code"] == "cf":
            return {
                "financials": [
                    {
                        "period_end": "2025-03-31",
                        "netCashFlowsFromUsedInOperatingActivities": 180,
                        "purchaseOfPropertyPlantAndEquipment": -40,
                    }
                ]
            }
        if url.endswith("/quote"):
            return {
                "ITC": {
                    "current_price": 100,
                    "shares": 1000,
                    "market_cap": 0.01,
                    "tradetime": "2025-05-20T15:30:00Z",
                }
            }
        if url.endswith("/shareholdings/pattern/ITC"):
            return {
                "symbol": "ITC",
                "columns": ["Dec 2024", "Mar 2025"],
                "rows": [
                    {"catagory": "Indian", "name": "Promoter Indian", "data": {"Mar 2025": 40}},
                    {"catagory": "InstitutionsForeign", "name": "Institutions foreign", "data": {"Mar 2025": 12}},
                    {"catagory": "InstitutionsDomestic", "name": "Institutions domestic", "data": {"Mar 2025": 8}},
                    {"catagory": "NonInstitutions", "name": "Non-institutions", "data": {"Mar 2025": 40}},
                ],
            }
        if url.endswith("/shareholdings/ownership-history/ITC"):
            return {
                "symbol": "ITC",
                "period": "quarterly",
                "ownership_history": [
                    {"header": "Dec 2024", "data": []},
                    {"header": "Mar 2025", "data": [
                        {
                            "shareholder_name": "Promoter and Promoter Group",
                            "shareholdingPct": 40,
                            "pledgedShares": 0,
                            "pledgedSharesPct": 0,
                        }
                    ]},
                ],
            }
        raise AssertionError(url)

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeFactorMapper(client).export(
        symbols=["ITC"],
        as_of=date(2026, 5, 28),
        output_root=str(tmp_path),
        sections=["financials", "valuations", "shareholding"],
    )

    assert report["passed"] is True
    assert report["row_counts"]["financials"] == 1
    assert report["row_counts"]["valuations"] == 1
    assert report["row_counts"]["shareholding"] == 1

    financial_rows = list(csv.DictReader((tmp_path / "financials.csv").open()))
    assert financial_rows[0]["symbol"] == "ITC"
    assert financial_rows[0]["statement_type"] == "annual_standalone"
    assert financial_rows[0]["revenue"] == "1000.0"
    assert financial_rows[0]["ebit"] == "150.0"
    assert financial_rows[0]["operating_cash_flow"] == "180.0"
    assert financial_rows[0]["capex"] == "40.0"
    assert financial_rows[0]["total_debt"] == "200.0"

    valuation_rows = list(csv.DictReader((tmp_path / "valuations.csv").open()))
    assert valuation_rows[0]["as_of"] == "2025-05-20"
    assert valuation_rows[0]["market_cap"] == "100000.0"
    assert valuation_rows[0]["shares_outstanding"] == "1000.0"

    shareholding_rows = list(csv.DictReader((tmp_path / "shareholding.csv").open()))
    assert shareholding_rows[0]["promoter_pct"] == "40.0"
    assert shareholding_rows[0]["fii_pct"] == "12.0"
    assert shareholding_rows[0]["dii_pct"] == "8.0"
    assert shareholding_rows[0]["public_pct"] == "40.0"

    detail_rows = list(csv.DictReader((tmp_path / "finedge_ownership_details.csv").open()))
    assert detail_rows[0]["pledged_shares_pct"] == "0.0"

    saved_report = json.loads((tmp_path / "finedge_factor_export_report.json").read_text())
    assert saved_report["notes"][0] == "No raw FinEdge payloads are persisted by this mapper."


def test_finedge_factor_mapper_exports_banking_csv(tmp_path: Path) -> None:
    def fake_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        if url.endswith("/financials/HDFCBANK") and params["statement_code"] == "pl":
            return {
                "financials": [
                    {"period_end": "2024-03-31", "interestEarned": 1000, "interestExpended": 600},
                    {"period_end": "2025-03-31", "interestEarned": 1200, "interestExpended": 700},
                ]
            }
        if url.endswith("/financials/HDFCBANK") and params["statement_code"] == "bs":
            return {
                "financials": [
                    {"period_end": "2024-03-31", "advances": 8000, "deposits": 10000},
                    {"period_end": "2025-03-31", "advances": 9200, "deposits": 11000},
                ]
            }
        if url.endswith("/ratios/HDFCBANK"):
            return {
                "ratios": [
                    {"year": 2024, "netInterestMargin": 0.032, "returnOnAsset": 0.017, "returnOnEquity": 0.15},
                    {"year": 2025, "netInterestMargin": 0.034, "returnOnAsset": 0.018, "returnOnEquity": 0.16},
                ]
            }
        if url.endswith("/basic-financials/HDFCBANK") and params["statement_code"] == "pl":
            return {
                "ratios": [
                    {
                        "year": 2024,
                        "percentageOfGrossNpa": 0.014,
                        "percentageOfNpa": 0.005,
                        "CET1Ratio": 0.165,
                        "operatingExpenses": 360,
                        "operatingRevenue": 1000,
                    },
                    {
                        "year": 2025,
                        "percentageOfGrossNpa": 0.012,
                        "percentageOfNpa": 0.004,
                        "CET1Ratio": 0.17,
                        "operatingExpenses": 420,
                        "operatingRevenue": 1200,
                    },
                ]
            }
        if url.endswith("/basic-financials/HDFCBANK") and params["statement_code"] == "bs":
            return {
                "ratios": [
                    {"year": 2024, "loans": 8000},
                    {"year": 2025, "loans": 9200},
                ]
            }
        raise AssertionError(url)

    client = FinEdgeClient(api_key="test-token", fetch_fn=fake_fetch)
    report = FinEdgeFactorMapper(client).export(
        symbols=["HDFCBANK"],
        as_of=date(2026, 5, 28),
        output_root=str(tmp_path),
        sections=["banking"],
    )

    assert report["passed"] is True
    assert report["row_counts"]["banking"] == 2
    banking_rows = list(csv.DictReader((tmp_path / "banking.csv").open()))
    assert banking_rows[0]["symbol"] == "HDFCBANK"
    assert banking_rows[0]["net_interest_income"] == "500.0"
    assert banking_rows[0]["net_interest_margin_pct"] == "3.4"
    assert banking_rows[0]["advances_growth_pct"] == "15.0"
    assert banking_rows[0]["deposits_growth_pct"] == "10.0"
    assert banking_rows[0]["gnpa_ratio_pct"] == "1.2"
    assert banking_rows[0]["nnpa_ratio_pct"] == "0.4"
    assert banking_rows[0]["cet1_ratio_pct"] == "17.0"
    assert banking_rows[0]["cost_to_income_ratio_pct"] == "35.0"
    assert banking_rows[0]["roa_pct"] == "1.8"
    assert banking_rows[0]["roe_pct"] == "16.0"
