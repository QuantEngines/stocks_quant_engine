from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from typing import Sequence

import pytest

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.core.entities import FundamentalsSnapshot, GovernanceSnapshot, StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    OHLCVBar,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_research_engine_outputs() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )
    output = engine.run()

    assert output["features"]
    assert output["scores"]
    assert output["long_signals"]
    assert output["swing_signals"]
    assert "long_portfolio_positions" in output
    assert "swing_portfolio_positions" in output
    assert "conviction_evidence" in output
    assert "source_confidence" in output["features"][0].values
    assert "cross_sectional_momentum_rank" in output["features"][0].values
    assert "research_readiness_score" in output["features"][0].values


class _MarketOnlyProvider(MarketDataProvider):
    def get_universe(self) -> list[str]:
        return ["AAA"]

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        return [
            {"date": start.isoformat(), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}
        ]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        return [
            StockSnapshot(
                symbol="AAA",
                as_of=date.today(),
                sector="Unknown",
                close=100.0,
                volume=1_000_000.0,
                delivery_ratio=0.0,
                pe_ratio=0.0,
                roe=0.0,
                debt_to_equity=0.0,
                earnings_growth=0.0,
                free_cash_flow_margin=0.0,
                promoter_holding_change=0.0,
                insider_activity_score=0.0,
            )
        ]


def test_market_only_snapshots_do_not_generate_synthetic_fundamentals() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=_MarketOnlyProvider(),
        text_data=MockTextEventProvider(),
    )
    output = engine.run(regime_score=0.0)
    fv = output["features"][0]

    assert fv.values["growth_quality"] == 0.0
    assert fv.values["profitability_quality"] == 0.0
    assert fv.values["valuation_sanity"] == 0.0


class _HistoricalCanonicalLikeProvider(_MarketOnlyProvider):
    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        return [
            {
                "date": date(2026, 1, day).isoformat(),
                "open": 100.0 + day,
                "high": 101.0 + day,
                "low": 99.0 + day,
                "close": 100.0 + day,
                "volume": 1_000_000,
            }
            for day in range(1, 8)
        ]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        return [
            StockSnapshot(
                symbol="AAA",
                as_of=date(2026, 1, 7),
                sector="IT",
                close=107.0,
                volume=1_000_000.0,
                delivery_ratio=0.0,
                pe_ratio=0.0,
                roe=0.0,
                debt_to_equity=0.0,
                earnings_growth=0.0,
                free_cash_flow_margin=0.0,
                promoter_holding_change=0.0,
                insider_activity_score=0.0,
            )
        ]

    def get_freshness_report(self, symbols: Sequence[str] | None = None) -> dict[str, object]:
        return {
            "passed": True,
            "issues": [],
            "warnings": ["historical canonical data is older than the current session"],
            "metrics": {"symbol_count": len(symbols or [])},
        }


def test_research_engine_allows_historical_canonical_snapshots() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=_HistoricalCanonicalLikeProvider(),
        text_data=MockTextEventProvider(),
    )

    output = engine.run(regime_score=0.0)

    assert output["quality_flags"]["snapshot"]["passed"] is True
    assert output["quality_flags"]["freshness"]["passed"] is True
    assert output["quality_flags"]["freshness"]["warnings"]


class _TwoSymbolCountingProvider(_MarketOnlyProvider):
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def get_universe(self) -> list[str]:
        return ["AAA", "BBB"]

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        return [
            {
                "date": date(2026, 1, day).isoformat(),
                "open": 100.0 + day,
                "high": 101.0 + day,
                "low": 99.0 + day,
                "close": 100.0 + day,
                "volume": 1_000_000 + day,
            }
            for day in range(1, 31)
        ]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        return [
            StockSnapshot(
                symbol=symbol,
                as_of=date(2026, 1, 30),
                sector="IT",
                close=130.0,
                volume=1_000_000.0,
                delivery_ratio=0.5,
                pe_ratio=20.0,
                roe=0.15,
                debt_to_equity=0.2,
                earnings_growth=0.1,
                free_cash_flow_margin=0.1,
                promoter_holding_change=0.0,
                insider_activity_score=0.0,
            )
            for symbol in symbols
        ]


def test_research_engine_reuses_index_history_per_date_window() -> None:
    settings = load_settings()
    market = _TwoSymbolCountingProvider()
    engine = ResearchEngine(
        settings=settings,
        market_data=market,
        text_data=MockTextEventProvider(),
    )

    engine.run(regime_score=0.0)

    assert market.calls["^NSEI"] == 1
    assert market.calls["AAA"] == 1
    assert market.calls["BBB"] == 1


def test_research_engine_uses_canonical_financial_statements(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", sector="IT", industry="Software"),
                SecurityMasterRecord(symbol="BBB", sector="IT", industry="Services"),
                SecurityMasterRecord(symbol="CCC", sector="IT", industry="Software"),
            ]
        )
        store.upsert_ohlcv(
            [
                OHLCVBar("NSE", "AAA", f"2026-05-{day:02d}", 100.0 + day, 102.0 + day, 99.0 + day, 101.0 + day, 2_000_000.0)
                for day in range(1, 12)
            ]
        )
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2025, 3, 31),
                    filing_date=date(2025, 4, 20),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=180.0,
                    net_income=120.0,
                    operating_cash_flow=160.0,
                    capex=40.0,
                    total_debt=250.0,
                    equity=600.0,
                    total_assets=1500.0,
                    current_assets=500.0,
                    current_liabilities=300.0,
                    interest_expense=30.0,
                    source_id="fy25",
                ),
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    statement_type="annual",
                    revenue=1300.0,
                    ebit=270.0,
                    net_income=210.0,
                    operating_cash_flow=280.0,
                    capex=60.0,
                    total_debt=200.0,
                    equity=760.0,
                    total_assets=1800.0,
                    current_assets=700.0,
                    current_liabilities=360.0,
                    interest_expense=24.0,
                    source_id="fy26",
                ),
                _engine_annual_statement("BBB", net_income=100.0, equity=500.0),
                _engine_annual_statement("CCC", net_income=100.0, equity=500.0),
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord(
                    venue="NSE",
                    symbol="AAA",
                    as_of=date(2026, 5, 10),
                    market_cap=12600.0,
                    shares_outstanding=100.0,
                    source_id="mcap",
                ),
                EquityValuationRecord(
                    venue="NSE",
                    symbol="BBB",
                    as_of=date(2026, 5, 10),
                    market_cap=1000.0,
                    shares_outstanding=100.0,
                    source_id="mcap",
                ),
                EquityValuationRecord(
                    venue="NSE",
                    symbol="CCC",
                    as_of=date(2026, 5, 10),
                    market_cap=2000.0,
                    shares_outstanding=100.0,
                    source_id="mcap",
                ),
            ]
        )
        store.upsert_shareholding(
            [
                ShareholdingRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2025, 12, 31),
                    filing_date=date(2026, 1, 20),
                    promoter_pct=48.0,
                    fii_pct=9.0,
                    dii_pct=14.0,
                    public_pct=29.0,
                    source_id="q3",
                ),
                ShareholdingRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    promoter_pct=52.0,
                    fii_pct=11.0,
                    dii_pct=16.0,
                    public_pct=21.0,
                    source_id="q4",
                ),
            ]
        )
    finally:
        store.close()

    settings = load_settings()
    market = SQLiteMarketDataProvider(sqlite_path=str(tmp_path / "market.db"), universe=["AAA"])
    financials = SQLiteFinancialsProvider(sqlite_path=str(tmp_path / "market.db"))
    engine = ResearchEngine(
        settings=settings,
        market_data=market,
        text_data=MockTextEventProvider(),
        financials=financials,
    )
    try:
        output = engine.run(symbols=["AAA"], regime_score=0.0)
    finally:
        market.close()
        financials.close()

    fv = output["features"][0]
    assert fv.values["revenue_growth"] > 0.0
    assert fv.values["profitability_quality"] > 0.0
    assert fv.values["cash_flow_quality"] > 0.0
    assert fv.values["pe_ratio"] == 60.0
    assert fv.values["sector_pe_zscore"] > 0.0
    assert fv.values["sector_pb_zscore"] > 0.0
    assert fv.values["governance_proxy"] > 0.0


class _BankMarketProvider(_MarketOnlyProvider):
    def get_universe(self) -> list[str]:
        return ["BANK"]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        return [
            StockSnapshot(
                symbol="BANK",
                as_of=date(2026, 5, 28),
                sector="Financial Services",
                close=100.0,
                volume=2_000_000.0,
                delivery_ratio=0.4,
                pe_ratio=0.0,
                roe=0.0,
                debt_to_equity=0.0,
                earnings_growth=0.0,
                free_cash_flow_margin=0.0,
                promoter_holding_change=0.0,
                insider_activity_score=0.0,
            )
        ]


class _BankFinancialsProvider:
    def get_fundamentals(self, symbols: Sequence[str]) -> dict[str, FundamentalsSnapshot]:
        return self.get_fundamentals_as_of(symbols, as_of=date(2026, 5, 28))

    def get_fundamentals_as_of(self, symbols: Sequence[str], as_of: date) -> dict[str, FundamentalsSnapshot]:
        return {
            "BANK": FundamentalsSnapshot(
                symbol="BANK",
                as_of=date(2026, 3, 31),
                pe_ratio=15.0,
                pb_ratio=2.0,
                roe=0.14,
                roa=0.017,
                debt_to_equity=0.8,
                earnings_growth_yoy=0.12,
                revenue_growth_yoy=0.10,
                free_cash_flow_margin=0.18,
                operating_margin=0.22,
                net_profit_margin=0.16,
            )
        }

    def get_governance(self, symbols: Sequence[str]) -> dict[str, GovernanceSnapshot]:
        return self.get_governance_as_of(symbols, as_of=date(2026, 5, 28))

    def get_governance_as_of(self, symbols: Sequence[str], as_of: date) -> dict[str, GovernanceSnapshot]:
        return {
            "BANK": GovernanceSnapshot(
                symbol="BANK",
                as_of=date(2026, 3, 31),
                promoter_holding_pct=0.0,
                institutional_holding_pct=80.0,
                fii_holding_pct=45.0,
                dii_holding_pct=35.0,
                insider_activity_score=0.3,
                audit_opinion="clean",
            )
        }

    def get_banking_factors_as_of(self, symbols: Sequence[str], as_of: date) -> dict[str, BankingFactorRecord]:
        return {
            "BANK": BankingFactorRecord(
                venue="NSE",
                symbol="BANK",
                period_end=date(2026, 3, 31),
                filing_date=date(2026, 4, 20),
                net_interest_income=1000.0,
                net_interest_margin_pct=3.2,
                advances_growth_pct=12.0,
                deposits_growth_pct=14.0,
                casa_ratio_pct=0.0,
                gnpa_ratio_pct=1.2,
                nnpa_ratio_pct=0.4,
                provision_coverage_ratio_pct=0.0,
                credit_cost_pct=0.0,
                capital_adequacy_ratio_pct=0.0,
                cet1_ratio_pct=19.7,
                cost_to_income_ratio_pct=68.0,
                roa_pct=1.7,
                roe_pct=13.4,
                loan_to_deposit_ratio_pct=95.0,
            )
        }


def test_research_engine_enriches_financial_services_with_banking_factors() -> None:
    settings = load_settings()
    output = ResearchEngine(
        settings=settings,
        market_data=_BankMarketProvider(),
        text_data=MockTextEventProvider(),
        financials=_BankFinancialsProvider(),
    ).run(symbols=["BANK"], regime_score=0.0)

    fv = output["features"][0]
    score = output["scores"][0]
    assert fv.values["banking_sector_applicable"] == 1.0
    assert fv.values["banking_factor_available"] == 1.0
    assert fv.values["banking_metric_coverage"] > 0.70
    assert fv.values["bank_nim_pct"] == 3.2
    assert fv.values["bank_gnpa_pct"] == 1.2
    assert fv.values["bank_cet1_pct"] == 19.7
    assert fv.values["banking_data_confidence"] == fv.values["banking_metric_coverage"]
    assert score.component_scores["conviction_source_confidence"] > 50.0


def test_research_engine_injects_calibration_evidence_into_conviction(tmp_path) -> None:
    calibration_path = tmp_path / "calibration_report_latest.json"
    calibration_path.write_text(
        json.dumps(
            {
                "net_quantile_ic": {"5": 0.08, "20": 0.04},
                "net_horizon_metrics": {
                    "5": {"top_quantile_hit_rate": 0.62, "avg_quantile_spread": 0.012},
                    "20": {"top_quantile_hit_rate": 0.68, "avg_quantile_spread": 0.018},
                },
                "report": {
                    "quantile_ic": {"5": 0.05, "20": 0.03},
                    "decay": {"5": 0.04, "20": 0.02},
                    "turnover_top_quantile": {"5": 0.2, "20": 0.3},
                },
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings()
    settings = replace(
        settings,
        scoring=replace(
            settings.scoring,
            calibration_auto_tune=replace(
                settings.scoring.calibration_auto_tune,
                report_path=str(calibration_path),
            ),
        ),
    )

    output = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    ).run(regime_score=0.0)

    fv = output["features"][0]
    score = output["scores"][0]
    assert output["conviction_evidence"]["backtest_evidence_loaded"] is True
    assert fv.values["backtest_information_coefficient"] == pytest.approx(0.06)
    assert fv.values["backtest_hit_rate"] == pytest.approx(0.65)
    assert 0.0 <= fv.values["source_confidence"] <= 1.0
    assert score.component_scores["conviction_backtest_evidence"] > 50.0


def _engine_annual_statement(symbol: str, net_income: float, equity: float) -> FinancialStatementRecord:
    return FinancialStatementRecord(
        venue="NSE",
        symbol=symbol,
        period_end=date(2026, 3, 31),
        filing_date=date(2026, 4, 20),
        statement_type="annual",
        revenue=net_income * 8.0,
        ebit=net_income * 1.25,
        net_income=net_income,
        operating_cash_flow=net_income * 1.1,
        capex=net_income * 0.2,
        total_debt=equity * 0.2,
        equity=equity,
        total_assets=equity * 2.0,
        current_assets=equity * 0.8,
        current_liabilities=equity * 0.3,
        interest_expense=max(1.0, net_income * 0.1),
        source_id="fy26",
    )
