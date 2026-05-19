from __future__ import annotations

from datetime import date
from typing import Sequence

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.schemas import (
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
