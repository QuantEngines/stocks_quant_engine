from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.schemas import FinancialStatementRecord, OHLCVBar
from stock_screener_engine.data_sources.security_master.csv_loader import load_security_master_csv
from stock_screener_engine.pipelines.backtest_readiness import (
    BacktestReadinessPipeline,
    BacktestReadinessThresholds,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_security_master_csv_loader_accepts_external_universe_metadata(tmp_path: Path) -> None:
    path = tmp_path / "nifty_seed.csv"
    path.write_text(
        "\n".join(
            [
                "symbol,exchange,company_name,sector,industry,active",
                "RELIANCE,NSE,Reliance Industries Ltd,Energy,Oil Gas,true",
                "TCS,NSE,Tata Consultancy Services Ltd,IT,IT Services,true",
            ]
        ),
        encoding="utf-8",
    )

    records = load_security_master_csv(str(path))

    assert [record.symbol for record in records] == ["RELIANCE", "TCS"]
    assert records[0].sector == "Energy"
    assert records[1].industry == "IT Services"


def test_security_master_csv_loader_accepts_official_nse_index_headers(tmp_path: Path) -> None:
    path = tmp_path / "ind_nifty50list.csv"
    path.write_text(
        "\n".join(
            [
                "Company Name,Industry,Symbol,Series,ISIN Code",
                "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018",
            ]
        ),
        encoding="utf-8",
    )

    records = load_security_master_csv(str(path))

    assert records[0].symbol == "RELIANCE"
    assert records[0].company_name == "Reliance Industries Ltd."
    assert records[0].isin == "INE002A01018"
    assert records[0].sector == "Oil Gas & Consumable Fuels"


def test_security_master_csv_loader_accepts_plain_symbol_list(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("RELIANCE\nTCS\n", encoding="utf-8")

    records = load_security_master_csv(str(path))

    assert [record.symbol for record in records] == ["RELIANCE", "TCS"]
    assert records[0].company_name == "RELIANCE"


def test_backtest_readiness_passes_with_minimum_history_and_forward_labels(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    start = date(2026, 1, 1)
    try:
        store.upsert_ohlcv(
            [
                OHLCVBar("NSE", "AAA", (start + timedelta(days=i)).isoformat(), 100 + i, 101 + i, 99 + i, 100 + i, 1000)
                for i in range(12)
            ]
        )
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2025, 12, 31),
                    filing_date=date(2026, 1, 15),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=200.0,
                    net_income=150.0,
                    operating_cash_flow=180.0,
                    capex=40.0,
                    total_debt=100.0,
                    equity=700.0,
                    total_assets=1000.0,
                    current_assets=400.0,
                    current_liabilities=150.0,
                    interest_expense=20.0,
                    source_id="fy25",
                )
            ]
        )
        report = BacktestReadinessPipeline(settings=settings, store=store).run(
            symbols=["AAA"],
            start=start,
            end=start + timedelta(days=11),
            horizons=[1, 5],
            thresholds=BacktestReadinessThresholds(min_history_years=0.01, min_history_rows=10),
        )

        assert report["passed"] is True
        assert report["summary"]["symbols_ready"] == 1
        assert report["per_symbol"]["AAA"]["forward_return_labels"]["5"] == 7
        assert (tmp_path / "quality" / "backtest_readiness_report.json").exists()
    finally:
        store.close()


def test_backtest_readiness_blocks_insufficient_history(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    try:
        store.upsert_ohlcv([OHLCVBar("NSE", "AAA", "2026-01-01", 100.0, 101.0, 99.0, 100.0, 1000.0)])

        report = BacktestReadinessPipeline(settings=settings, store=store).run(
            symbols=["AAA"],
            start=date(2026, 1, 1),
            end=date(2026, 1, 2),
            horizons=[1],
            thresholds=BacktestReadinessThresholds(min_history_rows=10),
        )

        assert report["passed"] is False
        assert "insufficient_history" in report["per_symbol"]["AAA"]["status"]
    finally:
        store.close()
