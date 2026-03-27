"""Tests for the SingleStockPipeline deep analysis mode."""

from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.pipelines.single_stock_deep import (
    SingleStockPipeline,
    _rsi,
    _sma,
    _macd,
    _EnrichedSnapshotProvider,
)
from stock_screener_engine.core.entities import StockSnapshot
from datetime import date


# ---------------------------------------------------------------------------
# Technical indicator unit tests
# ---------------------------------------------------------------------------

def test_rsi_returns_none_when_insufficient_data() -> None:
    assert _rsi([100.0] * 10) is None


def test_rsi_overbought_range() -> None:
    # Monotonically rising prices → RSI should be high
    closes = [float(i) for i in range(1, 30)]
    result = _rsi(closes)
    assert result is not None
    assert result > 60.0


def test_rsi_oversold_range() -> None:
    # Monotonically falling prices → RSI should be low
    closes = [float(30 - i) for i in range(30)]
    result = _rsi(closes)
    assert result is not None
    assert result < 40.0


def test_sma_returns_none_when_insufficient_data() -> None:
    assert _sma([1.0, 2.0], 50) is None


def test_sma_correct_value() -> None:
    closes = [float(i) for i in range(1, 11)]  # 1..10
    result = _sma(closes, 5)
    assert result == 8.0  # mean of [6,7,8,9,10]


def test_macd_returns_none_fields_when_short_series() -> None:
    result = _macd([100.0] * 10)
    assert result["macd"] is None
    assert result["signal"] is None
    assert result["histogram"] is None


def test_macd_returns_values_with_sufficient_data() -> None:
    closes = [100.0 + i * 0.5 for i in range(50)]
    result = _macd(closes)
    assert result["macd"] is not None
    assert result["signal"] is not None
    assert result["histogram"] is not None


# ---------------------------------------------------------------------------
# EnrichedSnapshotProvider tests
# ---------------------------------------------------------------------------

def test_enriched_snapshot_provider_uses_override() -> None:
    base = MockIndianMarketDataProvider()
    today = date.today()
    override = StockSnapshot(
        symbol="RELIANCE",
        as_of=today,
        sector="Energy",
        close=2800.0,
        volume=1_000_000.0,
        delivery_ratio=0.5,
        pe_ratio=20.0,
        roe=0.18,
        debt_to_equity=0.4,
        earnings_growth=0.12,
        free_cash_flow_margin=0.1,
        promoter_holding_change=0.0,
        insider_activity_score=0.0,
    )
    patched = _EnrichedSnapshotProvider(base=base, overrides={"RELIANCE": override})
    snaps = patched.get_snapshots(["RELIANCE"])
    assert len(snaps) == 1
    assert snaps[0].pe_ratio == 20.0
    assert snaps[0].roe == 0.18


def test_enriched_snapshot_provider_falls_back_for_unknown_symbol() -> None:
    base = MockIndianMarketDataProvider()
    patched = _EnrichedSnapshotProvider(base=base, overrides={})
    snaps = patched.get_snapshots(["TCS"])
    assert len(snaps) == 1
    assert snaps[0].symbol == "TCS"


# ---------------------------------------------------------------------------
# SingleStockPipeline integration test (offline — uses mock providers)
# ---------------------------------------------------------------------------

def test_single_stock_pipeline_returns_full_report_structure() -> None:
    settings = load_settings()
    pipeline = SingleStockPipeline(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
        text_pipeline=None,
    )
    report = pipeline.run("RELIANCE")

    # Top-level identity fields
    assert report["symbol"] == "RELIANCE"
    assert "as_of" in report

    # Required sections
    assert "price" in report
    assert "technical_indicators" in report
    assert "fundamentals" in report
    assert "scores" in report
    assert "score_breakdown" in report
    assert "investment_horizons" in report
    assert "key_drivers" in report
    assert "risk_flags" in report
    assert "entry_exit" in report
    assert "news" in report
    assert "nlp_signals" in report
    assert "all_features" in report

    # News section structure
    news = report["news"]
    assert "headlines" in news
    assert isinstance(news["headlines"], list)
    assert "headline_count" in news

    # Score fields
    scores = report["scores"]
    assert isinstance(scores["long_term_score"], float)
    assert isinstance(scores["swing_score"], float)
    assert isinstance(scores["risk_penalty"], float)

    # Technical indicators have correct keys
    tech = report["technical_indicators"]
    assert "rsi_14" in tech
    assert "macd" in tech
    assert "adx_14" in tech
    assert "momentum_20d_pct" in tech

    # Three horizons present
    horizons = report["investment_horizons"]
    assert "swing" in horizons
    assert "medium_term" in horizons
    assert "long_term" in horizons
    for h in horizons.values():
        assert "verdict" in h
        assert "horizon" in h
        assert "rationale" in h
        assert "key_catalysts" in h
        assert "key_risks" in h

    # Score breakdown has four categories
    breakdown = report["score_breakdown"]
    assert "fundamental" in breakdown
    assert "technical" in breakdown
    assert "risk_penalties" in breakdown


def test_single_stock_pipeline_symbol_normalised_to_uppercase() -> None:
    settings = load_settings()
    pipeline = SingleStockPipeline(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )
    report = pipeline.run("reliance")
    assert report["symbol"] == "RELIANCE"
