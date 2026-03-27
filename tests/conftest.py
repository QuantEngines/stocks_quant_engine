"""Shared pytest fixtures for the stock screener engine test suite."""

from __future__ import annotations

import pytest
from datetime import date

from stock_screener_engine.config.settings import AppSettings, load_settings
from stock_screener_engine.core.entities import (
    FeatureVector,
    FundamentalsSnapshot,
    GovernanceSnapshot,
    MarketSnapshot,
    ScoreCard,
    StockSnapshot,
)


@pytest.fixture()
def default_settings() -> AppSettings:
    return load_settings()


@pytest.fixture()
def sample_feature_vector() -> FeatureVector:
    return FeatureVector(
        symbol="TEST",
        as_of=date(2024, 1, 15),
        values={
            "growth_quality": 0.75,
            "profitability_quality": 0.70,
            "balance_sheet_health": 0.85,
            "cash_flow_quality": 0.65,
            "valuation_sanity": 0.60,
            "governance_proxy": 0.80,
            "revenue_growth": 0.55,
            "operating_margin": 0.45,
            "trend_strength": 0.70,
            "momentum_strength": 0.65,
            "relative_strength_proxy": 0.60,
            "volatility_regime": 0.50,
            "volume_confirmation": 0.75,
            "delivery_ratio": 0.70,
            "event_catalyst": 0.30,
            "governance_event": 0.40,
            "sentiment_score": 0.25,
            "news_sentiment": 0.20,
            "market_regime_score": 0.30,
            "sector_momentum": 0.20,
        },
    )


@pytest.fixture()
def zero_feature_vector() -> FeatureVector:
    """All features at zero — useful for testing floor/rejection behaviour."""
    return FeatureVector(
        symbol="ZERO",
        as_of=date(2024, 1, 15),
        values={k: 0.0 for k in [
            "growth_quality", "profitability_quality", "balance_sheet_health",
            "cash_flow_quality", "valuation_sanity", "governance_proxy",
            "revenue_growth", "operating_margin", "trend_strength",
            "momentum_strength", "relative_strength_proxy", "volatility_regime",
            "volume_confirmation", "delivery_ratio", "event_catalyst",
            "governance_event", "sentiment_score", "news_sentiment",
            "market_regime_score", "sector_momentum",
        ]},
    )


@pytest.fixture()
def sample_stock_snapshot() -> StockSnapshot:
    return StockSnapshot(
        symbol="TESTSYM",
        as_of=date(2024, 1, 15),
        sector="IT",
        close=1500.0,
        volume=2_500_000,
        delivery_ratio=0.65,
        pe_ratio=22.0,
        roe=0.22,
        debt_to_equity=0.35,
        earnings_growth=0.18,
        free_cash_flow_margin=0.15,
        promoter_holding_change=0.02,
        insider_activity_score=0.55,
    )


@pytest.fixture()
def sample_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="TESTSYM",
        as_of=date(2024, 1, 15),
        sector="IT",
        exchange="NSE",
        close=1500.0,
        open_price=1480.0,
        high=1520.0,
        low=1470.0,
        volume=2_500_000.0,
        delivery_ratio=0.65,
        market_cap=15_000_000_000.0,
    )


@pytest.fixture()
def sample_fundamentals() -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        symbol="TESTSYM",
        as_of=date(2024, 1, 15),
        pe_ratio=22.0,
        pb_ratio=3.5,
        roe=0.22,
        roa=0.12,
        roce=0.20,
        debt_to_equity=0.35,
        current_ratio=2.1,
        interest_coverage=10.0,
        earnings_growth_yoy=0.18,
        revenue_growth_yoy=0.14,
        free_cash_flow_margin=0.15,
        operating_margin=0.20,
        net_profit_margin=0.14,
    )


@pytest.fixture()
def sample_governance() -> GovernanceSnapshot:
    return GovernanceSnapshot(
        symbol="TESTSYM",
        as_of=date(2024, 1, 15),
        promoter_holding_pct=55.0,
        promoter_holding_change_qoq=0.02,
        institutional_holding_pct=30.0,
        fii_holding_pct=12.0,
        dii_holding_pct=18.0,
        insider_activity_score=0.55,
        audit_opinion="clean",
    )


@pytest.fixture()
def sample_score_card(sample_feature_vector: FeatureVector) -> ScoreCard:
    from stock_screener_engine.core.scoring import (
        LongTermScorer, SwingScorer, RiskPenaltyScorer, build_score_card,
    )
    return build_score_card(
        sample_feature_vector,
        LongTermScorer(),
        SwingScorer(),
        RiskPenaltyScorer(),
    )
