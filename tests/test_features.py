"""Tests for FeatureEngine — covers both granular and unified snapshot paths."""

from __future__ import annotations

import pytest
from datetime import date

from stock_screener_engine.core.entities import (
    FundamentalsSnapshot,
    GovernanceSnapshot,
    MarketSnapshot,
    StockSnapshot,
)
from stock_screener_engine.core.feature_specs import (
    FUNDAMENTAL_FEATURES,
    TECHNICAL_FEATURES,
)
from stock_screener_engine.core.features import FeatureEngine


@pytest.fixture()
def engine() -> FeatureEngine:
    return FeatureEngine(
        include_sentiment=True,
        include_event_signals=True,
        include_regime_features=True,
    )


class TestComputeFromGranularSnapshots:
    def test_returns_feature_vector_with_symbol(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
        sample_fundamentals: FundamentalsSnapshot,
        sample_governance: GovernanceSnapshot,
    ) -> None:
        fv = engine.compute(
            market=sample_market_snapshot,
            fundamentals=sample_fundamentals,
            governance=sample_governance,
        )
        assert fv.symbol == sample_market_snapshot.symbol

    def test_all_fundamental_features_present(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
        sample_fundamentals: FundamentalsSnapshot,
    ) -> None:
        fv = engine.compute(market=sample_market_snapshot, fundamentals=sample_fundamentals)
        missing = FUNDAMENTAL_FEATURES - set(fv.values.keys())
        assert not missing, f"Missing fundamental features: {missing}"

    def test_all_technical_features_present(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
    ) -> None:
        fv = engine.compute(market=sample_market_snapshot)
        missing = TECHNICAL_FEATURES - set(fv.values.keys())
        assert not missing, f"Missing technical features: {missing}"

    def test_feature_values_within_expected_range(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
        sample_fundamentals: FundamentalsSnapshot,
        sample_governance: GovernanceSnapshot,
    ) -> None:
        fv = engine.compute(
            market=sample_market_snapshot,
            fundamentals=sample_fundamentals,
            governance=sample_governance,
            sentiment_score=0.5,
            event_signal=0.3,
            market_regime_score=0.4,
        )
        # Raw ratio features are intentionally outside the standard [0, 1] range;
        # they are emitted as-is so scorers can apply their own normalisation.
        _raw_ratio_features = {"pe_ratio", "pb_ratio", "debt_to_equity", "cfo_pat_ratio"}
        for key, value in fv.values.items():
            if key in _raw_ratio_features:
                assert value >= 0.0, f"Raw ratio feature '{key}' = {value} should be non-negative"
            else:
                assert -1.0 <= value <= 1.5, f"Feature '{key}' = {value} is out of expected range"

    def test_none_fundamentals_returns_zero_filled_features(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
    ) -> None:
        fv = engine.compute(market=sample_market_snapshot, fundamentals=None, governance=None)
        for key in FUNDAMENTAL_FEATURES:
            assert fv.values.get(key) == 0.0, f"Expected 0.0 for {key} with no fundamentals"

    def test_none_governance_returns_zero_governance_proxy(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
    ) -> None:
        fv = engine.compute(market=sample_market_snapshot, governance=None)
        assert fv.values.get("governance_proxy") == 0.0

    def test_high_debt_reduces_balance_sheet_health(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
    ) -> None:
        low_debt = FundamentalsSnapshot(
            symbol="X", as_of=date(2024, 1, 1), debt_to_equity=0.1
        )
        high_debt = FundamentalsSnapshot(
            symbol="X", as_of=date(2024, 1, 1), debt_to_equity=3.0
        )
        fv_low = engine.compute(market=sample_market_snapshot, fundamentals=low_debt)
        fv_high = engine.compute(market=sample_market_snapshot, fundamentals=high_debt)
        assert fv_low.values["balance_sheet_health"] > fv_high.values["balance_sheet_health"]

    def test_explicit_stability_and_leverage_features_present(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
        sample_fundamentals: FundamentalsSnapshot,
    ) -> None:
        fv = engine.compute(market=sample_market_snapshot, fundamentals=sample_fundamentals)
        assert "earnings_stability" in fv.values
        assert "leverage_trend" in fv.values
        assert 0.0 <= fv.values["earnings_stability"] <= 1.0
        assert 0.0 <= fv.values["leverage_trend"] <= 1.0

    def test_valuation_context_updates_valuation_features(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
        sample_fundamentals: FundamentalsSnapshot,
    ) -> None:
        base = engine.compute(market=sample_market_snapshot, fundamentals=sample_fundamentals)
        stressed = engine.compute(
            market=sample_market_snapshot,
            fundamentals=sample_fundamentals,
            valuation_context={
                sample_market_snapshot.symbol: {
                    "sector_pe_zscore": 2.5,
                    "sector_pb_zscore": 2.5,
                    "rolling_pe_zscore": 2.0,
                    "rolling_pb_zscore": 2.0,
                }
            },
        )
        assert stressed.values["valuation_sanity"] < base.values["valuation_sanity"]
        assert stressed.values["sector_pe_zscore"] == 2.5
        assert stressed.values["rolling_pb_zscore"] == 2.0

    def test_qualified_audit_reduces_governance_proxy(
        self,
        engine: FeatureEngine,
        sample_market_snapshot: MarketSnapshot,
    ) -> None:
        clean = GovernanceSnapshot(symbol="X", as_of=date(2024, 1, 1), audit_opinion="clean")
        qualified = GovernanceSnapshot(
            symbol="X", as_of=date(2024, 1, 1), audit_opinion="qualified"
        )
        fv_clean = engine.compute(market=sample_market_snapshot, governance=clean)
        fv_qual = engine.compute(market=sample_market_snapshot, governance=qualified)
        assert fv_clean.values["governance_proxy"] > fv_qual.values["governance_proxy"]


class TestComputeFromSnapshot:
    def test_backward_compat_returns_same_symbol(
        self, engine: FeatureEngine, sample_stock_snapshot: StockSnapshot
    ) -> None:
        fv = engine.compute_from_snapshot(sample_stock_snapshot)
        assert fv.symbol == sample_stock_snapshot.symbol

    def test_backward_compat_covers_fundamental_features(
        self, engine: FeatureEngine, sample_stock_snapshot: StockSnapshot
    ) -> None:
        fv = engine.compute_from_snapshot(sample_stock_snapshot)
        missing = FUNDAMENTAL_FEATURES - set(fv.values.keys())
        assert not missing

    def test_sentiment_flag_off_excludes_sentiment_keys(
        self, sample_stock_snapshot: StockSnapshot
    ) -> None:
        engine_no_sentiment = FeatureEngine(
            include_sentiment=False,
            include_event_signals=True,
            include_regime_features=True,
        )
        fv = engine_no_sentiment.compute_from_snapshot(sample_stock_snapshot)
        assert "sentiment_score" not in fv.values
        assert "news_sentiment" not in fv.values
