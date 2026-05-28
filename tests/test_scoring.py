from __future__ import annotations

from datetime import date

from stock_screener_engine.core.entities import FeatureVector
from stock_screener_engine.core.scoring import (
    LongTermScorer,
    LongTermWeights,
    RegimeSwitchConfig,
    RiskPenaltyScorer,
    SwingScorer,
    SwingWeights,
    build_score_card,
)
from stock_screener_engine.core.scoring_conviction import ConvictionScorer


def _rich_fv(symbol: str = "TEST") -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        as_of=date.today(),
        values={
            "growth_quality": 0.8,
            "profitability_quality": 0.7,
            "balance_sheet_health": 0.9,
            "cash_flow_quality": 0.75,
            "valuation_sanity": 0.6,
            "governance_proxy": 0.8,
            "trend_strength": 0.65,
            "momentum_strength": 0.7,
            "relative_strength_proxy": 0.6,
            "volatility_regime": 0.55,
            "volume_confirmation": 0.7,
            "event_catalyst": 0.4,
            "sentiment_score": 0.3,
            "market_regime_score": 0.2,
        },
    )


def test_score_card_ranges() -> None:
    fv = _rich_fv()
    card = build_score_card(fv, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())
    assert 0.0 <= card.long_term_score <= 100.0
    assert 0.0 <= card.swing_score <= 100.0
    assert 0.0 <= card.risk_penalty <= 30.0
    assert 0.0 <= card.conviction <= 100.0
    assert "conviction_score_strength" in card.component_scores
    assert "conviction_data_completeness" in card.component_scores
    assert "risk_earnings_instability_risk" in card.component_scores


def test_zero_features_produce_zero_score() -> None:
    zero_fv = FeatureVector(
        symbol="ZERO",
        as_of=date.today(),
        values={k: 0.0 for k in [
            "growth_quality", "profitability_quality", "balance_sheet_health",
            "cash_flow_quality", "valuation_sanity", "governance_proxy",
            "trend_strength", "momentum_strength", "relative_strength_proxy",
            "volatility_regime", "volume_confirmation", "event_catalyst",
            "sentiment_score", "market_regime_score",
        ]},
    )
    card = build_score_card(zero_fv, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())
    assert card.long_term_score == 0.0
    assert card.swing_score == 0.0


def test_configurable_weights_change_score() -> None:
    """Doubling growth_quality weight should increase long-term score for a stock with high growth."""
    fv = _rich_fv()
    default_card = build_score_card(fv, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())

    heavy_growth_weights = LongTermWeights(
        growth_quality=36.0,  # double
        profitability_quality=17.0,
        balance_sheet_health=15.0,
        cash_flow_quality=12.0,
        valuation_sanity=12.0,
        governance_proxy=10.0,
        event_catalyst=8.0,
        regime_tailwind=8.0,
    )
    heavy_card = build_score_card(
        fv, LongTermScorer(weights=heavy_growth_weights), SwingScorer(), RiskPenaltyScorer()
    )
    # growth_quality = 0.8 means heavy-weight scorer should produce higher long-term score
    assert heavy_card.long_term_score > default_card.long_term_score


def test_configurable_swing_weights_change_score() -> None:
    """Zero-weighting event_catalyst should reduce swing score when catalyst is positive."""
    fv = _rich_fv()
    default_card = build_score_card(fv, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())

    zero_event_weights = SwingWeights(
        trend_strength=20.0,
        momentum_strength=18.0,
        relative_strength_proxy=14.0,
        volatility_regime=12.0,
        volume_confirmation=12.0,
        event_catalyst=0.0,  # zeroed out
        sentiment_score=12.0,
    )
    no_catalyst_card = build_score_card(
        fv, LongTermScorer(), SwingScorer(weights=zero_event_weights), RiskPenaltyScorer()
    )
    # With event_catalyst=0.4 and positive contribution, zeroing weight should reduce
    assert no_catalyst_card.swing_score < default_card.swing_score


def test_conviction_evidence_can_discount_same_score_strength() -> None:
    fv = _rich_fv()
    weak_evidence = FeatureVector(
        symbol=fv.symbol,
        as_of=fv.as_of,
        values={**fv.values, "backtest_hit_rate": 0.2, "source_confidence": 0.3},
    )
    strong_evidence = FeatureVector(
        symbol=fv.symbol,
        as_of=fv.as_of,
        values={**fv.values, "backtest_hit_rate": 0.8, "source_confidence": 0.9},
    )

    scorer = ConvictionScorer()
    weak = build_score_card(weak_evidence, LongTermScorer(), SwingScorer(), RiskPenaltyScorer(), scorer)
    strong = build_score_card(strong_evidence, LongTermScorer(), SwingScorer(), RiskPenaltyScorer(), scorer)

    assert strong.component_scores["conviction_backtest_evidence"] > weak.component_scores["conviction_backtest_evidence"]
    assert strong.component_scores["conviction_source_confidence"] > weak.component_scores["conviction_source_confidence"]
    assert strong.conviction > weak.conviction


def test_conviction_penalizes_cross_horizon_disagreement() -> None:
    scorer = ConvictionScorer()

    aligned = scorer.score(
        {},
        adjusted_long_score=70.0,
        adjusted_swing_score=70.0,
        risk_penalty=5.0,
    )
    conflicted = scorer.score(
        {},
        adjusted_long_score=95.0,
        adjusted_swing_score=45.0,
        risk_penalty=5.0,
    )

    assert aligned.components["score_strength"] == conflicted.components["score_strength"]
    assert aligned.components["signal_agreement"] > conflicted.components["signal_agreement"]
    assert aligned.score > conflicted.score


def test_swing_catalyst_component_is_split_not_double_counted() -> None:
    fv = _rich_fv()
    score, components = SwingScorer().score(fv)

    assert 0.0 <= score <= 100.0
    assert components["event_catalyst"] > 0.0
    assert components["sentiment_score"] > 0.0
    assert sum(components.values()) <= 100.0


def test_cross_sectional_context_lifts_research_scores() -> None:
    base = _rich_fv()
    enriched = FeatureVector(
        symbol=base.symbol,
        as_of=base.as_of,
        values={
            **base.values,
            "cross_sectional_momentum_rank": 1.0,
            "sector_relative_momentum_rank": 1.0,
            "cross_sectional_quality_rank": 1.0,
            "sector_relative_quality_rank": 1.0,
            "cross_sectional_value_rank": 1.0,
            "quality_value_composite": 1.0,
            "liquidity_percentile": 1.0,
            "feature_coverage_score": 1.0,
        },
    )

    base_card = build_score_card(base, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())
    enriched_card = build_score_card(enriched, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())

    assert enriched_card.long_term_score > base_card.long_term_score
    assert enriched_card.swing_score > base_card.swing_score
    assert enriched_card.component_scores["conviction_data_completeness"] >= base_card.component_scores["conviction_data_completeness"]


def test_from_dict_weights() -> None:
    weights = LongTermWeights.from_dict({"growth_quality": 25.0})
    assert weights.growth_quality == 25.0
    assert weights.profitability_quality == LongTermWeights().profitability_quality


def test_risk_penalty_flags_high_risk() -> None:
    """A stock with zero volume and adverse governance should trigger multiple risk flags."""
    risky_fv = FeatureVector(
        symbol="RISKY",
        as_of=date.today(),
        values={
            "volume_confirmation": 0.0,
            "event_catalyst": -0.5,
            "balance_sheet_health": 0.0,
            "governance_proxy": 0.0,
            "volatility_regime": 0.0,
        },
    )
    scorer = RiskPenaltyScorer(max_penalty=30.0)
    penalty, _, flags = scorer.score(risky_fv)
    assert penalty > 20.0
    assert len(flags) >= 3


def test_regime_switching_applies_weight_profile() -> None:
    fv = _rich_fv()
    fv = FeatureVector(symbol=fv.symbol, as_of=fv.as_of, values={**fv.values, "market_regime_score": 0.8})

    base = LongTermScorer()
    profile = LongTermScorer(
        regime_switch=RegimeSwitchConfig(enabled=True, bull_threshold=0.2, bear_threshold=-0.2),
        regime_profiles={"bull": {"growth_quality": 30.0, "valuation_sanity": 6.0}},
    )
    base_score, _ = base.score(fv)
    tuned_score, _ = profile.score(fv)
    assert tuned_score != base_score
