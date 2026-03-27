from __future__ import annotations

from datetime import date

from stock_screener_engine.core.explainability_engine import DeterministicExplainabilityEngine
from stock_screener_engine.core.scoring_base import ScoringResult
from stock_screener_engine.core.scoring_risk import RiskPenaltyResult
from stock_screener_engine.core.scoring import LongTermScorer, RiskPenaltyScorer, SwingScorer, build_score_card
from stock_screener_engine.core.entities import FeatureVector


def test_text_features_can_shift_scores() -> None:
    base = FeatureVector(
        symbol="ABC",
        as_of=date.today(),
        values={
            "growth_quality": 0.6,
            "profitability_quality": 0.6,
            "balance_sheet_health": 0.7,
            "cash_flow_quality": 0.6,
            "valuation_sanity": 0.6,
            "governance_proxy": 0.6,
            "trend_strength": 0.6,
            "momentum_strength": 0.6,
            "relative_strength_proxy": 0.6,
            "volatility_regime": 0.5,
            "volume_confirmation": 0.6,
            "event_catalyst": 0.0,
            "sentiment_score": 0.0,
            "market_regime_score": 0.1,
        },
    )
    enriched = FeatureVector(
        symbol="ABC",
        as_of=date.today(),
        values={
            **base.values,
            "management_tone_score": 0.8,
            "catalyst_strength_score": 0.9,
            "decayed_event_signal": 0.7,
            "earnings_sentiment_score": 0.6,
            "governance_flag_score": 0.7,
            "event_momentum_score": 0.8,
            "catalyst_presence_flag": 1.0,
        },
    )
    base_card = build_score_card(base, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())
    enriched_card = build_score_card(enriched, LongTermScorer(), SwingScorer(), RiskPenaltyScorer())
    assert enriched_card.long_term_score >= base_card.long_term_score
    assert enriched_card.swing_score >= base_card.swing_score


def test_explainability_contains_nlp_driver_language() -> None:
    exp = DeterministicExplainabilityEngine().build(
        long_term=ScoringResult(total_score=70.0, categories=[], component_map={"event_context": 12.0}, missing_features=[]),
        swing=ScoringResult(total_score=50.0, categories=[], component_map={}, missing_features=[]),
        risk=RiskPenaltyResult(
            total_penalty=6.0,
            components={"text_uncertainty_risk": 4.0},
            flags=[],
            missing_features=[],
        ),
        signal_type="long_term",
        passed_filter=True,
        min_score=10.0,
        final_score=65.0,
    )
    assert any("Positive driver:" in s for s in exp.positive_drivers)
    assert any("Risk flag:" in s for s in exp.negative_drivers)
