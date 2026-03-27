from __future__ import annotations

from stock_screener_engine.core.scoring_long_term import LongTermInvestmentScorer
from stock_screener_engine.core.scoring_risk import RiskPenaltyEngine
from stock_screener_engine.core.scoring_swing import SwingTradeScorer


def test_long_term_scorer_handles_missing_features() -> None:
    scorer = LongTermInvestmentScorer()
    result = scorer.score({"growth_quality": 0.7})
    assert 0.0 <= result.total_score <= 100.0
    assert result.missing_features


def test_swing_scorer_outputs_component_map() -> None:
    scorer = SwingTradeScorer()
    result = scorer.score(
        {
            "trend_strength": 0.8,
            "momentum_strength": 0.7,
            "volume_confirmation": 0.6,
            "volatility_regime": 0.5,
            "event_catalyst": 0.2,
            "sentiment_score": 0.1,
        }
    )
    assert result.component_map
    assert 0.0 <= result.total_score <= 100.0


def test_risk_penalty_engine_components() -> None:
    engine = RiskPenaltyEngine(max_penalty=30.0)
    result = engine.score(
        {
            "volume_confirmation": 0.2,
            "volatility_regime": 0.1,
            "balance_sheet_health": 0.3,
            "event_catalyst": -0.4,
            "governance_proxy": 0.2,
        }
    )
    assert result.components
    assert result.total_penalty > 0
    assert result.flags
