from __future__ import annotations

from datetime import date

from stock_screener_engine.core.scoring_base import ScoringResult
from stock_screener_engine.core.scoring_risk import RiskPenaltyResult
from stock_screener_engine.core.signal_generator import ResearchSignalGenerator, SignalThresholds, assign_ranks


def _base_long() -> ScoringResult:
    return ScoringResult(
        total_score=72.0,
        categories=[],
        component_map={"growth_quality": 12.0, "profitability_quality": 11.0},
        missing_features=["revenue_growth"],
    )


def _base_swing() -> ScoringResult:
    return ScoringResult(
        total_score=68.0,
        categories=[],
        component_map={"trend": 10.0, "momentum": 9.0},
        missing_features=[],
    )


def _base_risk() -> RiskPenaltyResult:
    return RiskPenaltyResult(
        total_penalty=12.0,
        components={"liquidity_risk": 3.0, "volatility_risk": 2.0},
        flags=["liquidity_risk"],
        missing_features=[],
    )


def test_signal_generation_and_ranking() -> None:
    gen = ResearchSignalGenerator(SignalThresholds(min_long_term=50.0, min_swing=50.0))

    s1 = gen.build_long_term("AAA", date.today(), _base_long(), _base_swing(), _base_risk())
    s2 = gen.build_long_term(
        "BBB",
        date.today(),
        ScoringResult(82.0, [], {"growth_quality": 13.0}, []),
        _base_swing(),
        _base_risk(),
    )

    ranked = assign_ranks([s1, s2])
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[0].final_score >= ranked[1].final_score


def test_rejection_reason_when_below_threshold() -> None:
    gen = ResearchSignalGenerator(SignalThresholds(min_long_term=90.0, min_swing=90.0))
    s = gen.build_long_term("AAA", date.today(), _base_long(), _base_swing(), _base_risk())
    assert s.signal_category.endswith("reject")
    assert s.rejection_reasons
