"""Standalone demo for the modular scoring and signal framework."""

from __future__ import annotations

from datetime import date

from stock_screener_engine.core.scoring_long_term import LongTermInvestmentScorer
from stock_screener_engine.core.scoring_risk import RiskPenaltyEngine
from stock_screener_engine.core.scoring_swing import SwingTradeScorer
from stock_screener_engine.core.signal_generator import ResearchSignalGenerator, SignalThresholds, assign_ranks


def run_demo() -> None:
    universe = {
        "RELIANCE": {
            "growth_quality": 0.72,
            "revenue_growth": 0.16,
            "profitability_quality": 0.65,
            "operating_margin": 0.18,
            "balance_sheet_health": 0.7,
            "cash_flow_quality": 0.66,
            "valuation_sanity": 0.58,
            "governance_proxy": 0.74,
            "event_catalyst": 0.15,
            "market_regime_score": 0.2,
            "trend_strength": 0.7,
            "momentum_strength": 0.62,
            "volume_confirmation": 0.68,
            "volatility_regime": 0.55,
            "relative_strength_proxy": 0.6,
            "sentiment_score": 0.1,
        },
        "TCS": {
            "growth_quality": 0.65,
            "revenue_growth": 0.12,
            "profitability_quality": 0.78,
            "operating_margin": 0.24,
            "balance_sheet_health": 0.85,
            "cash_flow_quality": 0.75,
            "valuation_sanity": 0.45,
            "governance_proxy": 0.82,
            "event_catalyst": 0.05,
            "market_regime_score": 0.2,
            "trend_strength": 0.55,
            "momentum_strength": 0.51,
            "volume_confirmation": 0.48,
            "volatility_regime": 0.6,
            "relative_strength_proxy": 0.58,
            "sentiment_score": 0.05,
        },
        "SMALLCAPX": {
            "growth_quality": 0.8,
            "profitability_quality": 0.42,
            "balance_sheet_health": 0.28,
            "cash_flow_quality": 0.25,
            "valuation_sanity": 0.7,
            "governance_proxy": 0.35,
            "event_catalyst": -0.35,
            "trend_strength": 0.82,
            "momentum_strength": 0.85,
            "volume_confirmation": 0.2,
            "volatility_regime": 0.1,
            "relative_strength_proxy": 0.75,
            "sentiment_score": -0.2,
        },
    }

    long_scorer = LongTermInvestmentScorer()
    swing_scorer = SwingTradeScorer()
    risk_engine = RiskPenaltyEngine(max_penalty=30.0)
    signal_gen = ResearchSignalGenerator(SignalThresholds(min_long_term=50.0, min_swing=52.0))

    long_signals = []
    swing_signals = []

    for symbol, feats in universe.items():
        lt = long_scorer.score(feats)
        sw = swing_scorer.score(feats)
        rk = risk_engine.score(feats)

        long_signals.append(
            signal_gen.build_long_term(
                symbol=symbol,
                as_of=date.today(),
                long_term=lt,
                swing=sw,
                risk=rk,
                stock_name=symbol,
            )
        )
        swing_signals.append(
            signal_gen.build_swing(
                symbol=symbol,
                as_of=date.today(),
                long_term=lt,
                swing=sw,
                risk=rk,
                stock_name=symbol,
            )
        )

    long_ranked = assign_ranks(long_signals)
    swing_ranked = assign_ranks(swing_signals)

    print("LONG-TERM CANDIDATES")
    for s in long_ranked:
        print(
            {
                "rank": s.rank,
                "symbol": s.symbol,
                "category": s.signal_category,
                "long_term": round(s.long_term_score, 2),
                "risk_penalty": round(s.risk_penalty, 2),
                "final": round(s.final_score, 2),
                "top_positive": s.drivers.top_positive,
                "top_negative": s.drivers.top_negative,
                "reasons": s.rejection_reasons,
            }
        )

    print("\nSWING CANDIDATES")
    for s in swing_ranked:
        print(
            {
                "rank": s.rank,
                "symbol": s.symbol,
                "category": s.signal_category,
                "swing": round(s.swing_score, 2),
                "risk_penalty": round(s.risk_penalty, 2),
                "final": round(s.final_score, 2),
                "top_positive": s.drivers.top_positive,
                "top_negative": s.drivers.top_negative,
                "reasons": s.rejection_reasons,
            }
        )


if __name__ == "__main__":
    run_demo()
