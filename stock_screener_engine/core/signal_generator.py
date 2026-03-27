"""Final signal generation combining long-term, swing, and risk engines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from stock_screener_engine.core.explainability_engine import DeterministicExplainabilityEngine
from stock_screener_engine.core.scoring_base import ScoringResult
from stock_screener_engine.core.scoring_risk import RiskPenaltyResult
from stock_screener_engine.core.signal_schemas import RankedSignal, SignalDrivers


@dataclass(frozen=True)
class SignalThresholds:
    min_long_term: float = 55.0
    min_swing: float = 58.0
    min_short: float = 58.0


class ResearchSignalGenerator:
    def __init__(self, thresholds: SignalThresholds) -> None:
        self.thresholds = thresholds
        self.explainer = DeterministicExplainabilityEngine()

    def build_long_term(
        self,
        symbol: str,
        as_of: date,
        long_term: ScoringResult,
        swing: ScoringResult,
        risk: RiskPenaltyResult,
        stock_name: str | None = None,
    ) -> RankedSignal:
        final_score = max(0.0, min(100.0, long_term.total_score - risk.total_penalty))
        passed = final_score >= self.thresholds.min_long_term
        exp = self.explainer.build(
            long_term=long_term,
            swing=swing,
            risk=risk,
            signal_type="long_term",
            passed_filter=passed,
            min_score=self.thresholds.min_long_term,
            final_score=final_score,
        )
        return RankedSignal(
            symbol=symbol,
            stock_name=stock_name,
            signal_category="long_term_candidate" if passed else "long_term_reject",
            as_of=as_of,
            long_term_score=long_term.total_score,
            swing_score=swing.total_score,
            risk_penalty=risk.total_penalty,
            final_score=final_score,
            rank=0,
            conviction=(final_score + swing.total_score) / 2.0,
            horizon="6-24 months",
            drivers=SignalDrivers(
                top_positive=exp.positive_drivers,
                top_negative=exp.negative_drivers,
                missing_features=exp.missing_features,
            ),
            rejection_reasons=exp.rejection_reasons,
        )

    def build_swing(
        self,
        symbol: str,
        as_of: date,
        long_term: ScoringResult,
        swing: ScoringResult,
        risk: RiskPenaltyResult,
        stock_name: str | None = None,
    ) -> RankedSignal:
        final_score = max(0.0, min(100.0, swing.total_score - risk.total_penalty))
        passed = final_score >= self.thresholds.min_swing
        exp = self.explainer.build(
            long_term=long_term,
            swing=swing,
            risk=risk,
            signal_type="swing",
            passed_filter=passed,
            min_score=self.thresholds.min_swing,
            final_score=final_score,
        )
        return RankedSignal(
            symbol=symbol,
            stock_name=stock_name,
            signal_category="swing_candidate" if passed else "swing_reject",
            as_of=as_of,
            long_term_score=long_term.total_score,
            swing_score=swing.total_score,
            risk_penalty=risk.total_penalty,
            final_score=final_score,
            rank=0,
            conviction=(final_score + long_term.total_score) / 2.0,
            horizon="3-15 trading days",
            drivers=SignalDrivers(
                top_positive=exp.positive_drivers,
                top_negative=exp.negative_drivers,
                missing_features=exp.missing_features,
            ),
            rejection_reasons=exp.rejection_reasons,
        )

    def build_short(
        self,
        symbol: str,
        as_of: date,
        short: ScoringResult,
        risk: RiskPenaltyResult,
        stock_name: str | None = None,
    ) -> RankedSignal:
        """Build a short/bearish signal from a short ScoringResult."""
        final_score = max(0.0, min(100.0, short.total_score))
        passed = final_score >= self.thresholds.min_short

        # Top bearish drivers are the highest-contributing short categories
        positive = sorted(short.component_map.items(), key=lambda kv: kv[1], reverse=True)
        neg_risk  = sorted(risk.components.items(),    key=lambda kv: kv[1], reverse=True)

        pos_drivers = [
            f"Bearish driver: {k.replace('_', ' ')} ({v:.2f})" for k, v in positive[:3]
        ]
        neg_drivers = [
            f"Risk amplifier: {k.replace('_', ' ')} ({v:.2f})" for k, v in neg_risk[:2]
        ]
        missing = sorted(set(short.missing_features + risk.missing_features))

        rejection_reasons: list[str] = []
        if not passed:
            rejection_reasons.append(
                f"short score {final_score:.1f} below threshold {self.thresholds.min_short:.1f}"
            )
        if missing:
            rejection_reasons.append(f"missing features: {', '.join(missing[:6])}")

        return RankedSignal(
            symbol=symbol,
            stock_name=stock_name,
            signal_category="short_candidate" if passed else "short_reject",
            as_of=as_of,
            long_term_score=0.0,
            swing_score=0.0,
            risk_penalty=risk.total_penalty,
            final_score=final_score,
            rank=0,
            conviction=final_score,
            horizon="3-20 trading days",
            drivers=SignalDrivers(
                top_positive=pos_drivers,
                top_negative=neg_drivers,
                missing_features=missing,
            ),
            rejection_reasons=rejection_reasons,
        )


def assign_ranks(signals: list[RankedSignal]) -> list[RankedSignal]:
    ordered = sorted(signals, key=lambda s: (s.final_score, s.conviction, s.symbol), reverse=True)
    return [replace(signal, rank=idx + 1) for idx, signal in enumerate(ordered)]
