"""Explainability engine for robust signal interpretation."""

from __future__ import annotations

from typing import Iterable

from stock_screener_engine.core.entities import ScoreCard, SignalExplanation


def _pretty_name(key: str) -> str:
    """Convert a snake_case component key to a readable label.

    Examples::
        "long_growth_quality"  -> "growth quality"
        "risk_leverage_risk"   -> "leverage risk"
    """
    for prefix in ("long_", "swing_", "risk_"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key.replace("_", " ")


def _top_components(
    component_scores: dict[str, float],
    prefix: str,
    top_n: int = 3,
    descending: bool = True,
) -> list[tuple[str, float]]:
    """Return the top ``top_n`` components with the given prefix, sorted by value."""
    filtered = [(k, v) for k, v in component_scores.items() if k.startswith(prefix)]
    return sorted(filtered, key=lambda x: x[1], reverse=descending)[:top_n]


class ExplanationEngine:
    def build(
        self,
        score_card: ScoreCard,
        signal_type: str,
        passed_filter: bool,
        min_score: float,
        risk_flags: Iterable[str],
    ) -> SignalExplanation:
        signal_score = score_card.long_term_score if signal_type == "long_term" else score_card.swing_score
        positive_prefix = "long_" if signal_type == "long_term" else "swing_"
        components = dict(score_card.component_scores)

        top_pos = _top_components(components, positive_prefix, top_n=3, descending=True)
        top_neg = _top_components(components, "risk_", top_n=3, descending=True)

        top_positive_drivers = [
            f"{_pretty_name(name)}: {value:.2f}" for name, value in top_pos
        ]
        top_negative_drivers = [
            f"{_pretty_name(name)}: {value:.2f}" for name, value in top_neg
        ]

        ranking_reason = (
            f"Ranked by {signal_type.replace('_', ' ')} score={signal_score:.1f}, "
            f"conviction={score_card.conviction:.1f}"
        )
        rejection_reason = (
            None
            if passed_filter
            else f"Score {signal_score:.1f} below threshold {min_score:.1f}"
        )

        horizon = "6–24 months" if signal_type == "long_term" else "3–15 trading days"

        return SignalExplanation(
            signal_type=signal_type,
            score=signal_score,
            top_positive_drivers=top_positive_drivers,
            top_negative_drivers=top_negative_drivers,
            ranking_reason=ranking_reason,
            rejection_reason=rejection_reason,
            holding_horizon=horizon,
            risk_flags=list(risk_flags),
            confidence=score_card.conviction,
            entry_logic=(
                "Enter on pullback to support with volume confirmation"
                if signal_type == "swing"
                else "Staggered accumulation on valuation comfort"
            ),
            invalidation_logic=(
                "Exit if trend breaks and catalyst fades"
                if signal_type == "swing"
                else "Review thesis if quality and cash flow degrade"
            ),
        )
