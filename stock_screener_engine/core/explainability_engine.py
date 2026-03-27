"""Deterministic explainability built from scoring components."""

from __future__ import annotations

from dataclasses import dataclass

from stock_screener_engine.core.scoring_base import ScoringResult
from stock_screener_engine.core.scoring_risk import RiskPenaltyResult


@dataclass(frozen=True)
class DeterministicExplanation:
    positive_drivers: list[str]
    negative_drivers: list[str]
    missing_features: list[str]
    summary: str
    rejection_reasons: list[str]


class DeterministicExplainabilityEngine:
    def build(
        self,
        long_term: ScoringResult,
        swing: ScoringResult,
        risk: RiskPenaltyResult,
        signal_type: str,
        passed_filter: bool,
        min_score: float,
        final_score: float,
    ) -> DeterministicExplanation:
        if signal_type == "long_term":
            positive = sorted(long_term.component_map.items(), key=lambda kv: kv[1], reverse=True)
        else:
            positive = sorted(swing.component_map.items(), key=lambda kv: kv[1], reverse=True)

        negative = sorted(risk.components.items(), key=lambda kv: kv[1], reverse=True)

        pos = [_render_positive_driver(k, v) for k, v in positive[:3]]
        neg = [_render_negative_driver(k, v) for k, v in negative[:3]]

        missing = sorted(set(long_term.missing_features + swing.missing_features + risk.missing_features))
        summary = (
            f"{signal_type.replace('_', ' ')} final={final_score:.1f}; "
            f"top+={pos[0] if pos else 'none'}; top-={neg[0] if neg else 'none'}"
        )

        rejection_reasons = []
        if not passed_filter:
            rejection_reasons.append(f"final score {final_score:.1f} below threshold {min_score:.1f}")
        if missing:
            rejection_reasons.append(f"missing features: {', '.join(missing[:6])}")

        return DeterministicExplanation(
            positive_drivers=pos,
            negative_drivers=neg,
            missing_features=missing,
            summary=summary,
            rejection_reasons=rejection_reasons,
        )


def _render_positive_driver(name: str, value: float) -> str:
    if "event" in name and value > 0:
        return f"Positive driver: Recent high-confidence event catalyst ({value:.2f})"
    if "governance" in name and value > 0:
        return f"Positive driver: Governance and management commentary support ({value:.2f})"
    if "growth" in name and value > 0:
        return f"Positive driver: Earnings and growth context improved ({value:.2f})"
    return f"{name.replace('_', ' ')}: {value:.2f}"


def _render_negative_driver(name: str, value: float) -> str:
    if "event" in name or "litigation" in name:
        return f"Risk flag: Negative regulatory/litigation style event risk ({value:.2f})"
    if "text_uncertainty" in name:
        return f"Risk flag: Mixed commentary with high uncertainty ({value:.2f})"
    if "governance" in name:
        return f"Risk flag: Governance red flag penalty ({value:.2f})"
    return f"{name.replace('_', ' ')}: {value:.2f}"
