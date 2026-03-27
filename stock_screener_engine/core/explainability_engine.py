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


# Human-readable labels for scoring categories
_CATEGORY_LABELS: dict[str, str] = {
    "growth_quality":            "Earnings & revenue growth trajectory",
    "profitability_quality":     "Profitability and return-on-equity strength",
    "balance_sheet_strength":    "Balance-sheet health and leverage control",
    "cash_flow_quality":         "Free-cash-flow quality vs accounting earnings",
    "valuation_sanity":          "Relative and absolute valuation attractiveness",
    "governance_risk_proxy":     "Promoter holding, insider activity and audit quality",
    "event_context":             "Recent catalyst and event-driven signals",
    "regime_context":            "Macro regime and sector momentum tailwind",
    "trend":                     "Price trend strength and relative outperformance",
    "momentum":                  "Price momentum and recent acceleration",
    "volume_participation":      "Volume confirmation of price moves",
    "volatility_regime":         "Volatility environment (lower = better setup)",
    "setup_quality":             "Technical setup — consolidation before breakout",
    "catalyst_awareness":        "Near-term catalyst and sentiment support",
}

_RISK_LABELS: dict[str, str] = {
    "liquidity_risk":            "Low liquidity — insufficient trading depth",
    "volatility_risk":           "Elevated price volatility",
    "leverage_risk":             "High debt or deteriorating balance-sheet",
    "earnings_instability_risk": "Inconsistent or declining earnings quality",
    "event_uncertainty_risk":    "Negative or uncertain pending event risk",
    "governance_risk":           "Governance red flag or insider selling",
    "text_uncertainty_risk":     "Ambiguous management commentary or low transcript quality",
}


class DeterministicExplainabilityEngine:
    """Build human-readable explanations from structured scoring results."""

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
        primary = long_term if signal_type == "long_term" else swing
        secondary = swing if signal_type == "long_term" else long_term

        # --- positive drivers from primary scorer categories
        cat_sorted = sorted(
            primary.categories,
            key=lambda c: c.contribution,
            reverse=True,
        ) if primary.categories else []

        pos = [
            _make_positive(c.name, c.score_0_1, c.contribution)
            for c in cat_sorted[:3]
            if c.contribution > 0.01
        ]

        # Supplement with component_map if categories unavailable
        if not pos:
            top_comps = sorted(
                primary.component_map.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:3]
            pos = [_make_positive(k, v, v) for k, v in top_comps if v > 0.01]

        # --- negative drivers from risk components
        risk_sorted = sorted(
            risk.components.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        neg = [
            _make_negative(k, v)
            for k, v in risk_sorted[:3]
            if v > 0.5
        ]

        # --- secondary scorer weakness as additional negative
        if secondary.categories:
            weakest = min(secondary.categories, key=lambda c: c.contribution)
            if weakest.contribution < 0.03 and len(neg) < 3:
                neg.append(
                    f"Weak {_label(weakest.name)}: score {weakest.score_0_1:.2f} "
                    f"(acts as secondary headwind)"
                )

        missing = sorted(
            set(long_term.missing_features + swing.missing_features + risk.missing_features)
        )

        summary = (
            f"{signal_type.replace('_', ' ').title()} — "
            f"final score {final_score:.1f} "
            f"({'PASS' if passed_filter else 'FAIL'} vs threshold {min_score:.1f}); "
            f"top driver: {pos[0] if pos else 'none'}; "
            f"top risk: {neg[0] if neg else 'none'}"
        )

        rejection_reasons: list[str] = []
        if not passed_filter:
            rejection_reasons.append(
                f"Final score {final_score:.1f} is below the minimum threshold of {min_score:.1f}"
            )
        if risk.flags:
            rejection_reasons.append(
                f"Active risk flags: {', '.join(_label(f) for f in risk.flags)}"
            )
        if missing:
            rejection_reasons.append(
                f"Missing features reduced confidence: {', '.join(missing[:5])}"
            )

        return DeterministicExplanation(
            positive_drivers=pos if pos else ["Insufficient feature coverage for positive drivers"],
            negative_drivers=neg if neg else ["No dominant risk penalties detected"],
            missing_features=missing,
            summary=summary,
            rejection_reasons=rejection_reasons,
        )


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _label(key: str) -> str:
    return _CATEGORY_LABELS.get(key, _RISK_LABELS.get(key, key.replace("_", " ").title()))


def _make_positive(name: str, score: float, contribution: float) -> str:
    label = _CATEGORY_LABELS.get(name, name.replace("_", " ").title())
    strength = "strong" if score > 0.70 else ("moderate" if score > 0.45 else "mild")
    return f"Positive driver: {label} — {strength} signal (score {score:.2f}, contribution {contribution:.3f})"


def _make_negative(name: str, penalty: float) -> str:
    label = _RISK_LABELS.get(name, name.replace("_", " ").title())
    severity = "elevated" if penalty > 3.0 else "moderate"
    return f"Risk flag: {label} — {severity} penalty ({penalty:.2f} pts)"
