"""Risk penalty engine reusable by long-term and swing scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.feature_access import FeatureAccessor
from stock_screener_engine.core.normalizers import clamp, inverse_linear_score


@dataclass(frozen=True)
class RiskPenaltyWeights:
    liquidity_risk: float = 0.20
    volatility_risk: float = 0.20
    leverage_risk: float = 0.20
    earnings_instability_risk: float = 0.15
    event_uncertainty_risk: float = 0.15
    governance_risk: float = 0.10
    text_uncertainty_risk: float = 0.05


@dataclass(frozen=True)
class RiskPenaltyResult:
    total_penalty: float
    components: dict[str, float]
    flags: list[str]
    missing_features: list[str]


@dataclass
class RiskPenaltyEngine:
    weights: RiskPenaltyWeights = field(default_factory=RiskPenaltyWeights)
    max_penalty: float = 30.0

    def score(self, features: Mapping[str, float] | None) -> RiskPenaltyResult:
        fa = FeatureAccessor(features)
        c = {
            "liquidity_risk": self.weights.liquidity_risk * self._liquidity_risk(fa),
            "volatility_risk": self.weights.volatility_risk * self._volatility_risk(fa),
            "leverage_risk": self.weights.leverage_risk * self._leverage_risk(fa),
            "earnings_instability_risk": self.weights.earnings_instability_risk * self._earnings_instability_risk(fa),
            "event_uncertainty_risk": self.weights.event_uncertainty_risk * self._event_uncertainty_risk(fa),
            "governance_risk": self.weights.governance_risk * self._governance_risk(fa),
            "text_uncertainty_risk": self.weights.text_uncertainty_risk * self._text_uncertainty_risk(fa),
        }
        weighted = sum(c.values())
        # components are 0..1 weighted; map to max_penalty scale
        total_penalty = min(self.max_penalty, weighted * self.max_penalty)
        flags = [name for name, value in c.items() if value >= 0.10]
        return RiskPenaltyResult(
            total_penalty=total_penalty,
            components={k: v * self.max_penalty for k, v in c.items()},
            flags=sorted(flags),
            missing_features=sorted(set(fa.missing_features() + fa.invalid_features())),
        )

    def _liquidity_risk(self, fa: FeatureAccessor) -> float:
        vc = clamp(fa.get("volume_confirmation", 0.0))
        return clamp(1.0 - vc)

    def _volatility_risk(self, fa: FeatureAccessor) -> float:
        regime = clamp(fa.get("volatility_regime", 0.0))
        return clamp(1.0 - regime)

    def _leverage_risk(self, fa: FeatureAccessor) -> float:
        bsh = clamp(fa.get("balance_sheet_health", 0.0))
        de = inverse_linear_score(fa.get("debt_to_equity", 0.8), 0.2, 2.0)
        trend = clamp(fa.get("leverage_trend", 0.0))
        return clamp(1.0 - (0.5 * bsh + 0.2 * de + 0.3 * trend))

    def _earnings_instability_risk(self, fa: FeatureAccessor) -> float:
        growth = clamp(fa.get("growth_quality", 0.0))
        stability = clamp(fa.get("earnings_stability", 0.0))
        return clamp(1.0 - (0.4 * growth + 0.6 * stability))

    def _event_uncertainty_risk(self, fa: FeatureAccessor) -> float:
        event = fa.get("event_catalyst", 0.0)
        event_risk = clamp(fa.get("event_risk_score", 0.0))
        return clamp(max(0.0, -event) * 0.6 + event_risk * 0.4)

    def _governance_risk(self, fa: FeatureAccessor) -> float:
        g = clamp(fa.get("governance_proxy", 0.0))
        governance_flag = clamp(fa.get("governance_flag_score", 0.0))
        governance_risk_score = clamp(fa.get("governance_risk_score", 0.0))
        return clamp(1.0 - (0.6 * g + 0.2 * (1.0 - governance_flag) + 0.2 * (1.0 - governance_risk_score)))

    def _text_uncertainty_risk(self, fa: FeatureAccessor) -> float:
        uncertainty = clamp(fa.get("uncertainty_penalty", 0.0))
        high_impact = clamp(fa.get("high_impact_event_flag", 0.0))
        transcript_quality = clamp(fa.get("transcript_quality_signal", 0.0))
        return clamp(0.6 * uncertainty + 0.25 * uncertainty * high_impact + 0.15 * (1.0 - transcript_quality))
