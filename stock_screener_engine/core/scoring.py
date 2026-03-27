"""Legacy compatibility layer for modular scoring engines.

Public classes in this module are retained to avoid breaking existing imports.
Internally they delegate to the modular scoring engines in:

- ``core/scoring_long_term.py``
- ``core/scoring_swing.py``
- ``core/scoring_risk.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.entities import FeatureVector, ScoreCard
from stock_screener_engine.core.scoring_long_term import (
    LongTermCategoryWeights,
    LongTermInvestmentScorer,
)
from stock_screener_engine.core.scoring_risk import RiskPenaltyEngine, RiskPenaltyWeights
from stock_screener_engine.core.scoring_swing import SwingCategoryWeights, SwingTradeScorer


def _get(values: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return float(values.get(key, default))


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, value))


# ---------------------------------------------------------------------------
# Configurable weight containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LongTermWeights:
    growth_quality:       float = 18.0
    profitability_quality: float = 17.0
    balance_sheet_health: float = 15.0
    cash_flow_quality:    float = 12.0
    valuation_sanity:     float = 12.0
    governance_proxy:     float = 10.0
    event_catalyst:       float = 8.0
    regime_tailwind:      float = 8.0

    @classmethod
    def from_dict(cls, d: dict) -> "LongTermWeights":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SwingWeights:
    trend_strength:         float = 20.0
    momentum_strength:      float = 18.0
    relative_strength_proxy: float = 14.0
    volatility_regime:      float = 12.0
    volume_confirmation:    float = 12.0
    event_catalyst:         float = 12.0
    sentiment_score:        float = 12.0

    @classmethod
    def from_dict(cls, d: dict) -> "SwingWeights":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RegimeSwitchConfig:
    enabled: bool = True
    bull_threshold: float = 0.25
    bear_threshold: float = -0.25


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LongTermScorer:
    weights: LongTermWeights = field(default_factory=LongTermWeights)
    regime_switch: RegimeSwitchConfig = field(default_factory=RegimeSwitchConfig)
    regime_profiles: Mapping[str, Mapping[str, float]] | None = None

    def score(self, fv: FeatureVector) -> tuple[float, dict[str, float]]:
        effective = self._weights_for_regime(fv.values.get("market_regime_score", 0.0))
        internal = LongTermInvestmentScorer(
            weights=LongTermCategoryWeights(
                growth_quality=effective.growth_quality,
                profitability_quality=effective.profitability_quality,
                balance_sheet_strength=effective.balance_sheet_health,
                cash_flow_quality=effective.cash_flow_quality,
                valuation_sanity=effective.valuation_sanity,
                governance_risk_proxy=effective.governance_proxy,
                event_context=effective.event_catalyst,
                regime_context=effective.regime_tailwind,
            )
        )
        result = internal.score(fv.values)
        components = {
            "growth_quality": result.component_map.get("growth_quality", 0.0),
            "profitability_quality": result.component_map.get("profitability_quality", 0.0),
            "balance_sheet_health": result.component_map.get("balance_sheet_strength", 0.0),
            "cash_flow_quality": result.component_map.get("cash_flow_quality", 0.0),
            "valuation_sanity": result.component_map.get("valuation_sanity", 0.0),
            "governance_proxy": result.component_map.get("governance_risk_proxy", 0.0),
            "event_catalyst": result.component_map.get("event_context", 0.0),
            "regime_tailwind": result.component_map.get("regime_context", 0.0),
        }
        return _clamp_0_100(sum(components.values())), components

    def _weights_for_regime(self, regime_score: float) -> LongTermWeights:
        if not self.regime_switch.enabled:
            return self.weights
        label = _regime_label(
            regime_score,
            bull_threshold=self.regime_switch.bull_threshold,
            bear_threshold=self.regime_switch.bear_threshold,
        )
        override = (self.regime_profiles or {}).get(label) or {}
        payload = {
            name: float(override.get(name, getattr(self.weights, name)))
            for name in self.weights.__dataclass_fields__
        }
        return LongTermWeights(**payload)


@dataclass(frozen=True)
class SwingScorer:
    weights: SwingWeights = field(default_factory=SwingWeights)
    regime_switch: RegimeSwitchConfig = field(default_factory=RegimeSwitchConfig)
    regime_profiles: Mapping[str, Mapping[str, float]] | None = None

    def score(self, fv: FeatureVector) -> tuple[float, dict[str, float]]:
        effective = self._weights_for_regime(fv.values.get("market_regime_score", 0.0))
        internal = SwingTradeScorer(
            weights=SwingCategoryWeights(
                trend=effective.trend_strength,
                momentum=effective.momentum_strength,
                volume_participation=effective.volume_confirmation,
                volatility_regime=effective.volatility_regime,
                setup_quality=effective.relative_strength_proxy,
                catalyst_awareness=(effective.event_catalyst + effective.sentiment_score) / 2.0,
            )
        )
        result = internal.score(fv.values)
        components = {
            "trend_strength": result.component_map.get("trend", 0.0),
            "momentum_strength": result.component_map.get("momentum", 0.0),
            "relative_strength_proxy": result.component_map.get("setup_quality", 0.0),
            "volatility_regime": result.component_map.get("volatility_regime", 0.0),
            "volume_confirmation": result.component_map.get("volume_participation", 0.0),
            "event_catalyst": result.component_map.get("catalyst_awareness", 0.0),
            "sentiment_score": result.component_map.get("catalyst_awareness", 0.0),
        }
        return _clamp_0_100(sum(components.values())), components

    def _weights_for_regime(self, regime_score: float) -> SwingWeights:
        if not self.regime_switch.enabled:
            return self.weights
        label = _regime_label(
            regime_score,
            bull_threshold=self.regime_switch.bull_threshold,
            bear_threshold=self.regime_switch.bear_threshold,
        )
        override = (self.regime_profiles or {}).get(label) or {}
        payload = {
            name: float(override.get(name, getattr(self.weights, name)))
            for name in self.weights.__dataclass_fields__
        }
        return SwingWeights(**payload)


@dataclass(frozen=True)
class RiskPenaltyScorer:
    max_penalty: float = 30.0
    weights: RiskPenaltyWeights = field(default_factory=RiskPenaltyWeights)

    def score(self, fv: FeatureVector) -> tuple[float, dict[str, float], list[str]]:
        result = RiskPenaltyEngine(weights=self.weights, max_penalty=self.max_penalty).score(fv.values)
        penalties = {
            "liquidity_risk": result.components.get("liquidity_risk", 0.0),
            "event_risk": result.components.get("event_uncertainty_risk", 0.0),
            "leverage_risk": result.components.get("leverage_risk", 0.0),
            "governance_risk": result.components.get("governance_risk", 0.0),
            "volatility_risk": result.components.get("volatility_risk", 0.0),
            "text_uncertainty_risk": result.components.get("text_uncertainty_risk", 0.0),
        }
        return result.total_penalty, penalties, result.flags


def build_score_card(
    fv: FeatureVector,
    long_term_scorer: LongTermScorer,
    swing_scorer: SwingScorer,
    risk_scorer: RiskPenaltyScorer,
) -> ScoreCard:
    long_score, long_components = long_term_scorer.score(fv)
    swing_score, swing_components = swing_scorer.score(fv)
    risk_penalty, risk_components, _ = risk_scorer.score(fv)

    adjusted_long = _clamp_0_100(long_score - risk_penalty)
    adjusted_swing = _clamp_0_100(swing_score - risk_penalty)
    conviction = _clamp_0_100((adjusted_long + adjusted_swing) / 2.0)

    all_components = {
        **{f"long_{k}": v for k, v in long_components.items()},
        **{f"swing_{k}": v for k, v in swing_components.items()},
        **{f"risk_{k}": v for k, v in risk_components.items()},
    }

    return ScoreCard(
        symbol=fv.symbol,
        as_of=fv.as_of,
        long_term_score=adjusted_long,
        swing_score=adjusted_swing,
        risk_penalty=risk_penalty,
        conviction=conviction,
        component_scores=all_components,
    )


def _regime_label(regime_score: float, bull_threshold: float, bear_threshold: float) -> str:
    if regime_score >= bull_threshold:
        return "bull"
    if regime_score <= bear_threshold:
        return "bear"
    return "sideways"
