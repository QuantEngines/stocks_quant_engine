"""Long-term investment scoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.feature_access import FeatureAccessor
from stock_screener_engine.core.normalizers import clamp, inverse_linear_score, linear_score, symmetric_score
from stock_screener_engine.core.scoring_base import CategoryScore, ScoringResult, combine_categories


@dataclass(frozen=True)
class LongTermCategoryWeights:
    growth_quality: float = 0.18
    profitability_quality: float = 0.17
    balance_sheet_strength: float = 0.15
    cash_flow_quality: float = 0.12
    valuation_sanity: float = 0.12
    governance_risk_proxy: float = 0.10
    event_context: float = 0.08
    regime_context: float = 0.08


@dataclass
class LongTermInvestmentScorer:
    weights: LongTermCategoryWeights = field(default_factory=LongTermCategoryWeights)

    def score(self, features: Mapping[str, float] | None) -> ScoringResult:
        fa = FeatureAccessor(features)
        categories = [
            self._growth_quality(fa),
            self._profitability_quality(fa),
            self._balance_sheet_strength(fa),
            self._cash_flow_quality(fa),
            self._valuation_sanity(fa),
            self._governance_proxy(fa),
            self._event_context(fa),
            self._regime_context(fa),
        ]
        result = combine_categories(categories, scale=100.0)
        return ScoringResult(
            total_score=result.total_score,
            categories=result.categories,
            component_map=result.component_map,
            missing_features=sorted(set(result.missing_features + fa.missing_features() + fa.invalid_features())),
        )

    def _growth_quality(self, fa: FeatureAccessor) -> CategoryScore:
        eg = linear_score(fa.get("growth_quality", 0.0), 0.0, 0.35)
        rg = linear_score(fa.get("revenue_growth", 0.0), 0.0, 0.30)
        stability = clamp(fa.get("earnings_stability", 0.0))
        earnings_sentiment = clamp((fa.get("earnings_sentiment_score", 0.0) + 1.0) / 2.0)
        score = clamp(0.45 * eg + 0.25 * rg + 0.2 * stability + 0.10 * earnings_sentiment)
        return CategoryScore(
            name="growth_quality",
            score_0_1=score,
            weight=self.weights.growth_quality,
            contribution=score * self.weights.growth_quality,
            missing_features=[k for k in ["growth_quality", "revenue_growth", "earnings_stability", "earnings_sentiment_score"] if k in fa.missing],
        )

    def _profitability_quality(self, fa: FeatureAccessor) -> CategoryScore:
        roe = linear_score(fa.get("profitability_quality", 0.0), 0.0, 0.30)
        opm = linear_score(fa.get("operating_margin", 0.0), 0.0, 0.30)
        score = clamp(0.7 * roe + 0.3 * opm)
        return CategoryScore(
            name="profitability_quality",
            score_0_1=score,
            weight=self.weights.profitability_quality,
            contribution=score * self.weights.profitability_quality,
            missing_features=[k for k in ["profitability_quality", "operating_margin"] if k in fa.missing],
        )

    def _balance_sheet_strength(self, fa: FeatureAccessor) -> CategoryScore:
        bsh = clamp(fa.get("balance_sheet_health", 0.0))
        leverage_hint = inverse_linear_score(fa.get("debt_to_equity", 0.8), 0.2, 2.0)
        leverage_trend = clamp(fa.get("leverage_trend", 0.0))
        score = clamp(0.6 * bsh + 0.2 * leverage_hint + 0.2 * leverage_trend)
        return CategoryScore(
            name="balance_sheet_strength",
            score_0_1=score,
            weight=self.weights.balance_sheet_strength,
            contribution=score * self.weights.balance_sheet_strength,
            missing_features=[k for k in ["balance_sheet_health", "debt_to_equity", "leverage_trend"] if k in fa.missing],
        )

    def _cash_flow_quality(self, fa: FeatureAccessor) -> CategoryScore:
        cf = clamp(fa.get("cash_flow_quality", 0.0))
        cfo_pat = symmetric_score(fa.get("cfo_pat_ratio", 1.0), center=1.0, tolerance=1.0)
        score = clamp(0.8 * cf + 0.2 * cfo_pat)
        return CategoryScore(
            name="cash_flow_quality",
            score_0_1=score,
            weight=self.weights.cash_flow_quality,
            contribution=score * self.weights.cash_flow_quality,
            missing_features=[k for k in ["cash_flow_quality", "cfo_pat_ratio"] if k in fa.missing],
        )

    def _valuation_sanity(self, fa: FeatureAccessor) -> CategoryScore:
        valuation = clamp(fa.get("valuation_sanity", 0.0))
        pe = inverse_linear_score(fa.get("pe_ratio", 25.0), 8.0, 45.0)
        score = clamp(0.7 * valuation + 0.3 * pe)
        return CategoryScore(
            name="valuation_sanity",
            score_0_1=score,
            weight=self.weights.valuation_sanity,
            contribution=score * self.weights.valuation_sanity,
            missing_features=[k for k in ["valuation_sanity", "pe_ratio"] if k in fa.missing],
        )

    def _governance_proxy(self, fa: FeatureAccessor) -> CategoryScore:
        g = clamp(fa.get("governance_proxy", 0.0))
        text_governance = clamp(fa.get("governance_flag_score", 0.0))
        tone = clamp((fa.get("management_tone_score", 0.0) + 1.0) / 2.0)
        score = clamp(0.6 * g + 0.25 * text_governance + 0.15 * tone)
        return CategoryScore(
            name="governance_risk_proxy",
            score_0_1=score,
            weight=self.weights.governance_risk_proxy,
            contribution=score * self.weights.governance_risk_proxy,
            missing_features=[k for k in ["governance_proxy", "governance_flag_score", "management_tone_score"] if k in fa.missing],
        )

    def _event_context(self, fa: FeatureAccessor) -> CategoryScore:
        event = fa.get("event_catalyst", 0.0)
        catalyst = clamp(fa.get("catalyst_presence_flag", 0.0))
        sentiment_trend = clamp((fa.get("sentiment_trend", 0.0) + 1.0) / 2.0)
        score = clamp(0.6 * ((event + 1.0) / 2.0) + 0.25 * catalyst + 0.15 * sentiment_trend)
        return CategoryScore(
            name="event_context",
            score_0_1=score,
            weight=self.weights.event_context,
            contribution=score * self.weights.event_context,
            missing_features=[k for k in ["event_catalyst", "catalyst_presence_flag", "sentiment_trend"] if k in fa.missing],
        )

    def _regime_context(self, fa: FeatureAccessor) -> CategoryScore:
        regime = fa.get("market_regime_score", 0.0)
        score = clamp((regime + 1.0) / 2.0)
        return CategoryScore(
            name="regime_context",
            score_0_1=score,
            weight=self.weights.regime_context,
            contribution=score * self.weights.regime_context,
            missing_features=[k for k in ["market_regime_score"] if k in fa.missing],
        )
