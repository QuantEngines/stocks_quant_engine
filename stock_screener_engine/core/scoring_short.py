"""Bearish / short-bias scoring engine.

A high short score (0–100) means the stock has strong bearish characteristics
and may be a candidate for short-selling or exits/reductions of longs.

The score rewards features that indicate:
  – Technical breakdown  (negative momentum, downtrend, below key moving averages)
  – Fundamental weakness (high leverage, declining earnings, poor cash flow)
  – Negative event/text signals (negative sentiment, governance concerns, uncertainty)
  – High risk environment (illiquidity, high volatility on downside, earnings risk)

Each category uses `inverse_linear_score` or `1 - value` for already-0-to-1 features,
meaning the *absence* of bullish characteristics becomes a bearish signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.feature_access import FeatureAccessor
from stock_screener_engine.core.normalizers import clamp, linear_score
from stock_screener_engine.core.scoring_base import CategoryScore, ScoringResult, combine_categories


@dataclass(frozen=True)
class ShortCategoryWeights:
    technical_breakdown:   float = 0.30   # most important: price is moving against you
    fundamental_weakness:  float = 0.25   # deteriorating quality
    negative_text_signals: float = 0.20   # news, events driving bearish flow
    risk_amplifiers:       float = 0.15   # risk that accelerates downside
    valuation_trap:        float = 0.10   # overvalued relative to fundamentals


@dataclass
class ShortScorer:
    """Scores how attractive a stock is as a short / sell candidate (0 = no signal, 100 = strong short)."""

    weights: ShortCategoryWeights = field(default_factory=ShortCategoryWeights)

    def score(self, features: Mapping[str, float] | None) -> ScoringResult:
        fa = FeatureAccessor(features)
        categories = [
            self._technical_breakdown(fa),
            self._fundamental_weakness(fa),
            self._negative_text_signals(fa),
            self._risk_amplifiers(fa),
            self._valuation_trap(fa),
        ]
        result = combine_categories(categories, scale=100.0)
        return ScoringResult(
            total_score=result.total_score,
            categories=result.categories,
            component_map=result.component_map,
            missing_features=sorted(
                set(result.missing_features + fa.missing_features() + fa.invalid_features())
            ),
        )

    # ── Category scorers ─────────────────────────────────────────────────────

    def _technical_breakdown(self, fa: FeatureAccessor) -> CategoryScore:
        # Low trend/momentum = bearish; invert 0-1 features
        trend_break = clamp(1.0 - fa.get("trend_strength", 0.5))
        mom_break   = clamp(1.0 - fa.get("momentum_strength", 0.5))
        rel_weak    = clamp(1.0 - fa.get("relative_strength_proxy", 0.5))
        # volatility_regime being low means compression before a breakdown; high = already volatile
        # We want high-downside-vol (high volatility_regime with bearish context)
        vol         = clamp(fa.get("volatility_regime", 0.0))
        # delivery_ratio_signal low = more speculative / churning = bearish
        del_break   = clamp(1.0 - fa.get("delivery_ratio_signal", 0.5))
        score = clamp(0.30 * trend_break + 0.30 * mom_break + 0.20 * rel_weak + 0.10 * vol + 0.10 * del_break)
        return CategoryScore(
            name="technical_breakdown",
            score_0_1=score,
            weight=self.weights.technical_breakdown,
            contribution=score * self.weights.technical_breakdown,
            missing_features=[k for k in ["trend_strength", "momentum_strength", "relative_strength_proxy"] if k in fa.missing],
        )

    def _fundamental_weakness(self, fa: FeatureAccessor) -> CategoryScore:
        # Poor quality = good short. Invert 0-1 quality features.
        growth_weak  = clamp(1.0 - fa.get("growth_quality", 0.5))
        profit_weak  = clamp(1.0 - fa.get("profitability_quality", 0.5))
        bs_weak      = clamp(1.0 - fa.get("balance_sheet_health", 0.5))
        cf_weak      = clamp(1.0 - fa.get("cash_flow_quality", 0.5))
        leverage_bad = clamp(fa.get("leverage_trend", 0.5))        # high leverage = bearish
        earn_unstab  = clamp(1.0 - fa.get("earnings_stability", 0.5))
        score = clamp(
            0.20 * growth_weak + 0.20 * profit_weak + 0.20 * bs_weak
            + 0.15 * cf_weak   + 0.15 * leverage_bad + 0.10 * earn_unstab
        )
        return CategoryScore(
            name="fundamental_weakness",
            score_0_1=score,
            weight=self.weights.fundamental_weakness,
            contribution=score * self.weights.fundamental_weakness,
            missing_features=[k for k in ["growth_quality", "profitability_quality", "balance_sheet_health"] if k in fa.missing],
        )

    def _negative_text_signals(self, fa: FeatureAccessor) -> CategoryScore:
        # Negative sentiment & high-risk events are bearish
        neg_sent  = clamp(1.0 - fa.get("sentiment_score_recent", 0.5))
        neg_news  = clamp(1.0 - fa.get("news_sentiment",         0.5))
        event_risk = clamp(fa.get("event_risk_score",           0.0))
        gov_flag   = clamp(fa.get("governance_flag_score",       0.0))
        uncertainty = clamp(fa.get("uncertainty_penalty",        0.0))
        neg_events  = clamp(fa.get("recent_negative_event_score", 0.0))
        # Negative management tone
        mgmt_neg  = clamp(1.0 - clamp(fa.get("management_tone_score", 0.5)))
        score = clamp(
            0.20 * neg_sent + 0.15 * neg_news + 0.15 * event_risk
            + 0.15 * neg_events + 0.15 * gov_flag + 0.10 * uncertainty + 0.10 * mgmt_neg
        )
        return CategoryScore(
            name="negative_text_signals",
            score_0_1=score,
            weight=self.weights.negative_text_signals,
            contribution=score * self.weights.negative_text_signals,
            missing_features=[k for k in ["sentiment_score_recent", "event_risk_score", "governance_flag_score"] if k in fa.missing],
        )

    def _risk_amplifiers(self, fa: FeatureAccessor) -> CategoryScore:
        # High risk factors that accelerate downside moves
        liq_risk      = clamp(fa.get("liquidity_risk",            0.0))
        lever_risk     = clamp(fa.get("leverage_risk",             0.0))
        earn_risk      = clamp(fa.get("earnings_instability_risk", 0.0))
        event_unc_risk = clamp(fa.get("event_uncertainty_risk",    0.0))
        gov_risk       = clamp(fa.get("governance_risk",           0.0))
        score = clamp(
            0.25 * liq_risk + 0.25 * lever_risk + 0.20 * earn_risk
            + 0.15 * event_unc_risk + 0.15 * gov_risk
        )
        return CategoryScore(
            name="risk_amplifiers",
            score_0_1=score,
            weight=self.weights.risk_amplifiers,
            contribution=score * self.weights.risk_amplifiers,
            missing_features=[k for k in ["leverage_risk", "earnings_instability_risk"] if k in fa.missing],
        )

    def _valuation_trap(self, fa: FeatureAccessor) -> CategoryScore:
        # Expensive stock with poor value = short candidate
        val_bad   = clamp(1.0 - fa.get("valuation_sanity", 0.5))
        pe_zscore = linear_score(fa.get("rolling_pe_zscore", 0.0), 0.5, 2.5)   # high z-score = overvalued
        pb_zscore = linear_score(fa.get("rolling_pb_zscore", 0.0), 0.5, 2.5)
        score = clamp(0.50 * val_bad + 0.25 * pe_zscore + 0.25 * pb_zscore)
        return CategoryScore(
            name="valuation_trap",
            score_0_1=score,
            weight=self.weights.valuation_trap,
            contribution=score * self.weights.valuation_trap,
            missing_features=[k for k in ["valuation_sanity"] if k in fa.missing],
        )
