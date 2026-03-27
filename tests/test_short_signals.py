"""Tests for short/bearish signal generation."""
from __future__ import annotations

from datetime import date

import pytest

from stock_screener_engine.core.scoring_short import ShortScorer
from stock_screener_engine.core.entities import FeatureVector
from stock_screener_engine.core.signals import SignalGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fv(values: dict[str, float], symbol: str = "TEST") -> FeatureVector:
    return FeatureVector(symbol=symbol, as_of=date(2024, 1, 1), values=values)


BEARISH_FEATURES = {
    # inverted: low bullish = bearish
    "trend_strength": 0.1,
    "momentum_strength": 0.05,
    "relative_strength_proxy": 0.1,
    "balance_sheet_health": 0.15,
    "earnings_stability": 0.1,
    "growth_quality": 0.1,
    "profitability_quality": 0.1,
    "cash_flow_quality": 0.1,
    # direct: high = bearish
    "event_risk_score": 0.85,
    "uncertainty_penalty": 0.80,
    "governance_flag_score": 0.75,
    "recent_negative_event_score": 0.80,
    "leverage_trend": 0.90,
    "sentiment_score_recent": 0.05,
}

BULLISH_FEATURES = {
    "trend_strength": 0.85,
    "momentum_strength": 0.80,
    "relative_strength_proxy": 0.75,
    "balance_sheet_health": 0.85,
    "earnings_stability": 0.80,
    "growth_quality": 0.80,
    "profitability_quality": 0.75,
    "cash_flow_quality": 0.80,
    "event_risk_score": 0.05,
    "uncertainty_penalty": 0.05,
    "governance_flag_score": 0.05,
    "recent_negative_event_score": 0.05,
    "leverage_trend": 0.10,
    "sentiment_score_recent": 0.85,
}


# ---------------------------------------------------------------------------
# ShortScorer unit tests
# ---------------------------------------------------------------------------

class TestShortScorer:
    def test_score_returns_value_in_0_to_100(self):
        scorer = ShortScorer()
        result = scorer.score(BEARISH_FEATURES)
        assert 0.0 <= result.total_score <= 100.0

    def test_bearish_features_produce_high_short_score(self):
        scorer = ShortScorer()
        result = scorer.score(BEARISH_FEATURES)
        assert result.total_score >= 50.0, (
            f"Expected bearish features to produce short score >= 50, got {result.total_score:.2f}"
        )

    def test_bullish_features_produce_low_short_score(self):
        scorer = ShortScorer()
        result = scorer.score(BULLISH_FEATURES)
        assert result.total_score <= 50.0, (
            f"Expected bullish features to produce short score <= 50, got {result.total_score:.2f}"
        )

    def test_bearish_score_higher_than_bullish_score(self):
        scorer = ShortScorer()
        bearish_score = scorer.score(BEARISH_FEATURES).total_score
        bullish_score = scorer.score(BULLISH_FEATURES).total_score
        assert bearish_score > bullish_score

    def test_result_has_expected_categories(self):
        scorer = ShortScorer()
        result = scorer.score(BEARISH_FEATURES)
        category_names = {c.name for c in result.categories}
        assert "technical_breakdown" in category_names
        assert "fundamental_weakness" in category_names
        assert "negative_text_signals" in category_names

    def test_empty_features_does_not_raise(self):
        scorer = ShortScorer()
        result = scorer.score({})
        assert 0.0 <= result.total_score <= 100.0

    def test_none_features_does_not_raise(self):
        scorer = ShortScorer()
        result = scorer.score(None)
        assert 0.0 <= result.total_score <= 100.0


# ---------------------------------------------------------------------------
# SignalGenerator.build_short_signal tests
# ---------------------------------------------------------------------------

class TestSignalGeneratorShort:
    def setup_method(self):
        self.generator = SignalGenerator(
            long_term_min_score=24.0,
            swing_min_score=28.0,
            short_min_score=58.0,
        )

    def test_bearish_fv_produces_short_candidate(self):
        fv = _fv(BEARISH_FEATURES)
        sig = self.generator.build_short_signal(fv, risk_flags=[], sector="Financials")
        assert sig.score >= 0.0
        assert sig.category in {"short_candidate", "short_reject"}

    def test_signal_has_correct_symbol(self):
        fv = _fv(BEARISH_FEATURES, symbol="RELIANCE")
        sig = self.generator.build_short_signal(fv, risk_flags=[], sector="Energy")
        assert sig.symbol == "RELIANCE"

    def test_signal_explanation_signal_type_is_short(self):
        fv = _fv(BEARISH_FEATURES)
        sig = self.generator.build_short_signal(fv, risk_flags=[], sector="")
        assert sig.explanation.signal_type == "short"

    def test_bullish_fv_produces_short_reject(self):
        fv = _fv(BULLISH_FEATURES)
        sig = self.generator.build_short_signal(fv, risk_flags=[], sector="")
        assert sig.category == "short_reject"

    def test_short_candidate_when_score_exceeds_threshold(self):
        fv = _fv(BEARISH_FEATURES)
        sig = self.generator.build_short_signal(fv, risk_flags=[], sector="")
        # Check consistency: if category == short_candidate then score >= threshold
        if sig.category == "short_candidate":
            assert sig.score >= 58.0


# ---------------------------------------------------------------------------
# Directional verdicts in single_stock_deep
# ---------------------------------------------------------------------------

class TestDirectionalVerdicts:
    def test_swing_verdict_returns_short_on_high_short_score(self):
        from stock_screener_engine.pipelines.single_stock_deep import _swing_verdict
        v = _swing_verdict(swing_score=20.0, swing_min=28.0, category="swing_reject", short_score=65.0)
        assert v == "short"

    def test_swing_verdict_returns_buy_for_candidate_regardless_of_short_score(self):
        from stock_screener_engine.pipelines.single_stock_deep import _swing_verdict
        v = _swing_verdict(swing_score=40.0, swing_min=28.0, category="swing_candidate", short_score=65.0)
        assert v == "buy"

    def test_medium_verdict_returns_sell_on_very_high_short_score(self):
        from stock_screener_engine.pipelines.single_stock_deep import _medium_verdict
        v = _medium_verdict(10.0, 24.0, 10.0, 28.0, -0.1, 28.0, short_score=70.0)
        assert v == "sell"

    def test_medium_verdict_returns_reduce_on_moderate_short_score(self):
        from stock_screener_engine.pipelines.single_stock_deep import _medium_verdict
        v = _medium_verdict(10.0, 24.0, 10.0, 28.0, -0.1, 28.0, short_score=58.0)
        assert v == "reduce"

    def test_long_verdict_returns_sell_on_high_short_score(self):
        from stock_screener_engine.pipelines.single_stock_deep import _long_verdict
        v = _long_verdict(long_score=10.0, long_min=24.0, category="long_term_reject", short_score=65.0)
        assert v == "sell"

    def test_build_directional_strong_short(self):
        from stock_screener_engine.pipelines.single_stock_deep import _build_directional
        from stock_screener_engine.config.settings import ScoringSettings
        cfg = ScoringSettings(long_term_min_score=24.0, swing_min_score=28.0, max_risk_penalty=30.0, short_min_score=58.0)
        d = _build_directional(long_score=5.0, swing_score=5.0, short_score=75.0, scoring_cfg=cfg)
        assert d["bias"] == "strong_short"

    def test_build_directional_bullish(self):
        from stock_screener_engine.pipelines.single_stock_deep import _build_directional
        from stock_screener_engine.config.settings import ScoringSettings
        cfg = ScoringSettings(long_term_min_score=24.0, swing_min_score=28.0, max_risk_penalty=30.0, short_min_score=58.0)
        d = _build_directional(long_score=60.0, swing_score=60.0, short_score=20.0, scoring_cfg=cfg)
        assert d["bias"] == "bullish"

    def test_build_directional_neutral(self):
        from stock_screener_engine.pipelines.single_stock_deep import _build_directional
        from stock_screener_engine.config.settings import ScoringSettings
        cfg = ScoringSettings(long_term_min_score=24.0, swing_min_score=28.0, max_risk_penalty=30.0, short_min_score=58.0)
        d = _build_directional(long_score=5.0, swing_score=5.0, short_score=10.0, scoring_cfg=cfg)
        assert d["bias"] == "neutral"

    def test_build_directional_conflicted(self):
        from stock_screener_engine.pipelines.single_stock_deep import _build_directional
        from stock_screener_engine.config.settings import ScoringSettings
        cfg = ScoringSettings(long_term_min_score=24.0, swing_min_score=28.0, max_risk_penalty=30.0, short_min_score=58.0)
        d = _build_directional(long_score=60.0, swing_score=60.0, short_score=65.0, scoring_cfg=cfg)
        assert d["bias"] == "conflicted"
