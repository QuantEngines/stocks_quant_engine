"""Tests for ExplanationEngine and the explainability helpers."""

from __future__ import annotations

from stock_screener_engine.core.entities import ScoreCard, SignalExplanation
from stock_screener_engine.core.explainability import ExplanationEngine, _pretty_name, _top_components


class TestPrettyName:
    def test_strips_long_prefix(self) -> None:
        assert _pretty_name("long_growth_quality") == "growth quality"

    def test_strips_swing_prefix(self) -> None:
        assert _pretty_name("swing_trend_strength") == "trend strength"

    def test_strips_risk_prefix(self) -> None:
        assert _pretty_name("risk_leverage_risk") == "leverage risk"

    def test_no_prefix(self) -> None:
        assert _pretty_name("some_feature") == "some feature"


class TestTopComponents:
    def test_returns_top_n_by_value(self) -> None:
        components = {
            "long_a": 5.0,
            "long_b": 10.0,
            "long_c": 8.0,
            "swing_x": 3.0,
        }
        result = _top_components(components, prefix="long_", top_n=2)
        keys = [k for k, _ in result]
        assert keys == ["long_b", "long_c"]

    def test_prefix_filter_excludes_other_prefixes(self) -> None:
        components = {"long_a": 9.0, "risk_b": 7.0, "swing_c": 6.0}
        result = _top_components(components, prefix="risk_", top_n=3)
        assert all(k.startswith("risk_") for k, _ in result)

    def test_descending_false_returns_smallest_first(self) -> None:
        components = {"risk_a": 1.0, "risk_b": 5.0, "risk_c": 3.0}
        result = _top_components(components, prefix="risk_", top_n=2, descending=False)
        values = [v for _, v in result]
        assert values == sorted(values)


class TestExplanationEngine:
    def test_builds_explanation_for_long_term_pass(
        self, sample_score_card: ScoreCard
    ) -> None:
        engine = ExplanationEngine()
        exp = engine.build(
            score_card=sample_score_card,
            signal_type="long_term",
            passed_filter=True,
            min_score=20.0,
            risk_flags=[],
        )
        assert isinstance(exp, SignalExplanation)
        assert exp.signal_type == "long_term"
        assert exp.rejection_reason is None
        assert exp.holding_horizon == "6\u201324 months"

    def test_builds_rejection_reason_when_below_threshold(
        self, sample_score_card: ScoreCard
    ) -> None:
        engine = ExplanationEngine()
        exp = engine.build(
            score_card=sample_score_card,
            signal_type="long_term",
            passed_filter=False,
            min_score=999.0,
            risk_flags=["leverage_risk"],
        )
        assert exp.rejection_reason is not None
        assert "999.0" in exp.rejection_reason
        assert "leverage_risk" in exp.risk_flags

    def test_swing_signal_has_correct_horizon(
        self, sample_score_card: ScoreCard
    ) -> None:
        engine = ExplanationEngine()
        exp = engine.build(
            score_card=sample_score_card,
            signal_type="swing",
            passed_filter=True,
            min_score=20.0,
            risk_flags=[],
        )
        assert exp.holding_horizon == "3\u201315 trading days"

    def test_positive_drivers_are_readable_strings(
        self, sample_score_card: ScoreCard
    ) -> None:
        engine = ExplanationEngine()
        exp = engine.build(
            score_card=sample_score_card,
            signal_type="long_term",
            passed_filter=True,
            min_score=20.0,
            risk_flags=[],
        )
        for driver in exp.top_positive_drivers:
            # Should be "label: value" format, not raw key like "long_growth_quality:5.00"
            assert "long_" not in driver, f"Raw prefix found in driver: {driver}"
            assert ":" in driver

    def test_no_duplicate_function_regression(
        self, sample_score_card: ScoreCard
    ) -> None:
        """Ensure negative drivers are sorted by risk prefix, not positive prefix."""
        engine = ExplanationEngine()
        exp = engine.build(
            score_card=sample_score_card,
            signal_type="long_term",
            passed_filter=True,
            min_score=20.0,
            risk_flags=[],
        )
        # Positive drivers should prefix 'long_' items; negative should prefix 'risk_' items
        # If _top_negative_risks was broken (copy of _top_drivers), they'd return same values
        negative_labels = [d.split(":")[0].strip() for d in exp.top_negative_drivers]
        # Negative driver labels should NOT be long_ feature names (e.g. "growth quality")
        positive_labels = [d.split(":")[0].strip() for d in exp.top_positive_drivers]
        overlap = set(positive_labels) & set(negative_labels)
        assert not overlap, f"Positive and negative drivers overlap: {overlap}"
