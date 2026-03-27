"""Signal generation with thresholding and explainability."""

from __future__ import annotations

from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalExplanation, SignalResult
from stock_screener_engine.core.scoring_base import ScoringResult
from stock_screener_engine.core.scoring_risk import RiskPenaltyResult
from stock_screener_engine.core.scoring_short import ShortScorer
from stock_screener_engine.core.signal_generator import ResearchSignalGenerator, SignalThresholds


class SignalGenerator:
    def __init__(
        self,
        long_term_min_score: float,
        swing_min_score: float,
        short_min_score: float = 58.0,
    ) -> None:
        self.long_term_min_score = long_term_min_score
        self.swing_min_score = swing_min_score
        self.short_min_score = short_min_score
        self.short_scorer = ShortScorer()
        self.modular = ResearchSignalGenerator(
            thresholds=SignalThresholds(
                min_long_term=long_term_min_score,
                min_swing=swing_min_score,
                min_short=short_min_score,
            )
        )

    def build_long_term_signal(
        self, score_card: ScoreCard, risk_flags: list[str], sector: str = ""
    ) -> SignalResult:
        modular_signal = self.modular.build_long_term(
            symbol=score_card.symbol,
            as_of=score_card.as_of,
            long_term=ScoringResult(
                total_score=score_card.long_term_score,
                categories=[],
                component_map={
                    k.replace("long_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("long_")
                },
                missing_features=[],
            ),
            swing=ScoringResult(
                total_score=score_card.swing_score,
                categories=[],
                component_map={
                    k.replace("swing_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("swing_")
                },
                missing_features=[],
            ),
            risk=RiskPenaltyResult(
                total_penalty=score_card.risk_penalty,
                components={
                    k.replace("risk_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("risk_")
                },
                flags=list(risk_flags),
                missing_features=[],
            ),
        )
        explanation = SignalExplanation(
            signal_type="long_term",
            score=score_card.long_term_score,
            top_positive_drivers=modular_signal.drivers.top_positive,
            top_negative_drivers=modular_signal.drivers.top_negative,
            ranking_reason=f"final score={modular_signal.final_score:.1f}, conviction={modular_signal.conviction:.1f}",
            rejection_reason=(
                None if modular_signal.signal_category.endswith("candidate") else "; ".join(modular_signal.rejection_reasons)
            ),
            holding_horizon=modular_signal.horizon,
            risk_flags=list(risk_flags),
            confidence=modular_signal.conviction,
            entry_logic="Staggered accumulation on valuation comfort",
            invalidation_logic="Review thesis if quality and cash flow degrade",
        )
        category = modular_signal.signal_category
        return SignalResult(
            symbol=score_card.symbol,
            category=category,
            score=score_card.long_term_score,
            explanation=explanation,
            sector=sector,
        )

    def build_swing_signal(
        self, score_card: ScoreCard, risk_flags: list[str], sector: str = ""
    ) -> SignalResult:
        modular_signal = self.modular.build_swing(
            symbol=score_card.symbol,
            as_of=score_card.as_of,
            long_term=ScoringResult(
                total_score=score_card.long_term_score,
                categories=[],
                component_map={
                    k.replace("long_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("long_")
                },
                missing_features=[],
            ),
            swing=ScoringResult(
                total_score=score_card.swing_score,
                categories=[],
                component_map={
                    k.replace("swing_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("swing_")
                },
                missing_features=[],
            ),
            risk=RiskPenaltyResult(
                total_penalty=score_card.risk_penalty,
                components={
                    k.replace("risk_", ""): v
                    for k, v in score_card.component_scores.items()
                    if k.startswith("risk_")
                },
                flags=list(risk_flags),
                missing_features=[],
            ),
        )
        explanation = SignalExplanation(
            signal_type="swing",
            score=score_card.swing_score,
            top_positive_drivers=modular_signal.drivers.top_positive,
            top_negative_drivers=modular_signal.drivers.top_negative,
            ranking_reason=f"final score={modular_signal.final_score:.1f}, conviction={modular_signal.conviction:.1f}",
            rejection_reason=(
                None if modular_signal.signal_category.endswith("candidate") else "; ".join(modular_signal.rejection_reasons)
            ),
            holding_horizon=modular_signal.horizon,
            risk_flags=list(risk_flags),
            confidence=modular_signal.conviction,
            entry_logic="Enter on pullback to support with volume confirmation",
            invalidation_logic="Exit if trend breaks and catalyst fades",
        )
        category = modular_signal.signal_category
        return SignalResult(
            symbol=score_card.symbol,
            category=category,
            score=score_card.swing_score,
            explanation=explanation,
            sector=sector,
        )

    def build_short_signal(
        self, feature_vector: FeatureVector, risk_flags: list[str], sector: str = ""
    ) -> SignalResult:
        """Score a stock for short/bearish potential from its raw feature vector."""
        short_result = self.short_scorer.score(feature_vector.values)
        final_score  = max(0.0, min(100.0, short_result.total_score))

        modular_signal = self.modular.build_short(
            symbol=feature_vector.symbol,
            as_of=feature_vector.as_of,
            short=short_result,
            risk=RiskPenaltyResult(
                total_penalty=0.0,
                components={},
                flags=list(risk_flags),
                missing_features=[],
            ),
        )

        explanation = SignalExplanation(
            signal_type="short",
            score=final_score,
            top_positive_drivers=modular_signal.drivers.top_positive,
            top_negative_drivers=modular_signal.drivers.top_negative,
            ranking_reason=f"short score={final_score:.1f}",
            rejection_reason=(
                None
                if modular_signal.signal_category == "short_candidate"
                else "; ".join(modular_signal.rejection_reasons)
            ),
            holding_horizon=modular_signal.horizon,
            risk_flags=list(risk_flags),
            confidence=final_score,
            entry_logic="Short on a failed rally or technical breakdown below key support",
            invalidation_logic="Cover immediately if price closes above the short-entry high on volume",
        )
        return SignalResult(
            symbol=feature_vector.symbol,
            category=modular_signal.signal_category,
            score=final_score,
            explanation=explanation,
            sector=sector,
        )
