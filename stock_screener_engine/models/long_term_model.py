"""Long-term scoring model — rule-based by default, ML-swappable via ScorerProtocol."""

from __future__ import annotations

from stock_screener_engine.core.entities import FeatureVector
from stock_screener_engine.core.scoring import LongTermScorer, LongTermWeights
from stock_screener_engine.models.protocols import ScorerProtocol


class LongTermModel:
    """Wraps a ``ScorerProtocol`` implementation for long-term scoring.

    Pass a custom ``scorer`` to swap the rule-based implementation for an ML
    model without touching any downstream code.
    """

    def __init__(self, scorer: ScorerProtocol | None = None) -> None:
        self._scorer: ScorerProtocol = scorer or LongTermScorer()

    @classmethod
    def with_weights(cls, weights: LongTermWeights) -> "LongTermModel":
        """Convenience factory to build a model with custom scoring weights."""
        return cls(scorer=LongTermScorer(weights=weights))

    def predict_score(self, features: FeatureVector) -> tuple[float, dict[str, float]]:
        return self._scorer.score(features)
