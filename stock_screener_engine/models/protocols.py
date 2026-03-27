"""Structural protocols for the models layer.

Using ``typing.Protocol`` (structural sub-typing) means any callable that
matches the interface satisfies the contract without explicit inheritance.
This allows ML models, ensemble wrappers, or rule-based scorers to be swapped
transparently inside ``LongTermModel`` and ``SwingModel``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stock_screener_engine.core.entities import FeatureVector


@runtime_checkable
class ScorerProtocol(Protocol):
    """Structural interface for any component that scores a ``FeatureVector``."""

    def score(self, fv: FeatureVector) -> tuple[float, dict[str, float]]:
        """Return ``(composite_score, component_scores)`` for the given features.

        ``composite_score`` should be in the range ``[0, 100]``.
        ``component_scores`` is a mapping of component name → contribution value.
        """
        ...
