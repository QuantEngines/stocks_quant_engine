"""Universe selection — accepts both MarketSnapshot and StockSnapshot."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

from stock_screener_engine.core.entities import MarketSnapshot, StockSnapshot

AnySnapshot = Union[MarketSnapshot, StockSnapshot]


class UniverseSelector:
    """Filter snapshots by minimum liquidity (close * volume).

    Accepts either the granular ``MarketSnapshot`` or the legacy
    ``StockSnapshot`` so callers can upgrade incrementally.
    """

    def __init__(self, min_liquidity: float) -> None:
        self._min_liquidity = min_liquidity

    def select(self, snapshots: Iterable[AnySnapshot]) -> list[AnySnapshot]:
        selected: list[AnySnapshot] = []
        for snap in snapshots:
            liquidity = snap.close * snap.volume
            if liquidity >= self._min_liquidity:
                selected.append(snap)
        return selected

    def select_symbols(self, snapshots: Iterable[AnySnapshot]) -> list[str]:
        """Convenience method returning only the accepted symbols."""
        return [s.symbol for s in self.select(snapshots)]
