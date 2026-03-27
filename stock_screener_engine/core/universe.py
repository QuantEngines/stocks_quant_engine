"""Universe selection logic independent from data provider internals."""

from __future__ import annotations

from collections.abc import Iterable

from stock_screener_engine.core.entities import StockSnapshot


class UniverseSelector:
    def __init__(self, min_liquidity: float) -> None:
        self._min_liquidity = min_liquidity

    def select(self, snapshots: Iterable[StockSnapshot]) -> list[StockSnapshot]:
        selected: list[StockSnapshot] = []
        for snap in snapshots:
            liquidity = snap.close * snap.volume
            if liquidity >= self._min_liquidity:
                selected.append(snap)
        return selected
