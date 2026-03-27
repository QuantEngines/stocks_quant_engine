"""Tests for UniverseSelector."""

from __future__ import annotations
from datetime import date

from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.core.universe import UniverseSelector


def _make_snapshot(symbol: str, volume: float, sector: str = "IT") -> StockSnapshot:
    return StockSnapshot(
        symbol=symbol,
        as_of=date(2024, 1, 15),
        sector=sector,
        close=500.0,
        volume=volume,
        delivery_ratio=0.50,
        pe_ratio=20.0,
        roe=0.18,
        debt_to_equity=0.30,
        earnings_growth=0.15,
        free_cash_flow_margin=0.12,
        promoter_holding_change=0.01,
        insider_activity_score=0.40,
    )


class TestUniverseSelector:
    def test_filters_below_min_liquidity(self) -> None:
        # close=500.0; liquidity = close×volume
        # LIQUID:   500 × 5_000_000 = 2.5 B  >= 1 B ✓
        # ILLIQUID: 500 × 100       = 50 K    < 1 B ✗
        selector = UniverseSelector(min_liquidity=1_000_000_000)
        snapshots = [
            _make_snapshot("LIQUID", 5_000_000),
            _make_snapshot("ILLIQUID", 100),
        ]
        selected = selector.select(snapshots)
        symbols = {s.symbol for s in selected}
        assert "LIQUID" in symbols
        assert "ILLIQUID" not in symbols

    def test_all_above_threshold_pass(self) -> None:
        # close=500, volume=2_000_000 → liquidity = 1 B >= 1 M
        selector = UniverseSelector(min_liquidity=1_000_000)
        snapshots = [_make_snapshot(f"SYM{i}", 2_000_000) for i in range(5)]
        selected = selector.select(snapshots)
        assert len(selected) == 5

    def test_empty_input_returns_empty(self) -> None:
        selector = UniverseSelector(min_liquidity=1_000_000)
        assert selector.select([]) == []

    def test_all_below_threshold_returns_empty(self) -> None:
        # close=500, volume=1_000 → 500_000 < 10_000_000_000
        selector = UniverseSelector(min_liquidity=10_000_000_000)
        snapshots = [_make_snapshot("LOW", 1_000)]
        assert selector.select(snapshots) == []

    def test_zero_min_liquidity_passes_all(self) -> None:
        selector = UniverseSelector(min_liquidity=0)
        snapshots = [_make_snapshot(f"S{i}", float(i + 1)) for i in range(3)]
        selected = selector.select(snapshots)
        assert len(selected) == 3
