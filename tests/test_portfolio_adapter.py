from __future__ import annotations

from stock_screener_engine.core.entities import SignalExplanation, SignalResult
from stock_screener_engine.execution.portfolio_adapter import (
    PortfolioConstructionAdapter,
    PortfolioConstraints,
)


def _signal(symbol: str, score: float, sector: str = "IT") -> SignalResult:
    return SignalResult(
        symbol=symbol,
        category="long_term_candidate",
        score=score,
        explanation=SignalExplanation(signal_type="long_term", score=score),
        sector=sector,
    )


def test_portfolio_adapter_applies_caps_and_liquidity_filters() -> None:
    adapter = PortfolioConstructionAdapter()
    signals = [
        _signal("AAA", 82.0, "IT"),
        _signal("BBB", 80.0, "IT"),
        _signal("CCC", 78.0, "IT"),
        _signal("DDD", 76.0, "Banking"),
    ]
    plan = adapter.construct(
        ranked_signals=signals,
        sector_by_symbol={"AAA": "IT", "BBB": "IT", "CCC": "IT", "DDD": "Banking"},
        price_by_symbol={"AAA": 100.0, "BBB": 120.0, "CCC": 90.0, "DDD": 200.0},
        volume_by_symbol={"AAA": 2_000_000.0, "BBB": 2_000_000.0, "CCC": 100_000.0, "DDD": 2_000_000.0},
        constraints=PortfolioConstraints(
            max_positions=3,
            max_sector_positions=2,
            min_avg_daily_volume=500_000.0,
            max_single_position_weight=0.60,
            capital_base=1_000_000.0,
        ),
    )

    symbols = {p.symbol for p in plan.positions}
    assert "AAA" in symbols
    assert "BBB" in symbols
    assert "DDD" in symbols
    assert "CCC" not in symbols
    assert abs(sum(p.target_weight for p in plan.positions) - 1.0) < 1e-6


def test_portfolio_adapter_enforces_min_position_notional() -> None:
    adapter = PortfolioConstructionAdapter()
    signals = [_signal("AAA", 60.0, "IT")]
    plan = adapter.construct(
        ranked_signals=signals,
        sector_by_symbol={"AAA": "IT"},
        price_by_symbol={"AAA": 1_000_000.0},
        volume_by_symbol={"AAA": 5_000_000.0},
        constraints=PortfolioConstraints(
            max_positions=1,
            max_sector_positions=1,
            min_avg_daily_volume=500_000.0,
            max_single_position_weight=0.50,
            capital_base=100_000.0,
            min_position_notional=25_000.0,
        ),
    )

    assert not plan.positions
    assert any(r.reason == "position_too_small" for r in plan.rejected)


def test_portfolio_adapter_applies_sector_target_weights() -> None:
    adapter = PortfolioConstructionAdapter()
    signals = [
        _signal("AAA", 90.0, "IT"),
        _signal("BBB", 80.0, "IT"),
        _signal("CCC", 70.0, "Banking"),
    ]
    plan = adapter.construct(
        ranked_signals=signals,
        sector_by_symbol={"AAA": "IT", "BBB": "IT", "CCC": "Banking"},
        price_by_symbol={"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
        volume_by_symbol={"AAA": 2_000_000.0, "BBB": 2_000_000.0, "CCC": 2_000_000.0},
        constraints=PortfolioConstraints(
            max_positions=3,
            max_sector_positions=3,
            min_avg_daily_volume=500_000.0,
            max_single_position_weight=0.90,
            capital_base=1_000_000.0,
            min_position_notional=10_000.0,
            sector_target_weights={"IT": 0.30, "Banking": 0.70},
            sector_target_tolerance=0.0,
        ),
    )

    by_sector: dict[str, float] = {"IT": 0.0, "Banking": 0.0}
    for pos in plan.positions:
        by_sector[pos.sector] = by_sector.get(pos.sector, 0.0) + pos.target_weight

    assert by_sector["Banking"] > by_sector["IT"]
