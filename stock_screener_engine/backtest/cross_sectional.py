"""Cross-sectional backtest helpers for ranking efficacy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossSectionalStats:
    hit_rate: float
    avg_return: float
    max_drawdown: float
    quantile_spread: float


class CrossSectionalBacktester:
    def evaluate(self, returns_by_rank: list[float]) -> CrossSectionalStats:
        if not returns_by_rank:
            return CrossSectionalStats(0.0, 0.0, 0.0, 0.0)
        hits = sum(r > 0 for r in returns_by_rank)
        hit_rate = hits / len(returns_by_rank)
        avg_return = sum(returns_by_rank) / len(returns_by_rank)

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for r in returns_by_rank:
            cumulative += r
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        q = max(1, len(returns_by_rank) // 5)
        top = sum(sorted(returns_by_rank, reverse=True)[:q]) / q
        bottom = sum(sorted(returns_by_rank)[:q]) / q
        quantile_spread = top - bottom

        return CrossSectionalStats(hit_rate, avg_return, max_drawdown, quantile_spread)
