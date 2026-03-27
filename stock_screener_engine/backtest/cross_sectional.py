"""Cross-sectional backtest helpers for ranking efficacy.

Metrics produced
----------------
* hit_rate          — fraction of top-quintile stocks with positive forward return
* avg_return        — mean forward return across panel
* max_drawdown      — drawdown of a cumulative long strategy
* quantile_spread   — top-quintile avg return minus bottom-quintile avg return
* information_ratio — mean(returns_ranked_high) / std(returns), annualised
* ic                — Spearman rank-IC between predicted rank and realised return
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrossSectionalStats:
    hit_rate: float
    avg_return: float
    max_drawdown: float
    quantile_spread: float
    information_ratio: float = 0.0
    ic: float = 0.0
    ic_t_stat: float = 0.0


class CrossSectionalBacktester:
    """Evaluate signal quality from a list of (score, forward_return) pairs.

    ``evaluate_panel`` is the preferred entry point — it accepts explicit
    score/return vectors and avoids the ambiguity of the legacy interface.
    """

    def evaluate(self, returns_by_rank: list[float]) -> CrossSectionalStats:
        """Legacy interface: returns ordered from highest-score to lowest.

        Prefer ``evaluate_panel`` for new code.
        """
        n = len(returns_by_rank)
        if n < 2:
            return CrossSectionalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        scores = list(range(n, 0, -1))  # implicit rank: first element = highest score
        return self.evaluate_panel(scores, returns_by_rank)

    def evaluate_panel(
        self,
        scores: list[float],
        forward_returns: list[float],
    ) -> CrossSectionalStats:
        """Evaluate a single cross-section of scores vs realised forward returns.

        Parameters
        ----------
        scores:          predicted scores (higher = better expected return)
        forward_returns: realised returns over the target horizon (same length)
        """
        n = len(scores)
        assert len(forward_returns) == n, "scores and forward_returns must be same length"
        if n < 2:
            return CrossSectionalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        paired = sorted(zip(scores, forward_returns), key=lambda x: x[0], reverse=True)
        rets   = [r for _, r in paired]

        hits        = sum(r > 0 for r in rets)
        hit_rate    = hits / n
        avg_return  = sum(rets) / n
        max_dd      = _max_drawdown(rets)

        q            = max(1, n // 5)
        top_avg      = sum(rets[:q]) / q
        bottom_avg   = sum(rets[-q:]) / q
        q_spread     = top_avg - bottom_avg

        ic, ic_t     = _spearman_ic(scores, forward_returns)
        ir           = _information_ratio(rets[:q])

        return CrossSectionalStats(
            hit_rate=hit_rate,
            avg_return=avg_return,
            max_drawdown=max_dd,
            quantile_spread=q_spread,
            information_ratio=ir,
            ic=ic,
            ic_t_stat=ic_t,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_drawdown(returns: list[float]) -> float:
    cumulative = 0.0
    peak       = 0.0
    max_dd     = 0.0
    for r in returns:
        cumulative += r
        peak        = max(peak, cumulative)
        max_dd      = max(max_dd, peak - cumulative)
    return max_dd


def _spearman_ic(scores: list[float], returns: list[float]) -> tuple[float, float]:
    """Return (spearman_ic, t_statistic)."""
    n = len(scores)
    if n < 4:
        return 0.0, 0.0

    def _rank(lst: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks      = [0.0] * n
        for r, idx in enumerate(sorted_idx):
            ranks[idx] = float(r + 1)
        return ranks

    rs  = _rank(scores)
    rr  = _rank(returns)
    ms  = sum(rs) / n
    mr  = sum(rr) / n
    num = sum((rs[i] - ms) * (rr[i] - mr) for i in range(n))
    ds  = math.sqrt(sum((rs[i] - ms) ** 2 for i in range(n)))
    dr  = math.sqrt(sum((rr[i] - mr) ** 2 for i in range(n)))
    if ds < 1e-12 or dr < 1e-12:
        return 0.0, 0.0
    ic  = num / (ds * dr)
    # t-stat for testing IC != 0
    denom = math.sqrt(max(1e-12, 1.0 - ic ** 2))
    t_stat = ic * math.sqrt(n - 2) / denom
    return ic, t_stat


def _information_ratio(top_returns: list[float]) -> float:
    """Simple annualised information ratio from top-quintile return series."""
    n = len(top_returns)
    if n < 2:
        return 0.0
    mean = sum(top_returns) / n
    var  = sum((r - mean) ** 2 for r in top_returns) / (n - 1)
    std  = math.sqrt(var)
    if std < 1e-12:
        return 0.0
    return mean / std * math.sqrt(252)   # annualise assuming daily obs
