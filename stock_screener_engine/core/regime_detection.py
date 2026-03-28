"""Market regime detection helpers.

The detector turns index and breadth observations into a continuous regime
score and a discrete label that downstream scorers can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class RegimeSnapshot:
    label: str
    score: float
    momentum: float
    realized_volatility: float
    breadth: float


@dataclass(frozen=True)
class RegimeThresholds:
    bull_threshold: float = 0.25
    bear_threshold: float = -0.25


class RegimeDetector:
    def __init__(self, thresholds: RegimeThresholds = RegimeThresholds()) -> None:
        self.thresholds = thresholds

    def detect(self, index_bars: list[dict], universe_returns: list[float] | None = None) -> RegimeSnapshot:
        closes = [float(row.get("close", 0.0)) for row in index_bars if float(row.get("close", 0.0)) > 0.0]
        if len(closes) < 3:
            return RegimeSnapshot(
                label="neutral",
                score=0.0,
                momentum=0.0,
                realized_volatility=0.0,
                breadth=0.5,
            )

        returns = self._returns_from_closes(closes)
        realized_vol = self._annualized_volatility(returns)
        momentum = (closes[-1] / closes[max(0, len(closes) - 21)] - 1.0) if len(closes) > 21 else (closes[-1] / closes[0] - 1.0)

        if universe_returns:
            positive = sum(1 for r in universe_returns if r > 0.0)
            breadth = positive / len(universe_returns)
        else:
            positive = sum(1 for r in returns if r > 0.0)
            breadth = positive / len(returns)

        momentum_component = self._clip(momentum / 0.12, -1.0, 1.0)
        volatility_component = self._clip((realized_vol - 0.18) / 0.22, -1.0, 1.0)
        breadth_component = self._clip((breadth - 0.5) / 0.5, -1.0, 1.0)

        score = 0.50 * momentum_component + 0.30 * breadth_component - 0.20 * volatility_component
        label = self._label(score)

        return RegimeSnapshot(
            label=label,
            score=score,
            momentum=momentum,
            realized_volatility=realized_vol,
            breadth=breadth,
        )

    def _label(self, score: float) -> str:
        if score >= self.thresholds.bull_threshold:
            return "bull"
        if score <= self.thresholds.bear_threshold:
            return "bear"
        return "neutral"

    @staticmethod
    def _returns_from_closes(closes: list[float]) -> list[float]:
        rows: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            cur = closes[i]
            if prev <= 0:
                continue
            rows.append(cur / prev - 1.0)
        return rows or [0.0]

    @staticmethod
    def _annualized_volatility(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return sqrt(max(var, 0.0)) * sqrt(252.0)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
