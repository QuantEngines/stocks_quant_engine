"""Model calibration dataset and evaluation metrics.

Implements quantile IC, turnover, and decay diagnostics over panel-style
prediction data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationRow:
    as_of: str
    symbol: str
    score: float
    returns: dict[int, float]


@dataclass(frozen=True)
class CalibrationReport:
    quantile_ic: dict[int, float]
    turnover_top_quantile: dict[int, float]
    decay: dict[int, float]


@dataclass(frozen=True)
class TunedWeightPriors:
    long_term: dict[str, float]
    swing: dict[str, float]
    diagnostics: dict[str, float]


class CalibrationDatasetBuilder:
    def build(
        self,
        scores_by_date_symbol: dict[tuple[str, str], float],
        returns_by_date_symbol_horizon: dict[tuple[str, str, int], float],
        horizons: list[int],
    ) -> list[CalibrationRow]:
        rows: list[CalibrationRow] = []
        for (as_of, symbol), score in scores_by_date_symbol.items():
            hmap = {
                h: float(returns_by_date_symbol_horizon.get((as_of, symbol, h), 0.0))
                for h in horizons
            }
            rows.append(CalibrationRow(as_of=as_of, symbol=symbol, score=float(score), returns=hmap))
        rows.sort(key=lambda r: (r.as_of, r.symbol))
        return rows


class ModelCalibrator:
    def evaluate(self, rows: list[CalibrationRow], horizons: list[int]) -> CalibrationReport:
        by_date: dict[str, list[CalibrationRow]] = {}
        for row in rows:
            by_date.setdefault(row.as_of, []).append(row)

        quantile_ic: dict[int, float] = {}
        decay: dict[int, float] = {}
        turnover: dict[int, float] = {}

        for h in horizons:
            day_ics: list[float] = []
            top_sets: list[set[str]] = []
            for as_of in sorted(by_date.keys()):
                panel = by_date[as_of]
                if len(panel) < 5:
                    continue
                preds = [r.score for r in panel]
                rets = [r.returns.get(h, 0.0) for r in panel]
                day_ics.append(_spearman(preds, rets))

                q = max(1, len(panel) // 5)
                top = sorted(panel, key=lambda r: r.score, reverse=True)[:q]
                top_sets.append({r.symbol for r in top})

            quantile_ic[h] = sum(day_ics) / len(day_ics) if day_ics else 0.0
            decay[h] = quantile_ic[h]
            turnover[h] = _avg_turnover(top_sets)

        return CalibrationReport(
            quantile_ic=quantile_ic,
            turnover_top_quantile=turnover,
            decay=decay,
        )


class WeightPriorAutoTuner:
    """Auto-tune long/swing weight priors from calibration diagnostics."""

    def tune(
        self,
        report: CalibrationReport,
        long_term_weights: dict[str, float],
        swing_weights: dict[str, float],
        learning_rate: float = 0.30,
    ) -> TunedWeightPriors:
        ic = _avg(report.quantile_ic.values())
        decay = _avg(report.decay.values())
        turnover = _avg(report.turnover_top_quantile.values())

        persistence = _clamp01((ic + decay + 1.0) / 2.0)
        churn = _clamp01(turnover)

        long_term = self._rescale(
            long_term_weights,
            stable_keys={
                "profitability_quality",
                "balance_sheet_health",
                "cash_flow_quality",
                "governance_proxy",
            },
            tactical_keys={"event_catalyst", "regime_tailwind", "growth_quality"},
            persistence=persistence,
            churn=churn,
            learning_rate=learning_rate,
        )

        swing = self._rescale(
            swing_weights,
            stable_keys={"trend_strength", "volatility_regime", "volume_confirmation"},
            tactical_keys={"event_catalyst", "sentiment_score", "momentum_strength"},
            persistence=persistence,
            churn=churn,
            learning_rate=learning_rate,
        )

        return TunedWeightPriors(
            long_term=long_term,
            swing=swing,
            diagnostics={
                "mean_ic": ic,
                "mean_decay": decay,
                "mean_turnover": turnover,
                "persistence": persistence,
                "churn": churn,
            },
        )

    def _rescale(
        self,
        weights: dict[str, float],
        stable_keys: set[str],
        tactical_keys: set[str],
        persistence: float,
        churn: float,
        learning_rate: float,
    ) -> dict[str, float]:
        tuned: dict[str, float] = {}

        stable_mult = 1.0 + learning_rate * (persistence - churn)
        tactical_mult = 1.0 + learning_rate * (churn - persistence)

        for key, value in weights.items():
            base = float(value)
            if key in stable_keys:
                tuned[key] = max(0.0, base * stable_mult)
            elif key in tactical_keys:
                tuned[key] = max(0.0, base * tactical_mult)
            else:
                tuned[key] = max(0.0, base)

        total_in = sum(float(v) for v in weights.values())
        total_out = sum(tuned.values())
        if total_in > 0 and total_out > 0:
            scale = total_in / total_out
            tuned = {k: v * scale for k, v in tuned.items()}

        return tuned


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _ranks(x)
    ry = _ranks(y)
    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    denom = (var_x * var_y) ** 0.5
    if denom <= 1e-12:
        return 0.0
    return cov / denom


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _avg_turnover(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 0.0
    vals: list[float] = []
    for prev, nxt in zip(sets[:-1], sets[1:]):
        if not prev and not nxt:
            vals.append(0.0)
            continue
        inter = len(prev & nxt)
        denom = max(1, len(prev | nxt))
        vals.append(1.0 - (inter / denom))
    return sum(vals) / len(vals)


def _avg(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
