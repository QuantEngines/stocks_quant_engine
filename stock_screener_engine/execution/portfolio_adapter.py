"""Portfolio construction adapter layer for ranked research signals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from stock_screener_engine.core.entities import SignalResult


@dataclass(frozen=True)
class PortfolioConstraints:
    max_positions: int = 12
    max_sector_positions: int = 3
    min_avg_daily_volume: float = 1_000_000.0
    max_single_position_weight: float = 0.12
    capital_base: float = 1_000_000.0
    min_position_notional: float = 25_000.0
    sector_target_weights: dict[str, float] = field(default_factory=dict)
    sector_target_tolerance: float = 0.05


@dataclass(frozen=True)
class PortfolioPosition:
    symbol: str
    sector: str
    target_weight: float
    target_notional: float
    target_shares: int
    signal_score: float


@dataclass(frozen=True)
class PortfolioRejection:
    symbol: str
    reason: str


@dataclass(frozen=True)
class PortfolioPlan:
    positions: list[PortfolioPosition] = field(default_factory=list)
    rejected: list[PortfolioRejection] = field(default_factory=list)


class PortfolioConstructionAdapter:
    def construct(
        self,
        ranked_signals: list[SignalResult],
        sector_by_symbol: dict[str, str],
        price_by_symbol: dict[str, float],
        volume_by_symbol: dict[str, float],
        constraints: PortfolioConstraints,
    ) -> PortfolioPlan:
        candidates = [
            s for s in ranked_signals
            if s.category.endswith("candidate")
        ]
        candidates.sort(key=lambda s: (s.score, s.symbol), reverse=True)

        selected: list[SignalResult] = []
        rejected: list[PortfolioRejection] = []
        sector_counts: dict[str, int] = defaultdict(int)

        for signal in candidates:
            if len(selected) >= constraints.max_positions:
                rejected.append(PortfolioRejection(signal.symbol, "max_positions_reached"))
                continue

            volume = float(volume_by_symbol.get(signal.symbol, 0.0))
            if volume < constraints.min_avg_daily_volume:
                rejected.append(PortfolioRejection(signal.symbol, "below_liquidity_threshold"))
                continue

            sector = sector_by_symbol.get(signal.symbol, "UNKNOWN") or "UNKNOWN"
            if sector_counts[sector] >= constraints.max_sector_positions:
                rejected.append(PortfolioRejection(signal.symbol, "sector_cap_reached"))
                continue

            price = float(price_by_symbol.get(signal.symbol, 0.0))
            if price <= 0:
                rejected.append(PortfolioRejection(signal.symbol, "invalid_price"))
                continue

            sector_counts[sector] += 1
            selected.append(signal)

        if not selected:
            return PortfolioPlan(positions=[], rejected=rejected)

        raw_scores = [max(0.01, s.score) for s in selected]
        total = sum(raw_scores)
        weights = [x / total for x in raw_scores]
        sectors = [sector_by_symbol.get(s.symbol, "UNKNOWN") or "UNKNOWN" for s in selected]
        weights = _apply_sector_targets(
            weights=weights,
            sectors=sectors,
            target_weights=constraints.sector_target_weights,
            tolerance=constraints.sector_target_tolerance,
        )
        capped = _cap_weights(weights, constraints.max_single_position_weight)

        positions: list[PortfolioPosition] = []
        for signal, weight in zip(selected, capped):
            price = float(price_by_symbol.get(signal.symbol, 0.0))
            notional = constraints.capital_base * weight
            shares = int(notional / price) if price > 0 else 0
            actual_notional = shares * price
            if shares <= 0:
                rejected.append(PortfolioRejection(signal.symbol, "position_too_small"))
                continue
            if actual_notional < constraints.min_position_notional:
                rejected.append(PortfolioRejection(signal.symbol, "below_min_position_notional"))
                continue
            positions.append(
                PortfolioPosition(
                    symbol=signal.symbol,
                    sector=sector_by_symbol.get(signal.symbol, "UNKNOWN") or "UNKNOWN",
                    target_weight=weight,
                    target_notional=actual_notional,
                    target_shares=shares,
                    signal_score=signal.score,
                )
            )

        positions.sort(key=lambda p: (p.target_weight, p.signal_score, p.symbol), reverse=True)
        return PortfolioPlan(positions=positions, rejected=rejected)


def _cap_weights(weights: list[float], cap: float) -> list[float]:
    if not weights:
        return []
    cap = max(0.01, min(1.0, cap))
    adjusted = list(weights)

    for _ in range(5):
        overflow = 0.0
        under_idx: list[int] = []
        for i, w in enumerate(adjusted):
            if w > cap:
                overflow += (w - cap)
                adjusted[i] = cap
            else:
                under_idx.append(i)

        if overflow <= 1e-9 or not under_idx:
            break

        under_total = sum(adjusted[i] for i in under_idx)
        if under_total <= 1e-12:
            even = overflow / len(under_idx)
            for i in under_idx:
                adjusted[i] += even
        else:
            for i in under_idx:
                adjusted[i] += overflow * (adjusted[i] / under_total)

    s = sum(adjusted)
    if s <= 1e-12:
        return [1.0 / len(adjusted)] * len(adjusted)
    return [w / s for w in adjusted]


def _apply_sector_targets(
    weights: list[float],
    sectors: list[str],
    target_weights: Mapping[str, float],
    tolerance: float,
) -> list[float]:
    if not weights or not target_weights:
        return list(weights)

    targets = {str(k): max(0.0, float(v)) for k, v in target_weights.items()}
    t_sum = sum(targets.values())
    if t_sum <= 1e-12:
        return list(weights)
    targets = {k: v / t_sum for k, v in targets.items()}

    current: dict[str, float] = defaultdict(float)
    for w, sector in zip(weights, sectors):
        current[sector] += w

    adjusted = list(weights)
    tol = max(0.0, float(tolerance))
    for sector, target in targets.items():
        cur = current.get(sector, 0.0)
        if cur <= 1e-12 or abs(cur - target) <= tol:
            continue
        factor = target / cur
        for i, sec in enumerate(sectors):
            if sec == sector:
                adjusted[i] *= factor

    s = sum(adjusted)
    if s <= 1e-12:
        return list(weights)
    return [w / s for w in adjusted]
