"""Live invalidation checks for active positions/signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class ActiveSignal:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entered_on: date
    stop_loss_pct: float = 0.08
    max_holding_days: int = 30
    required_thesis_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InvalidationDecision:
    symbol: str
    as_of: date
    invalidated: bool
    reasons: list[str]


class LiveInvalidationMonitor:
    """Evaluate whether a live signal remains valid."""

    def evaluate(
        self,
        signal: ActiveSignal,
        as_of: date,
        latest_price: float,
        active_thesis_flags: list[str] | None = None,
    ) -> InvalidationDecision:
        reasons: list[str] = []

        if latest_price <= 0.0:
            reasons.append("latest price unavailable or invalid")
        else:
            if signal.side == "long":
                stop = signal.entry_price * (1.0 - signal.stop_loss_pct)
                if latest_price <= stop:
                    reasons.append(
                        f"long stop breached: price {latest_price:.2f} <= stop {stop:.2f}"
                    )
            elif signal.side == "short":
                stop = signal.entry_price * (1.0 + signal.stop_loss_pct)
                if latest_price >= stop:
                    reasons.append(
                        f"short stop breached: price {latest_price:.2f} >= stop {stop:.2f}"
                    )
            else:
                reasons.append(f"unsupported side: {signal.side}")

        holding_days = (as_of - signal.entered_on).days
        if holding_days > signal.max_holding_days:
            reasons.append(
                f"holding period exceeded: {holding_days}d > {signal.max_holding_days}d"
            )

        active = set(active_thesis_flags or [])
        missing = [flag for flag in signal.required_thesis_flags if flag not in active]
        if missing:
            reasons.append("thesis flags missing: " + ", ".join(sorted(missing)))

        return InvalidationDecision(
            symbol=signal.symbol,
            as_of=as_of,
            invalidated=bool(reasons),
            reasons=reasons,
        )
