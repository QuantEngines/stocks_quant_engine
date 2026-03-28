"""Pipeline for evaluating and storing live signal invalidations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.monitoring.live_invalidation import (
    ActiveSignal,
    LiveInvalidationMonitor,
)
from stock_screener_engine.storage.local_files import LocalFileStorage


class LiveInvalidationPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.monitor = LiveInvalidationMonitor()
        self.storage = LocalFileStorage(settings.storage.root_dir)

    def run(
        self,
        active_signals: list[ActiveSignal],
        latest_price_by_symbol: dict[str, float],
        thesis_flags_by_symbol: dict[str, list[str]] | None = None,
        as_of: date | None = None,
        run_label: str | None = None,
    ) -> dict:
        as_of = as_of or date.today()
        run_label = run_label or as_of.isoformat()
        thesis_flags_by_symbol = thesis_flags_by_symbol or {}

        decisions = []
        for signal in active_signals:
            decision = self.monitor.evaluate(
                signal=signal,
                as_of=as_of,
                latest_price=float(latest_price_by_symbol.get(signal.symbol, 0.0)),
                active_thesis_flags=thesis_flags_by_symbol.get(signal.symbol, []),
            )
            decisions.append(decision)

        invalidated = [d for d in decisions if d.invalidated]
        payload = {
            "as_of": as_of.isoformat(),
            "total": len(decisions),
            "invalidated": len(invalidated),
            "active": len(decisions) - len(invalidated),
            "rows": [asdict(d) | {"as_of": d.as_of.isoformat()} for d in decisions],
        }

        self.storage.save_json(
            payload=payload,
            filename=f"live_invalidation_{run_label}.json",
            subdir="signals",
        )
        self.storage.save_rows_csv(
            rows=payload["rows"],
            filename=f"live_invalidation_{run_label}.csv",
            subdir="signals",
        )
        return payload
