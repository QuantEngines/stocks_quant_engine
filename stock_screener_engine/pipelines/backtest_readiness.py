"""Backtest-readiness diagnostics for the canonical market data store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence, cast

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.calendar.market_calendar import MarketCalendar
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


@dataclass(frozen=True)
class BacktestReadinessThresholds:
    min_history_years: float = 5.0
    min_history_rows: int | None = None
    min_ohlcv_coverage: float = 0.95
    require_fundamentals: bool = False

    def resolved_min_rows(self) -> int:
        if self.min_history_rows is not None:
            return self.min_history_rows
        return max(1, int(self.min_history_years * 200))


class BacktestReadinessPipeline:
    """Validate whether stored data can support meaningful historical tests."""

    def __init__(
        self,
        settings: AppSettings,
        store: MarketDataStore | None = None,
        file_store: LocalFileStorage | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MarketDataStore(settings.storage.sqlite_path)
        self.file_store = file_store or LocalFileStorage(settings.storage.root_dir)

    def run(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str = "1d",
        horizons: Sequence[int] = (5, 20, 60),
        thresholds: BacktestReadinessThresholds = BacktestReadinessThresholds(),
    ) -> dict[str, object]:
        requested = [s.strip().upper() for s in symbols if s.strip()]
        min_rows = thresholds.resolved_min_rows()
        expected_trading_days = len(MarketCalendar(venue=self.settings.runtime_data.canonical_venue).trading_days(start, end))

        per_symbol: dict[str, dict[str, object]] = {}
        issues: list[str] = []
        warnings: list[str] = []
        if not requested:
            issues.append("No symbols requested for backtest-readiness checks")

        security_map = {record.symbol: record for record in self.store.get_security_master(requested)}
        ready_symbols = 0
        forward_label_totals = {str(h): 0 for h in horizons}

        for symbol in requested:
            bars = self.store.get_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                venue=self.settings.runtime_data.canonical_venue,
                interval=interval,
                adjusted=self.settings.runtime_data.canonical_adjusted_history,
            )
            row_count = len(bars)
            first_ts = bars[0].ts if bars else None
            last_ts = bars[-1].ts if bars else None
            coverage_vs_weekdays = round(row_count / expected_trading_days, 4) if expected_trading_days else 0.0
            forward_counts = {str(h): max(0, row_count - int(h)) for h in horizons}
            for horizon, count in forward_counts.items():
                forward_label_totals[horizon] += int(count)

            record = security_map.get(symbol)
            unknown_sector = record is None or not record.sector.strip() or record.sector.strip().lower() == "unknown"
            if unknown_sector:
                warnings.append(f"{symbol}: missing sector metadata in security master")

            status = "ready"
            if row_count == 0:
                status = "missing_ohlcv"
                issues.append(f"{symbol}: no OHLCV bars in requested window")
            elif row_count < min_rows:
                status = "insufficient_history"
                issues.append(f"{symbol}: only {row_count} OHLCV rows; need at least {min_rows}")
            elif not all(count > 0 for count in forward_counts.values()):
                status = "insufficient_forward_labels"
                issues.append(f"{symbol}: insufficient bars to compute all requested forward-return horizons")
            else:
                ready_symbols += 1

            per_symbol[symbol] = {
                "status": status,
                "row_count": row_count,
                "first_bar": first_ts,
                "last_bar": last_ts,
                "coverage_vs_weekday_calendar": coverage_vs_weekdays,
                "forward_return_labels": forward_counts,
                "security_master": {
                    "available": record is not None,
                    "sector": record.sector if record else "Unknown",
                    "industry": record.industry if record else "Unknown",
                    "company_name": record.company_name if record else "",
                },
                "corporate_action_records": len(
                    self.store.get_corporate_actions(
                        symbol=symbol,
                        venue=self.settings.runtime_data.canonical_venue,
                    )
                ),
            }

        coverage = self.store.coverage_summary(symbols=requested, start=start, end=end, interval=interval)
        ohlcv_coverage = _coverage_value(coverage)
        if ohlcv_coverage < thresholds.min_ohlcv_coverage:
            issues.append(
                f"OHLCV symbol coverage {ohlcv_coverage:.1%} below readiness threshold "
                f"{thresholds.min_ohlcv_coverage:.1%}"
            )

        financials_coverage = self.store.financial_statement_coverage(
            requested,
            as_of=end,
            venue=self.settings.runtime_data.canonical_venue,
        )
        valuation_coverage = self.store.equity_valuation_coverage(
            requested,
            as_of=end,
            venue=self.settings.runtime_data.canonical_venue,
        )
        shareholding_coverage = self.store.shareholding_coverage(
            requested,
            as_of=end,
            venue=self.settings.runtime_data.canonical_venue,
        )
        if _coverage_value(financials_coverage) < 1.0:
            message = "Financial statement coverage is incomplete; long-term backtests will be technical-only for missing stocks"
            if thresholds.require_fundamentals:
                issues.append(message)
            else:
                warnings.append(message)
        if _coverage_value(valuation_coverage) < 1.0:
            warnings.append("Valuation coverage is incomplete; valuation factors will be unavailable for missing stocks")
        if _coverage_value(shareholding_coverage) < 1.0:
            warnings.append("Shareholding coverage is incomplete; ownership/governance factors will be unavailable for missing stocks")
        if not self.settings.runtime_data.canonical_adjusted_history:
            warnings.append("Canonical adjusted history is disabled; split/bonus-aware backtests should enable adjusted history")

        report = {
            "pipeline": "backtest_readiness",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "passed": not issues,
            "thresholds": {
                "min_history_years": thresholds.min_history_years,
                "min_history_rows": min_rows,
                "min_ohlcv_coverage": thresholds.min_ohlcv_coverage,
                "require_fundamentals": thresholds.require_fundamentals,
            },
            "summary": {
                "symbols_requested": len(requested),
                "symbols_ready": ready_symbols,
                "ohlcv_coverage": ohlcv_coverage,
                "expected_weekday_sessions": expected_trading_days,
                "adjusted_history_enabled": self.settings.runtime_data.canonical_adjusted_history,
                "forward_return_label_totals": forward_label_totals,
            },
            "coverage": {
                "ohlcv": coverage,
                "financials": financials_coverage,
                "valuation": valuation_coverage,
                "shareholding": shareholding_coverage,
            },
            "per_symbol": per_symbol,
            "issues": issues,
            "warnings": warnings,
        }
        self.file_store.save_json(report, filename="backtest_readiness_report.json", subdir="quality")
        return report

    def close(self) -> None:
        self.store.close()


def _coverage_value(report: Mapping[str, object]) -> float:
    return float(cast(Any, report.get("coverage") or 0.0))
