"""MarketDataProvider backed by the canonical SQLite market-data store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import MarketSnapshot, StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider
from stock_screener_engine.data_sources.schemas import DeliveryTurnoverRecord, OHLCVBar, SecurityMasterRecord
from stock_screener_engine.storage.market_data_store import MarketDataStore


@dataclass(frozen=True)
class CanonicalFreshnessReport:
    passed: bool
    issues: list[str]
    warnings: list[str]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


class SQLiteMarketDataProvider(MarketDataProvider):
    """Read market history and snapshots from the canonical SQLite store.

    Historical bars are adjusted by default so feature research and backtests
    consume split/bonus-adjusted history. Snapshots use unadjusted latest close
    because current trading signals need executable prices.
    """

    def __init__(
        self,
        sqlite_path: str,
        universe: Sequence[str] | None = None,
        venue: str = "NSE",
        adjusted_history: bool = True,
        strict_freshness: bool = False,
        max_staleness_days: int = 3,
        store: MarketDataStore | None = None,
    ) -> None:
        self.store = store or MarketDataStore(sqlite_path)
        self.venue = venue.strip().upper() or "NSE"
        self.adjusted_history = adjusted_history
        self.strict_freshness = strict_freshness
        self.max_staleness_days = max(0, int(max_staleness_days))
        self._universe = [s.strip().upper() for s in universe or [] if s.strip()]

    def get_universe(self) -> list[str]:
        securities = [record for record in self.store.get_security_master() if record.active]
        stored_symbols = [record.symbol for record in securities] or self.store.list_ohlcv_symbols()
        if self._universe:
            stored_set = set(stored_symbols)
            configured_and_stored = [symbol for symbol in self._universe if symbol in stored_set]
            return configured_and_stored or self._universe[:]
        if securities:
            return [record.symbol for record in securities]
        return stored_symbols

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        bars = self.store.get_ohlcv(
            symbol=symbol,
            start=start,
            end=end,
            venue=self.venue,
            interval=interval,
            adjusted=self.adjusted_history,
        )
        return [_bar_to_dict(bar) for bar in bars]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        out: list[StockSnapshot] = []
        security_by_symbol = {
            record.symbol: record
            for record in self.store.get_security_master([s.strip().upper() for s in symbols if s.strip()])
        }
        for symbol in [s.strip().upper() for s in symbols if s.strip()]:
            bars = self.store.get_ohlcv(symbol=symbol, venue=self.venue, interval="1d", adjusted=False)
            if not bars:
                continue
            last = bars[-1]
            security = security_by_symbol.get(symbol)
            delivery = self.store.latest_delivery_turnover(
                symbol=symbol,
                as_of=_parse_bar_date(last),
                venue=self.venue,
            )
            out.append(_snapshot_from_bar(last, security, delivery))
        return out

    def get_market_snapshots(self, symbols: Sequence[str]) -> list[MarketSnapshot]:
        return [
            MarketSnapshot(
                symbol=s.symbol,
                as_of=s.as_of,
                sector=s.sector,
                exchange=self.venue,
                close=s.close,
                volume=s.volume,
                delivery_ratio=s.delivery_ratio,
            )
            for s in self.get_snapshots(symbols)
        ]

    def get_company_metadata(self, symbols: Sequence[str]) -> dict[str, dict[str, object]]:
        return self.store.company_metadata(symbols)

    def get_freshness_report(self, symbols: Sequence[str] | None = None) -> dict[str, object]:
        symbols = [s.strip().upper() for s in symbols or self.get_universe() if s.strip()]
        today = date.today()
        expected = self._expected_latest_trading_day(today)
        latest_by_symbol: dict[str, str | None] = {}
        stale_symbols: list[str] = []
        missing_symbols: list[str] = []
        for symbol in symbols:
            bars = self.store.get_ohlcv(symbol=symbol, venue=self.venue, interval="1d", adjusted=False)
            if not bars:
                latest_by_symbol[symbol] = None
                missing_symbols.append(symbol)
                continue
            latest = _parse_bar_date(bars[-1])
            latest_by_symbol[symbol] = latest.isoformat()
            if latest < expected - timedelta(days=self.max_staleness_days):
                stale_symbols.append(symbol)

        issues: list[str] = []
        warnings: list[str] = []
        if missing_symbols:
            issues.append(f"Missing canonical bars for: {', '.join(missing_symbols[:10])}")
        if stale_symbols:
            message = (
                f"Stale canonical bars versus expected {expected.isoformat()}: "
                f"{', '.join(stale_symbols[:10])}"
            )
            if self.strict_freshness:
                issues.append(message)
            else:
                warnings.append(message)

        report = CanonicalFreshnessReport(
            passed=not issues,
            issues=issues,
            warnings=warnings,
            metrics={
                "expected_latest_trading_day": expected.isoformat(),
                "strict_freshness": self.strict_freshness,
                "max_staleness_days": self.max_staleness_days,
                "symbol_count": len(symbols),
                "missing_symbol_count": len(missing_symbols),
                "stale_symbol_count": len(stale_symbols),
                "latest_by_symbol": latest_by_symbol,
            },
        )
        return report.to_dict()

    def _expected_latest_trading_day(self, today: date) -> date:
        start = today - timedelta(days=max(10, self.max_staleness_days * 4 + 5))
        sessions = self.store.get_market_sessions(self.venue, start=start, end=today)
        trading_days = [session.session_date for session in sessions if session.is_trading_day]
        if trading_days:
            return trading_days[-1]
        day = today
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day

    def close(self) -> None:
        self.store.close()


def _bar_to_dict(bar: OHLCVBar) -> dict[str, object]:
    return {
        "date": bar.ts,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": int(bar.volume),
    }


def _snapshot_from_bar(
    bar: OHLCVBar,
    security: SecurityMasterRecord | None,
    delivery: DeliveryTurnoverRecord | None = None,
) -> StockSnapshot:
    return StockSnapshot(
        symbol=bar.symbol,
        as_of=_parse_bar_date(bar),
        sector=security.sector if security is not None else "Unknown",
        close=bar.close,
        volume=bar.volume,
        delivery_ratio=delivery.delivery_pct / 100.0 if delivery is not None else 0.0,
        pe_ratio=0.0,
        roe=0.0,
        debt_to_equity=0.0,
        earnings_growth=0.0,
        free_cash_flow_margin=0.0,
        promoter_holding_change=0.0,
        insider_activity_score=0.0,
    )


def _parse_bar_date(bar: OHLCVBar) -> date:
    return date.fromisoformat(bar.ts[:10])
