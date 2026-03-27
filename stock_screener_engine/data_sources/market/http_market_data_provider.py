"""MarketDataProvider implementation backed by free NSE HTTP endpoints."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider
from stock_screener_engine.data_sources.exchange.nse_http_adapter import NSEHTTPAdapter

_DEFAULT_UNIVERSE = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
    "HINDUNILVR",
    "BHARTIARTL",
]


class NSEHTTPMarketDataProvider(MarketDataProvider):
    """Uses public NSE endpoints for historical bars and snapshot construction."""

    def __init__(self, universe: list[str] | None = None, adapter: NSEHTTPAdapter | None = None) -> None:
        self._universe = universe[:] if universe else _DEFAULT_UNIVERSE[:]
        self._adapter = adapter or NSEHTTPAdapter()

    def get_universe(self) -> list[str]:
        return self._universe[:]

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        bars = self._adapter.fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
        return [
            {
                "date": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": int(bar.volume),
            }
            for bar in bars
        ]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        out: list[StockSnapshot] = []
        today = date.today()
        lookback = today - timedelta(days=35)

        for symbol in symbols:
            bars = self.get_historical(symbol=symbol, interval="1d", start=lookback, end=today)
            if not bars:
                continue
            closes = [float(r.get("close", 0.0) or 0.0) for r in bars]
            vols = [float(r.get("volume", 0.0) or 0.0) for r in bars]
            last = bars[-1]
            close = float(last.get("close", 0.0) or 0.0)
            volume = float(last.get("volume", 0.0) or 0.0)
            avg_20 = sum(vols[-20:]) / max(1, min(20, len(vols)))
            mom = 0.0
            if len(closes) > 20 and closes[-21] > 0:
                mom = (closes[-1] - closes[-21]) / closes[-21]

            out.append(
                StockSnapshot(
                    symbol=symbol,
                    as_of=today,
                    sector="Unknown",
                    close=close,
                    volume=volume,
                    delivery_ratio=0.5,
                    pe_ratio=15.0,
                    roe=max(0.02, min(0.35, 0.1 + 0.4 * mom)),
                    debt_to_equity=max(0.05, min(2.0, 1.0 - mom)),
                    earnings_growth=max(-0.2, min(0.6, mom)),
                    free_cash_flow_margin=max(-0.2, min(0.4, (volume / max(1.0, avg_20) - 1.0) * 0.05)),
                    promoter_holding_change=0.0,
                    insider_activity_score=max(-1.0, min(1.0, mom * 2.0)),
                )
            )
        return out
