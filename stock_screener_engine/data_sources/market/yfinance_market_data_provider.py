"""MarketDataProvider backed by Yahoo Finance (yfinance) — free, no API key needed.

NSE-listed symbols are mapped to Yahoo Finance tickers by appending the '.NS' suffix,
e.g. 'RELIANCE' → 'RELIANCE.NS'.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import yfinance as yf

from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider

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

_YF_SUFFIX = ".NS"


def _to_yf_symbol(symbol: str) -> str:
    if symbol.startswith("^") or symbol.endswith(_YF_SUFFIX):
        return symbol
    return symbol + _YF_SUFFIX


class YFinanceMarketDataProvider(MarketDataProvider):
    """Uses Yahoo Finance for free NSE OHLCV data."""

    def __init__(self, universe: list[str] | None = None) -> None:
        self._universe = universe[:] if universe else _DEFAULT_UNIVERSE[:]

    def get_universe(self) -> list[str]:
        return self._universe[:]

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        yf_symbol = _to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # end is exclusive in yfinance
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return []
        bars: list[dict] = []
        for ts, row in df.iterrows():
            bars.append(
                {
                    "date": str(ts.date()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )
        return bars

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        out: list[StockSnapshot] = []
        today = date.today()
        lookback = today - timedelta(days=35)

        for symbol in symbols:
            bars = self.get_historical(symbol=symbol, interval="1d", start=lookback, end=today)
            if not bars:
                continue
            closes = [float(r["close"]) for r in bars]
            vols = [float(r["volume"]) for r in bars]
            last = bars[-1]
            close = float(last["close"])
            volume = float(last["volume"])
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
