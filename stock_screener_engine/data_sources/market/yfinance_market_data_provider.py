"""MarketDataProvider backed by Yahoo Finance (yfinance) — free, no API key needed.

NSE-listed symbols are mapped to Yahoo Finance tickers by appending the '.NS' suffix,
e.g. 'RELIANCE' → 'RELIANCE.NS'.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

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


def _require_yfinance():
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The yfinance market provider requires the optional 'yfinance' package. "
            "Install it or set SSE_MARKET_PROVIDER=nse_http, zerodha, or icici_breeze."
        ) from exc
    return yf


class YFinanceMarketDataProvider(MarketDataProvider):
    """Uses Yahoo Finance for free NSE OHLCV data."""

    def __init__(self, universe: list[str] | None = None) -> None:
        self._universe = universe[:] if universe else _DEFAULT_UNIVERSE[:]

    def get_universe(self) -> list[str]:
        return self._universe[:]

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        yf = _require_yfinance()
        yf_symbol = _to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # end is exclusive in yfinance
            interval=_yf_interval(interval),
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
            last = bars[-1]
            close = float(last["close"])
            volume = float(last["volume"])

            out.append(
                StockSnapshot(
                    symbol=symbol,
                    as_of=today,
                    sector="Unknown",
                    close=close,
                    volume=volume,
                    delivery_ratio=0.5,
                    pe_ratio=0.0,
                    roe=0.0,
                    debt_to_equity=0.0,
                    earnings_growth=0.0,
                    free_cash_flow_margin=0.0,
                    promoter_holding_change=0.0,
                    insider_activity_score=0.0,
                )
            )
        return out


def _yf_interval(interval: str) -> str:
    mapping = {
        "1d": "1d",
        "day": "1d",
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
    }
    return mapping.get(interval.lower(), interval)
