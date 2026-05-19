"""MarketDataProvider backed by an enabled broker adapter.

Broker APIs are useful as alternate live market-data sources when public NSE
or Yahoo endpoints are unavailable. This provider intentionally supplies only
market fields; fundamentals remain absent unless a separate FinancialsProvider
is wired into the research engine.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import BrokerAdapter, MarketDataProvider


class BrokerMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        broker: BrokerAdapter,
        universe: list[str] | None = None,
        broker_name: str = "broker",
    ) -> None:
        if not broker.is_enabled():
            raise ValueError(f"{broker_name} market data source is disabled or missing credentials")
        self._broker = broker
        self._broker_name = broker_name
        self._universe = [s.strip().upper() for s in (universe or []) if s.strip()]

    def get_universe(self) -> list[str]:
        if self._universe:
            return list(self._universe)
        instruments = self._broker.get_instruments()
        symbols: list[str] = []
        for row in instruments:
            exchange = str(row.get("exchange", row.get("exchange_code", "NSE"))).upper()
            symbol = str(row.get("tradingsymbol", row.get("stock_code", row.get("symbol", "")))).strip().upper()
            if symbol and exchange in {"NSE", ""}:
                symbols.append(symbol)
        return sorted(set(symbols))

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        return [_normalize_bar(row) for row in self._broker.get_historical(symbol, interval, start, end)]

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        out: list[StockSnapshot] = []
        today = date.today()
        lookback = today - timedelta(days=35)
        clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        quotes = self._broker.get_quote(clean_symbols) if clean_symbols else {}

        for symbol in clean_symbols:
            bars = self.get_historical(symbol=symbol, interval="1d", start=lookback, end=today)
            quote = quotes.get(symbol, {})
            close = _safe_float(_pick(quote, "ltp", "last_price", "price", "close"))
            volume = _safe_float(_pick(quote, "volume", "total_quantity_traded"))
            if bars:
                last = bars[-1]
                close = close or _safe_float(last.get("close"))
                volume = volume or _safe_float(last.get("volume"))
            if close <= 0.0:
                continue
            out.append(
                StockSnapshot(
                    symbol=symbol,
                    as_of=today,
                    sector="Unknown",
                    close=close,
                    volume=volume,
                    delivery_ratio=0.0,
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


def _normalize_bar(row: dict) -> dict:
    return {
        "date": str(_pick(row, "date", "datetime", "timestamp", "time") or ""),
        "open": _safe_float(_pick(row, "open", "open_price")),
        "high": _safe_float(_pick(row, "high", "high_price")),
        "low": _safe_float(_pick(row, "low", "low_price")),
        "close": _safe_float(_pick(row, "close", "close_price")),
        "volume": int(_safe_float(_pick(row, "volume", "total_quantity_traded"))),
    }


def _pick(row: dict, *keys: str) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _safe_float(value: object) -> float:
    try:
        return float(value) if value is not None and str(value).strip() else 0.0
    except (TypeError, ValueError):
        return 0.0
