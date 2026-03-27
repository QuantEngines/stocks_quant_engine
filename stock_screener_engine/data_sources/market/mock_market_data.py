"""Mock market adapter for local research and demo workflows."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import MarketSnapshot, StockSnapshot
from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider

_SECTORS = ["Energy", "IT", "Banking", "CapitalGoods", "Pharma"]


class MockIndianMarketDataProvider(MarketDataProvider):
    def __init__(self) -> None:
        self._universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "LT", "SBIN", "ITC", "SUNPHARMA"]

    def get_universe(self) -> list[str]:
        return list(self._universe)

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        rows: list[dict] = []
        day = start
        seed = sum(ord(ch) for ch in symbol)
        base = 100.0 + (seed % 1000)
        while day <= end:
            drift = ((day.toordinal() % 17) - 8) * 0.002
            close = base * (1.0 + drift)
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": round(close * 0.995, 2),
                    "high": round(close * 1.01, 2),
                    "low": round(close * 0.99, 2),
                    "close": round(close, 2),
                    "volume": int(500_000 + (seed * day.toordinal()) % 5_000_000),
                }
            )
            day += timedelta(days=1)
        return rows

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        today = date.today()
        symbols = list(symbols)
        snapshots: list[StockSnapshot] = []
        for idx, symbol in enumerate(symbols):
            factor = (idx + 1) / (len(symbols) + 2)
            snapshots.append(
                StockSnapshot(
                    symbol=symbol,
                    as_of=today,
                    sector=_SECTORS[idx % len(_SECTORS)],
                    close=150.0 + idx * 320.0,
                    volume=1_000_000 + idx * 350_000,
                    delivery_ratio=0.35 + factor * 0.5,
                    pe_ratio=14.0 + idx * 2.2,
                    roe=0.12 + factor * 0.28,
                    debt_to_equity=0.15 + (1 - factor) * 1.1,
                    earnings_growth=0.05 + factor * 0.4,
                    free_cash_flow_margin=0.07 + factor * 0.32,
                    promoter_holding_change=-0.02 + factor * 0.08,
                    insider_activity_score=0.3 + factor * 0.5,
                )
            )
        return snapshots

    def get_market_snapshots(self, symbols: Sequence[str]) -> list[MarketSnapshot]:
        today = date.today()
        symbols = list(symbols)
        snapshots: list[MarketSnapshot] = []
        for idx, symbol in enumerate(symbols):
            factor = (idx + 1) / (len(symbols) + 2)
            volume = 1_000_000 + idx * 350_000
            snapshots.append(
                MarketSnapshot(
                    symbol=symbol,
                    as_of=today,
                    sector=_SECTORS[idx % len(_SECTORS)],
                    exchange="NSE",
                    close=150.0 + idx * 320.0,
                    open_price=148.0 + idx * 320.0,
                    high=155.0 + idx * 320.0,
                    low=147.0 + idx * 320.0,
                    volume=float(volume),
                    delivery_volume=float(int(volume * (0.35 + factor * 0.5))),
                    delivery_ratio=0.35 + factor * 0.5,
                    market_cap=(150.0 + idx * 320.0) * 1_000_000,
                    avg_volume_20d=float(volume) * 0.95,
                )
            )
        return snapshots

