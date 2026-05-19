"""Adapt a MarketDataProvider into the normalized ingestion contract."""

from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.base.interfaces import MarketDataProvider, MarketIngestionAdapter
from stock_screener_engine.data_sources.schemas import OHLCVBar


class ProviderMarketIngestionAdapter(MarketIngestionAdapter):
    def __init__(self, provider: MarketDataProvider, venue: str = "PROVIDER") -> None:
        self.provider = provider
        self.venue = venue

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> list[OHLCVBar]:
        rows = self.provider.get_historical(symbol=symbol, interval=interval, start=start, end=end)
        return [
            OHLCVBar(
                venue=self.venue,
                symbol=symbol.strip().upper(),
                ts=str(row.get("date") or row.get("ts") or ""),
                open=float(row.get("open", 0.0) or 0.0),
                high=float(row.get("high", 0.0) or 0.0),
                low=float(row.get("low", 0.0) or 0.0),
                close=float(row.get("close", 0.0) or 0.0),
                volume=float(row.get("volume", 0.0) or 0.0),
            )
            for row in rows
        ]
