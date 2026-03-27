"""Abstract interfaces for all external data and broker adapters.

Dependency rule
---------------
These interfaces are the ONLY contracts that ``core/`` is allowed to import
from ``data_sources/``.  Concrete adapters (broker SDKs, HTTP clients, DB
connections) must live in the adapter packages and NEVER be imported by core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from stock_screener_engine.core.entities import (
    FundamentalsSnapshot,
    GovernanceSnapshot,
    MarketSnapshot,
    StockSnapshot,
)
from stock_screener_engine.data_sources.schemas import (
    AnnouncementRecord,
    CorporateActionRecord,
    OHLCVBar,
    ShareholdingRecord,
)


# ---------------------------------------------------------------------------
# Execution / order data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    price: float | None = None


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class MarketDataProvider(ABC):
    """Provides daily / intraday price and volume data."""

    @abstractmethod
    def get_universe(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        """Return unified snapshots (backward-compat; prefer get_market_snapshots)."""
        raise NotImplementedError

    def get_market_snapshots(self, symbols: Sequence[str]) -> list[MarketSnapshot]:
        """Return granular market snapshots.  Default: convert from get_snapshots."""
        unified = self.get_snapshots(symbols)
        return [
            MarketSnapshot(
                symbol=s.symbol,
                as_of=s.as_of,
                sector=s.sector,
                close=s.close,
                volume=s.volume,
                delivery_ratio=s.delivery_ratio,
            )
            for s in unified
        ]


class MarketIngestionAdapter(ABC):
    """Normalized OHLCV ingestion contract for exchange venue adapters."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> list[OHLCVBar]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fundamentals (quarterly/annual financial data)
# ---------------------------------------------------------------------------

class FinancialsProvider(ABC):
    """Provides quarterly and annual financial statement data."""

    @abstractmethod
    def get_fundamentals(self, symbols: Sequence[str]) -> dict[str, FundamentalsSnapshot]:
        """Return the latest fundamentals snapshot keyed by symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_governance(self, symbols: Sequence[str]) -> dict[str, GovernanceSnapshot]:
        """Return the latest governance snapshot keyed by symbol."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Exchange / filings
# ---------------------------------------------------------------------------

class ExchangeAdapter(ABC):
    """Provides exchange-sourced data: corporate actions and filings."""

    @abstractmethod
    def get_corporate_actions(self, symbols: Sequence[str]) -> dict[str, list[dict]]:
        raise NotImplementedError

    @abstractmethod
    def get_exchange_announcements(self, symbols: Sequence[str]) -> dict[str, list[dict]]:
        raise NotImplementedError


class ExchangeIngestionAdapter(ABC):
    """Normalized exchange events and shareholding ingestion contract."""

    @abstractmethod
    def fetch_corporate_actions(
        self, symbols: Sequence[str], start: date, end: date
    ) -> list[CorporateActionRecord]:
        raise NotImplementedError

    @abstractmethod
    def fetch_shareholding(self, symbols: Sequence[str], as_of: date) -> list[ShareholdingRecord]:
        raise NotImplementedError

    @abstractmethod
    def fetch_announcements(
        self, symbols: Sequence[str], start: date, end: date
    ) -> list[AnnouncementRecord]:
        raise NotImplementedError


class FilingsProvider(ABC):
    """Provides structured access to regulatory filings (SEBI, NSE/BSE)."""

    @abstractmethod
    def get_recent_filings(
        self, symbols: Sequence[str], lookback_days: int = 90
    ) -> dict[str, list[dict]]:
        """Return recent filing metadata keyed by symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_filing_sentiment(self, symbol: str, lookback_days: int = 90) -> float:
        """Return aggregate sentiment score [-1, 1] from recent filings."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# News / text / unstructured
# ---------------------------------------------------------------------------

class TextEventProvider(ABC):
    """Provides news and event text for sentiment and catalyst features."""

    @abstractmethod
    def get_recent_events(
        self, symbols: Sequence[str], lookback_days: int = 30
    ) -> dict[str, list[str]]:
        raise NotImplementedError

    @abstractmethod
    def get_sentiment_score(self, symbol: str, lookback_days: int = 30) -> float:
        raise NotImplementedError


class NewsProvider(ABC):
    """Dedicated news provider (may overlap with TextEventProvider for simple cases)."""

    @abstractmethod
    def get_recent_news(
        self, symbols: Sequence[str], lookback_days: int = 7
    ) -> dict[str, list[str]]:
        raise NotImplementedError

    @abstractmethod
    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Broker / execution
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Optional broker integration interface — must fail gracefully when disabled."""

    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_instruments(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_quote(self, symbols: Iterable[str]) -> dict[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, order_request: OrderRequest) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_holdings(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_order_history(self, order_id: str) -> list[dict]:
        raise NotImplementedError
