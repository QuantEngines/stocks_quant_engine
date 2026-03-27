"""Null FilingsProvider — returns empty results when no filings source is available."""

from __future__ import annotations

from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import FilingsProvider


class NullFilingsProvider(FilingsProvider):
    """No-op provider used when exchange filing endpoints are unavailable."""

    def get_recent_filings(
        self,
        symbols: Sequence[str],
        lookback_days: int = 90,
    ) -> dict[str, list[dict]]:
        return {str(s): [] for s in symbols}

    def get_filing_sentiment(self, symbol: str, lookback_days: int = 90) -> float:
        return 0.0
