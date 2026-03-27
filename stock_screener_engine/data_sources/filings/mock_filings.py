"""Mock filings provider for local research and demo workflows."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import FilingsProvider


class MockFilingsProvider(FilingsProvider):
    """Returns synthetic regulatory-filing stubs for demo and unit tests.

    Each symbol gets a small set of plausible BSE/NSE announcements so that
    downstream pipeline stages can exercise the full data flow without a live
    exchange connection.
    """

    _FILING_TEMPLATES = [
        {"type": "board_meeting", "subject": "Board meeting to consider Q3 results", "sentiment": 0.1},
        {"type": "shareholding", "subject": "Shareholding pattern for Q3", "sentiment": 0.0},
        {"type": "dividend", "subject": "Interim dividend declared", "sentiment": 0.4},
        {"type": "acquisition", "subject": "Acquisition of subsidiary", "sentiment": 0.2},
        {"type": "insider_trading", "subject": "Promoter acquisition disclosed", "sentiment": 0.3},
    ]

    def get_recent_filings(
        self, symbols: Sequence[str], lookback_days: int = 90
    ) -> dict[str, list[dict]]:
        today = date.today()
        result: dict[str, list[dict]] = {}
        for idx, symbol in enumerate(symbols):
            filings = []
            for i, tpl in enumerate(self._FILING_TEMPLATES):
                filing_date = today - timedelta(days=(i + 1) * max(1, lookback_days // 5))
                filings.append(
                    {
                        "symbol": symbol,
                        "filing_date": filing_date.isoformat(),
                        "type": tpl["type"],
                        "subject": tpl["subject"],
                        "exchange": "BSE",
                        "url": f"https://example.com/filings/{symbol}/{tpl['type']}",
                    }
                )
            result[symbol] = filings
        return result

    def get_filing_sentiment(self, symbol: str, lookback_days: int = 90) -> float:
        """Return a synthetic aggregate sentiment score in [-1, 1]."""
        seed = sum(ord(c) for c in symbol) % 10
        return round((seed - 5) / 10.0, 2)  # deterministic range: -0.5 to 0.4
