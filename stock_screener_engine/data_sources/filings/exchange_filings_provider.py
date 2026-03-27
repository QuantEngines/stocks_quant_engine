"""Free filings provider built from public exchange announcements."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import FilingsProvider
from stock_screener_engine.data_sources.exchange.nse_http_adapter import NSEHTTPAdapter


class ExchangeFilingsProvider(FilingsProvider):
    """Uses NSE public announcements endpoint as a filings-like source."""

    def __init__(self, adapter: NSEHTTPAdapter | None = None) -> None:
        self.adapter = adapter or NSEHTTPAdapter()

    def get_recent_filings(
        self,
        symbols: Sequence[str],
        lookback_days: int = 90,
    ) -> dict[str, list[dict]]:
        end = date.today()
        start = end - timedelta(days=max(1, lookback_days))
        rows = self.adapter.fetch_announcements(list(symbols), start=start, end=end)
        out: dict[str, list[dict]] = {str(s): [] for s in symbols}
        for rec in rows:
            out.setdefault(rec.symbol, []).append(
                {
                    "filing_date": _normalize_ts(rec.published_at),
                    "subject": rec.subject,
                    "type": rec.category,
                    "exchange": rec.venue,
                    "url": rec.url,
                }
            )
        return out

    def get_filing_sentiment(self, symbol: str, lookback_days: int = 90) -> float:
        payload = self.get_recent_filings([symbol], lookback_days=lookback_days)
        filings = payload.get(symbol, [])
        if not filings:
            return 0.0
        score = 0.0
        for item in filings:
            text = f"{item.get('subject', '')} {item.get('type', '')}".lower()
            score += _sentiment(text)
        return max(-1.0, min(1.0, score / max(1, len(filings))))


def _normalize_ts(text: str) -> str:
    text = text.strip()
    if not text:
        return datetime.utcnow().isoformat()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return datetime.utcnow().isoformat()


def _sentiment(text: str) -> float:
    positives = ("award", "wins", "approval", "raises", "allotment", "expansion")
    negatives = ("penalty", "investigation", "default", "resign", "delay", "downgrade")
    p = sum(k in text for k in positives)
    n = sum(k in text for k in negatives)
    if p + n == 0:
        return 0.0
    return max(-1.0, min(1.0, (p - n) / max(1, p + n)))
