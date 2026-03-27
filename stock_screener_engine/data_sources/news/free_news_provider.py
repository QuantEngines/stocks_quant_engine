"""Free news/text provider using public RSS endpoints (no paid API key)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from stock_screener_engine.data_sources.base.interfaces import NewsProvider, TextEventProvider


class FreeRSSNewsProvider(NewsProvider, TextEventProvider):
    """Fetches business headlines from public RSS search endpoints per symbol."""

    def __init__(self, timeout_seconds: int = 12, max_headlines_per_symbol: int = 12) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_headlines_per_symbol = max_headlines_per_symbol

    def get_recent_news(self, symbols: Sequence[str], lookback_days: int = 7) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for symbol in symbols:
            out[str(symbol)] = self._fetch_symbol_headlines(str(symbol), lookback_days=lookback_days)
        return out

    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> float:
        headlines = self._fetch_symbol_headlines(symbol, lookback_days=lookback_days)
        if not headlines:
            return 0.0
        score = 0.0
        for h in headlines:
            score += _headline_sentiment(h)
        return max(-1.0, min(1.0, score / max(1, len(headlines))))

    def get_recent_events(self, symbols: Sequence[str], lookback_days: int = 30) -> dict[str, list[str]]:
        return self.get_recent_news(symbols, lookback_days=lookback_days)

    def get_sentiment_score(self, symbol: str, lookback_days: int = 30) -> float:
        return self.get_news_sentiment(symbol=symbol, lookback_days=lookback_days)

    def _fetch_symbol_headlines(self, symbol: str, lookback_days: int) -> list[str]:
        q = quote_plus(f"{symbol} NSE India stock when:{max(1, lookback_days)}d")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8", errors="ignore")
        except OSError:
            return []

        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError:
            return []

        items = root.findall(".//item")
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max(1, lookback_days))
        headlines: list[str] = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            pub_date_text = (item.findtext("pubDate") or "").strip()
            if pub_date_text:
                dt = _parse_rfc822(pub_date_text)
                if dt is not None and dt < cutoff:
                    continue
            headlines.append(title)
            if len(headlines) >= self.max_headlines_per_symbol:
                break
        return headlines


def _parse_rfc822(text: str) -> datetime | None:
    # Example format: Thu, 26 Mar 2026 11:40:00 GMT
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%d %b %Y %H:%M:%S %Z"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _headline_sentiment(headline: str) -> float:
    text = headline.lower()
    positives = (
        "beats",
        "beat",
        "surge",
        "jumps",
        "wins",
        "order win",
        "raises",
        "upgrade",
        "profit rises",
        "strong",
    )
    negatives = (
        "misses",
        "miss",
        "drops",
        "falls",
        "cut",
        "downgrade",
        "penalty",
        "fraud",
        "probe",
        "weak",
    )
    p = sum(token in text for token in positives)
    n = sum(token in text for token in negatives)
    if p + n == 0:
        return 0.0
    return max(-1.0, min(1.0, (p - n) / max(1, p + n)))
