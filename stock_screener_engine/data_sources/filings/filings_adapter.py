"""Adapter converting filing provider records into normalized NLP documents."""

from __future__ import annotations

from datetime import datetime

from stock_screener_engine.data_sources.base.interfaces import FilingsProvider
from stock_screener_engine.nlp.schemas.events import NormalizedDocument, SourceType


class FilingsAdapter:
    def __init__(self, provider: FilingsProvider) -> None:
        self.provider = provider

    def fetch_documents(self, symbols: list[str], lookback_days: int) -> list[NormalizedDocument]:
        rows = self.provider.get_recent_filings(symbols, lookback_days=lookback_days)
        out: list[NormalizedDocument] = []
        for symbol in symbols:
            filings = rows.get(symbol, [])
            for idx, filing in enumerate(filings):
                ts = _parse_ts(str(filing.get("filing_date", "")))
                title = str(filing.get("subject", "Filing update"))
                body = f"{filing.get('type', 'filing')}: {title}"
                out.append(
                    NormalizedDocument(
                        id=f"filing:{symbol}:{idx}:{abs(hash(body)) % 100000}",
                        source=SourceType.FILING,
                        timestamp=ts,
                        symbol=symbol,
                        company_name=symbol,
                        title=title,
                        body_text=body,
                        source_name=str(filing.get("exchange", "filing")),
                        url=str(filing.get("url", "")),
                        metadata={
                            "exchange": str(filing.get("exchange", "")),
                            "type": str(filing.get("type", "")),
                            "url": str(filing.get("url", "")),
                        },
                    )
                )
        return out


def _parse_ts(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.utcnow()
