"""Generic news adapter that converts NewsProvider output into normalized documents."""

from __future__ import annotations

from datetime import datetime, timedelta

from stock_screener_engine.data_sources.base.interfaces import NewsProvider
from stock_screener_engine.data_sources.news.base_news_adapter import BaseNewsAdapter
from stock_screener_engine.nlp.schemas.events import NormalizedDocument, SourceType


class GenericNewsAdapter(BaseNewsAdapter):
    def __init__(self, provider: NewsProvider) -> None:
        self.provider = provider

    def fetch_documents(self, symbols: list[str], lookback_days: int) -> list[NormalizedDocument]:
        rows = self.provider.get_recent_news(symbols, lookback_days=lookback_days)
        now = datetime.utcnow()
        out: list[NormalizedDocument] = []
        for symbol in symbols:
            headlines = rows.get(symbol, [])
            for idx, headline in enumerate(headlines):
                ts = now - timedelta(hours=idx * 4)
                out.append(
                    NormalizedDocument(
                        id=f"news:{symbol}:{idx}:{abs(hash(headline)) % 100000}",
                        source=SourceType.NEWS,
                        timestamp=ts,
                        symbol=symbol,
                        company_name=symbol,
                        title=headline,
                        body_text=headline,
                        source_name=self.provider.__class__.__name__,
                        url="",
                        metadata={"provider": self.provider.__class__.__name__},
                    )
                )
        return out
