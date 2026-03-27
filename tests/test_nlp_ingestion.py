from __future__ import annotations

from stock_screener_engine.data_sources.filings.filings_adapter import FilingsAdapter
from stock_screener_engine.data_sources.filings.mock_filings import MockFilingsProvider
from stock_screener_engine.data_sources.news.generic_news_adapter import GenericNewsAdapter
from stock_screener_engine.data_sources.news.mock_news import MockNewsProvider
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.health_reporting import IngestionHealthSink


def test_text_document_ingestion_from_multiple_adapters() -> None:
    ingestor = TextDocumentIngestor(
        adapters=[
            GenericNewsAdapter(MockNewsProvider()),
            FilingsAdapter(MockFilingsProvider()),
        ]
    )
    docs = ingestor.ingest(symbols=["RELIANCE", "TCS"], lookback_days=30)
    assert docs
    assert any(d.source.value == "news" for d in docs)
    assert any(d.source.value == "filing" for d in docs)


def test_ingestion_writes_health_report(tmp_path) -> None:
    ingestor = TextDocumentIngestor(
        adapters=[
            GenericNewsAdapter(MockNewsProvider()),
            FilingsAdapter(MockFilingsProvider()),
        ],
        health_sink=IngestionHealthSink(str(tmp_path)),
    )

    _ = ingestor.ingest(symbols=["RELIANCE"], lookback_days=10)

    files = list((tmp_path / "ingestion_health").glob("*/ingestion_health.jsonl"))
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert '"news"' in content
    assert '"filings"' in content
