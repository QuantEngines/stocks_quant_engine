"""NSE HTTP ingestion adapter with normalized contracts.

This adapter targets publicly accessible NSE endpoints and normalizes records
into typed schema objects. Endpoints and response payloads can change over time,
so this module keeps parsing logic isolated and easy to patch.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import ExchangeIngestionAdapter, MarketIngestionAdapter
from stock_screener_engine.data_sources.exchange.http_client import HTTPRetryConfig, RetryingHTTPClient
from stock_screener_engine.data_sources.schemas import (
    AnnouncementRecord,
    CorporateActionRecord,
    OHLCVBar,
    ShareholdingRecord,
)


class NSEHTTPAdapter(MarketIngestionAdapter, ExchangeIngestionAdapter):
    venue = "NSE"

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = RetryingHTTPClient(
            HTTPRetryConfig(timeout_seconds=timeout_seconds)
        )
        self._bootstrap = "https://www.nseindia.com"

    _CANDIDATE_ENDPOINTS = {
        "ohlcv": [
            "https://www.nseindia.com/api/historical/cm/equity",
        ],
        "corporate_actions": [
            "https://www.nseindia.com/api/corporates-corporateActions",
        ],
        "shareholding": [
            "https://www.nseindia.com/api/corporate-share-holdings",
        ],
        "announcements": [
            "https://www.nseindia.com/api/corporate-announcements",
        ],
    }

    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> list[OHLCVBar]:
        payload = self._client.get_json(
            urls=self._CANDIDATE_ENDPOINTS["ohlcv"],
            params={
            "symbol": symbol,
            "from": start.strftime("%d-%m-%Y"),
            "to": end.strftime("%d-%m-%Y"),
            "series": "[\"EQ\"]",
            },
            bootstrap_url=self._bootstrap,
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        bars: list[OHLCVBar] = []
        for row in rows:
            bars.append(
                OHLCVBar(
                    venue=self.venue,
                    symbol=symbol,
                    ts=str(row.get("CH_TIMESTAMP", "")),
                    open=float(row.get("CH_OPENING_PRICE", 0.0) or 0.0),
                    high=float(row.get("CH_TRADE_HIGH_PRICE", 0.0) or 0.0),
                    low=float(row.get("CH_TRADE_LOW_PRICE", 0.0) or 0.0),
                    close=float(row.get("CH_CLOSING_PRICE", 0.0) or 0.0),
                    volume=float(row.get("CH_TOT_TRADED_QTY", 0.0) or 0.0),
                )
            )
        return list(reversed(bars))

    def fetch_corporate_actions(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateActionRecord]:
        records: list[CorporateActionRecord] = []
        for symbol in symbols:
            payload = self._client.get_json(
                urls=self._CANDIDATE_ENDPOINTS["corporate_actions"],
                params={"symbol": symbol, "from_date": start.isoformat(), "to_date": end.isoformat()},
                bootstrap_url=self._bootstrap,
            )
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            for row in rows:
                records.append(
                    CorporateActionRecord(
                        venue=self.venue,
                        symbol=symbol,
                        action_type=str(row.get("subject", "unknown")).strip().lower().replace(" ", "_"),
                        ex_date=str(row.get("exDate", "")),
                        record_date=str(row.get("recordDate")) if row.get("recordDate") else None,
                        ratio=str(row.get("bcStartDate")) if row.get("bcStartDate") else None,
                        cash_amount=None,
                        source_id=str(row.get("attchmntFile", "")),
                    )
                )
        return records

    def fetch_shareholding(self, symbols: Sequence[str], as_of: date) -> list[ShareholdingRecord]:
        records: list[ShareholdingRecord] = []
        for symbol in symbols:
            payload = self._client.get_json(
                urls=self._CANDIDATE_ENDPOINTS["shareholding"],
                params={"symbol": symbol, "as_of": as_of.isoformat()},
                bootstrap_url=self._bootstrap,
            )
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            for row in rows:
                filing_date = _safe_date(str(row.get("filingDate", as_of.isoformat())))
                period_end = _safe_date(str(row.get("periodEnd", as_of.isoformat())))
                promoter = float(row.get("promoter", 0.0) or 0.0)
                fii = float(row.get("fii", 0.0) or 0.0)
                dii = float(row.get("dii", 0.0) or 0.0)
                public = max(0.0, 100.0 - promoter - fii - dii)
                records.append(
                    ShareholdingRecord(
                        venue=self.venue,
                        symbol=symbol,
                        period_end=period_end,
                        filing_date=filing_date,
                        promoter_pct=promoter,
                        fii_pct=fii,
                        dii_pct=dii,
                        public_pct=public,
                        source_id=str(row.get("id", "")),
                    )
                )
        return records

    def fetch_announcements(self, symbols: Sequence[str], start: date, end: date) -> list[AnnouncementRecord]:
        records: list[AnnouncementRecord] = []
        for symbol in symbols:
            payload = self._client.get_json(
                urls=self._CANDIDATE_ENDPOINTS["announcements"],
                params={"symbol": symbol, "from_date": start.isoformat(), "to_date": end.isoformat()},
                bootstrap_url=self._bootstrap,
            )
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            for row in rows:
                records.append(
                    AnnouncementRecord(
                        venue=self.venue,
                        symbol=symbol,
                        published_at=str(row.get("an_dt", "")),
                        category=str(row.get("desc", "general")).strip().lower().replace(" ", "_"),
                        subject=str(row.get("sm_name", "")),
                        url=str(row.get("attchmntFile", "")),
                        source_id=str(row.get("newsid", "")),
                    )
                )
        return records

def _safe_date(text: str) -> date:
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
        try:
            return date.fromisoformat(text) if fmt == "%Y-%m-%d" else date.fromisoformat(_to_iso(text, fmt))
        except ValueError:
            continue
    return date.today()


def _to_iso(text: str, fmt: str) -> str:
    from datetime import datetime

    return datetime.strptime(text, fmt).date().isoformat()
