"""BSE HTTP ingestion adapter with normalized contracts."""

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


class BSEHTTPAdapter(MarketIngestionAdapter, ExchangeIngestionAdapter):
    venue = "BSE"

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = RetryingHTTPClient(
            HTTPRetryConfig(timeout_seconds=timeout_seconds)
        )

    _CANDIDATE_ENDPOINTS = {
        "ohlcv": [
            "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w",
        ],
        "corporate_actions": [
            "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w",
        ],
        "shareholding": [
            "https://api.bseindia.com/BseIndiaAPI/api/Shareholding/w",
        ],
        "announcements": [
            "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
        ],
    }

    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> list[OHLCVBar]:
        payload = self._client.get_json(
            urls=self._CANDIDATE_ENDPOINTS["ohlcv"],
            params={
                "scripcode": symbol,
                "flag": "1",
                "fromdate": start.strftime("%d/%m/%Y"),
                "todate": end.strftime("%d/%m/%Y"),
            },
        )
        rows = payload.get("Data", []) if isinstance(payload, dict) else []
        bars: list[OHLCVBar] = []
        for row in rows:
            bars.append(
                OHLCVBar(
                    venue=self.venue,
                    symbol=symbol,
                    ts=str(row.get("dttm", "")),
                    open=float(row.get("Open", 0.0) or 0.0),
                    high=float(row.get("High", 0.0) or 0.0),
                    low=float(row.get("Low", 0.0) or 0.0),
                    close=float(row.get("Close", 0.0) or 0.0),
                    volume=float(row.get("Volume", 0.0) or 0.0),
                )
            )
        return bars

    def fetch_corporate_actions(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateActionRecord]:
        records: list[CorporateActionRecord] = []
        for symbol in symbols:
            payload = self._client.get_json(
                urls=self._CANDIDATE_ENDPOINTS["corporate_actions"],
                params={"scripcode": symbol, "from": start.isoformat(), "to": end.isoformat()},
            )
            rows = payload.get("Table", []) if isinstance(payload, dict) else []
            for row in rows:
                records.append(
                    CorporateActionRecord(
                        venue=self.venue,
                        symbol=symbol,
                        action_type=str(row.get("Purpose", "unknown")).strip().lower().replace(" ", "_"),
                        ex_date=str(row.get("ExDate", "")),
                        record_date=str(row.get("RecordDate")) if row.get("RecordDate") else None,
                        ratio=str(row.get("Ratio")) if row.get("Ratio") else None,
                        cash_amount=_safe_float(row.get("Amount")),
                        source_id=str(row.get("AttachmentName", "")),
                    )
                )
        return records

    def fetch_shareholding(self, symbols: Sequence[str], as_of: date) -> list[ShareholdingRecord]:
        records: list[ShareholdingRecord] = []
        for symbol in symbols:
            payload = self._client.get_json(
                urls=self._CANDIDATE_ENDPOINTS["shareholding"],
                params={"scripcode": symbol, "asof": as_of.isoformat()},
            )
            rows = payload.get("Table", []) if isinstance(payload, dict) else []
            for row in rows:
                promoter = _safe_float(row.get("Promoter"))
                fii = _safe_float(row.get("FII"))
                dii = _safe_float(row.get("DII"))
                public = max(0.0, 100.0 - promoter - fii - dii)
                records.append(
                    ShareholdingRecord(
                        venue=self.venue,
                        symbol=symbol,
                        period_end=as_of,
                        filing_date=as_of,
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
                params={"strScrip": symbol, "FromDate": start.isoformat(), "ToDate": end.isoformat()},
            )
            rows = payload.get("Table", []) if isinstance(payload, dict) else []
            for row in rows:
                records.append(
                    AnnouncementRecord(
                        venue=self.venue,
                        symbol=symbol,
                        published_at=str(row.get("NEWS_DT", "")),
                        category=str(row.get("CATEGORYNAME", "general")).strip().lower().replace(" ", "_"),
                        subject=str(row.get("HEADLINE", "")),
                        url=str(row.get("ATTACHMENTNAME", "")),
                        source_id=str(row.get("NEWSID", "")),
                    )
                )
        return records

def _safe_float(value: object) -> float:
    try:
        return float(str(value).strip()) if value is not None and str(value).strip() != "" else 0.0
    except (TypeError, ValueError):
        return 0.0
