"""CSV loader for NSE/BSE delivery and turnover files."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from stock_screener_engine.data_sources.schemas import DeliveryTurnoverRecord


def load_delivery_turnover_csv(
    file_path: str | Path,
    *,
    venue: str,
    default_trade_date: date | None = None,
    source_id: str = "",
) -> list[DeliveryTurnoverRecord]:
    """Load delivery/turnover rows from a normalized or exchange-style CSV.

    Accepted headers include normalized names such as ``symbol``,
    ``trade_date``, ``traded_quantity``, ``delivery_quantity``,
    ``delivery_pct`` and common NSE/BSE variants such as ``SYMBOL``,
    ``DATE1``, ``TTL_TRD_QNTY``, ``DELIV_QTY``, and ``DELIV_PER``.
    """
    path = Path(file_path)
    records: list[DeliveryTurnoverRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, raw in enumerate(reader, start=2):
            row = {_clean_key(key): _clean_value(value) for key, value in raw.items()}
            symbol = _first(row, "symbol", "security_symbol", "scrip_name", "scrip_cd", "security_code")
            if not symbol:
                continue
            trade_date = _parse_date(
                _first(row, "trade_date", "date", "date1", "trading_date"),
                default=default_trade_date,
            )
            if trade_date is None:
                raise ValueError(f"Missing trade date in {path} row {row_number}")
            traded_quantity = _number(_first(row, "traded_quantity", "ttl_trd_qnty", "total_traded_quantity", "volume"))
            delivery_quantity = _number(_first(row, "delivery_quantity", "deliv_qty", "deliverable_quantity"))
            delivery_pct = _number(_first(row, "delivery_pct", "delivery_percent", "deliv_per", "deliverable_percent"))
            if delivery_pct <= 0 and traded_quantity > 0 and delivery_quantity >= 0:
                delivery_pct = delivery_quantity / traded_quantity * 100.0
            records.append(
                DeliveryTurnoverRecord(
                    venue=venue.strip().upper(),
                    symbol=symbol.strip().upper(),
                    trade_date=trade_date,
                    traded_quantity=traded_quantity,
                    delivery_quantity=delivery_quantity,
                    delivery_pct=delivery_pct,
                    source_id=source_id or path.name,
                )
            )
    return records


def _clean_key(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _clean_value(value: object) -> str:
    return str(value or "").strip().replace(",", "")


def _first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def _number(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_date(value: str, *, default: date | None) -> date | None:
    if not value:
        return default
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return default
