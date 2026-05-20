"""CSV and plain-text universe loaders for canonical security metadata."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Iterable

from stock_screener_engine.data_sources.schemas import SecurityMasterRecord


_SYMBOL_KEYS = ("symbol", "tradingsymbol", "ticker", "nse_symbol", "SYMBOL")


def load_security_master_csv(
    file_path: str,
    default_exchange: str = "NSE",
) -> list[SecurityMasterRecord]:
    """Load an external security master or simple symbol list.

    The loader accepts either:
    * a CSV with headers such as ``symbol,company_name,sector,industry``
    * an NSE-style CSV with headers such as ``SYMBOL`` and ``NAME OF COMPANY``
    * a plain one-symbol-per-line text/CSV file

    External universe files are intentionally read at runtime so broad index
    universes and research metadata can live outside the git repository.
    """
    path = Path(file_path).expanduser()
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    if _looks_like_header(lines[0]):
        return _load_header_csv(lines, default_exchange=default_exchange)
    return _load_plain_symbol_rows(lines, default_exchange=default_exchange)


def _load_header_csv(lines: list[str], default_exchange: str) -> list[SecurityMasterRecord]:
    reader = csv.DictReader(lines)
    records: list[SecurityMasterRecord] = []
    for row in reader:
        symbol = _first(row, *_SYMBOL_KEYS).strip().upper()
        if not symbol:
            continue
        exchange = (_first(row, "exchange", "venue") or default_exchange).strip().upper()
        records.append(
            SecurityMasterRecord(
                symbol=symbol,
                exchange=exchange,
                isin=_first(row, "isin", "ISIN", "ISIN Code", "isin code", "isin_number", "ISIN NUMBER").strip(),
                series=(_first(row, "series", "SERIES") or "EQ").strip() or "EQ",
                company_name=_first(
                    row,
                    "company_name",
                    "Company Name",
                    "company name",
                    "name",
                    "company",
                    "NAME OF COMPANY",
                    "security_name",
                ).strip(),
                sector=(_first(row, "sector", "Industry", "industry_macro", "basic_industry") or "Unknown").strip() or "Unknown",
                industry=(_first(row, "industry", "Industry", "industry_name", "sub_sector") or "Unknown").strip() or "Unknown",
                listing_date=_optional_date(_first(row, "listing_date", "DATE OF LISTING")),
                delisting_date=_optional_date(_first(row, "delisting_date")),
                active=_bool(_first(row, "active", "status"), default=True),
                lot_size=_int(_first(row, "lot_size"), default=1),
                tick_size=_float(_first(row, "tick_size"), default=0.05),
                source=(_first(row, "source", "source_id") or "universe_file").strip() or "universe_file",
            )
        )
    return records


def _load_plain_symbol_rows(lines: Iterable[str], default_exchange: str) -> list[SecurityMasterRecord]:
    records: list[SecurityMasterRecord] = []
    for line in lines:
        row = next(csv.reader([line]))
        symbol = (row[0] if row else "").strip().upper()
        if not symbol:
            continue
        records.append(
            SecurityMasterRecord(
                symbol=symbol,
                exchange=default_exchange.strip().upper(),
                company_name=symbol,
                source="universe_file",
            )
        )
    return records


def _looks_like_header(first_line: str) -> bool:
    headers = {col.strip().lower() for col in next(csv.reader([first_line]))}
    return bool(headers & {"symbol", "tradingsymbol", "ticker", "nse_symbol", "name of company"})


def _first(row: dict[str, str | None], *keys: str) -> str:
    lower_lookup = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
        value = lower_lookup.get(key.strip().lower())
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _optional_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _bool(value: object, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "active", "listed", "eq"}:
        return True
    if text in {"0", "false", "no", "n", "inactive", "delisted", "suspended"}:
        return False
    return default


def _int(value: object, default: int) -> int:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _float(value: object, default: float) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default
