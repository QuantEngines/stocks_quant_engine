"""NSE/BSE exchange data-foundation utilities."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

from stock_screener_engine.data_sources.exchange.delivery_csv_loader import load_delivery_turnover_csv
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


def ingest_delivery_turnover_csv(
    *,
    store: MarketDataStore,
    file_store: LocalFileStorage,
    file_path: str,
    venue: str,
    default_trade_date: date | None = None,
    source_id: str = "",
) -> dict[str, object]:
    """Ingest delivery/turnover CSV rows into canonical storage."""
    records = load_delivery_turnover_csv(
        file_path,
        venue=venue,
        default_trade_date=default_trade_date,
        source_id=source_id,
    )
    persisted = store.upsert_delivery_turnover(records)
    by_symbol: dict[str, int] = {}
    for record in records:
        by_symbol[record.symbol] = by_symbol.get(record.symbol, 0) + 1
    report = {
        "pipeline": "exchange_delivery_ingest",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "venue": venue.strip().upper(),
        "file": str(Path(file_path)),
        "default_trade_date": default_trade_date.isoformat() if default_trade_date else None,
        "input_rows": len(records),
        "persisted": persisted,
        "symbols": len(by_symbol),
        "rows_by_symbol": dict(sorted(by_symbol.items())),
        "sample_rows": [asdict(record) for record in records[:5]],
        "passed": persisted == len(records),
    }
    quality_dir = file_store.root / "quality"
    report["artifacts"] = {
        "json": str(quality_dir / "exchange_delivery_ingest_report.json"),
    }
    file_store.save_json(report, filename="exchange_delivery_ingest_report.json", subdir="quality")
    return report


def build_exchange_foundation_status(
    *,
    store: MarketDataStore,
    symbols: Sequence[str],
    as_of: date,
    start: date,
    venue: str,
    interval: str = "1d",
) -> dict[str, object]:
    """Summarize current exchange-foundation coverage and remaining blockers."""
    normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    metadata = store.company_metadata(normalized_symbols)
    security_covered = sorted(symbol for symbol in normalized_symbols if symbol in metadata)
    ohlcv = store.coverage_summary(normalized_symbols, start=start, end=as_of, interval=interval)
    delivery = store.delivery_turnover_coverage(normalized_symbols, as_of=as_of, venue=venue)
    corporate_actions = _corporate_action_coverage(store, normalized_symbols, venue=venue)

    rows = [
        _row(
            domain="security_master",
            coverage=_coverage(len(security_covered), len(normalized_symbols)),
            covered=len(security_covered),
            total=len(normalized_symbols),
            status=_status(_coverage(len(security_covered), len(normalized_symbols))),
            next_action="Automate NSE/BSE security master and BSE scrip-code mapping.",
        ),
        _row(
            domain="bhavcopy_ohlcv",
            coverage=float(ohlcv.get("coverage", 0.0) or 0.0),
            covered=int(ohlcv.get("symbols_with_bars", 0) or 0),
            total=int(ohlcv.get("symbols_requested", 0) or 0),
            status=_status(float(ohlcv.get("coverage", 0.0) or 0.0)),
            next_action="Add official NSE/BSE bhavcopy ingestion and reconcile with broker/yfinance bars.",
        ),
        _row(
            domain="delivery_turnover",
            coverage=float(delivery.get("coverage", 0.0) or 0.0),
            covered=int(delivery.get("symbols_with_delivery", 0) or 0),
            total=int(delivery.get("symbols_requested", 0) or 0),
            status=_status(float(delivery.get("coverage", 0.0) or 0.0)),
            next_action="Ingest official NSE/BSE delivery/turnover CSVs into canonical delivery_turnover.",
        ),
        _row(
            domain="corporate_actions",
            coverage=float(corporate_actions.get("coverage", 0.0) or 0.0),
            covered=int(corporate_actions.get("symbols_with_actions", 0) or 0),
            total=int(corporate_actions.get("symbols_requested", 0) or 0),
            status="partial" if corporate_actions.get("symbols_with_actions") else "gap",
            next_action="Build robust exchange/vendor corporate-action ingestion for adjusted history.",
        ),
        _row(
            domain="announcements_pdfs",
            coverage=0.0,
            covered=0,
            total=len(normalized_symbols),
            status="gap",
            next_action="Replace brittle filings endpoint with cached NSE/BSE announcement/PDF ingestion.",
        ),
        _row(
            domain="historical_constituents_symbol_changes",
            coverage=0.0,
            covered=0,
            total=len(normalized_symbols),
            status="gap",
            next_action="Acquire historical constituents, delistings, and symbol changes before institutional backtests.",
        ),
    ]
    report = {
        "pipeline": "exchange_foundation_status",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "venue": venue.strip().upper(),
        "as_of": as_of.isoformat(),
        "start": start.isoformat(),
        "symbols": len(normalized_symbols),
        "domains": rows,
    }
    report["markdown"] = render_exchange_foundation_markdown(report)
    return report


def render_exchange_foundation_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# NSE/BSE Exchange Foundation Status",
        "",
        f"- Venue: {report.get('venue')}",
        f"- As of: {report.get('as_of')}",
        f"- Symbols: {report.get('symbols')}",
        "",
        "| Domain | Coverage | Status | Covered | Total | Next Action |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in report.get("domains", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {domain} | {coverage:.1%} | {status} | {covered} | {total} | {next_action} |".format(
                domain=row.get("domain", ""),
                coverage=float(row.get("coverage", 0.0) or 0.0),
                status=row.get("status", ""),
                covered=row.get("covered", 0),
                total=row.get("total", 0),
                next_action=row.get("next_action", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _corporate_action_coverage(store: MarketDataStore, symbols: Sequence[str], venue: str) -> dict[str, object]:
    requested = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
    available = []
    for symbol in requested:
        if store.get_corporate_actions(symbol=symbol, venue=venue):
            available.append(symbol)
    missing = sorted(requested - set(available))
    return {
        "symbols_requested": len(requested),
        "symbols_with_actions": len(available),
        "symbols_without_actions": missing,
        "coverage": _coverage(len(available), len(requested)),
    }


def _row(
    *,
    domain: str,
    coverage: float,
    covered: int,
    total: int,
    status: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "domain": domain,
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "covered": covered,
        "total": total,
        "status": status,
        "next_action": next_action,
    }


def _coverage(covered: int, total: int) -> float:
    return covered / total if total else 0.0


def _status(coverage: float) -> str:
    if coverage >= 0.95:
        return "ready"
    if coverage >= 0.70:
        return "partial"
    if coverage > 0:
        return "thin"
    return "gap"
