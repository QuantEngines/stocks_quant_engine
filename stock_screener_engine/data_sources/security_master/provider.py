"""Security master providers for canonical Indian equity identifiers."""

from __future__ import annotations

from typing import Iterable, Sequence

from stock_screener_engine.data_sources.schemas import SecurityMasterRecord


class StaticSecurityMasterProvider:
    """In-memory security master provider for tests, configs, and bootstrapping."""

    def __init__(self, records: Iterable[SecurityMasterRecord]) -> None:
        self._records = {record.symbol.upper(): record for record in records}

    def get_records(self, symbols: Sequence[str] | None = None) -> list[SecurityMasterRecord]:
        if symbols is None:
            return sorted(self._records.values(), key=lambda r: (r.exchange, r.symbol))
        requested = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
        return [
            record
            for symbol, record in sorted(self._records.items())
            if symbol in requested
        ]


def build_minimal_security_master(symbols: Sequence[str], exchange: str = "NSE") -> list[SecurityMasterRecord]:
    """Create explicit placeholder records when no richer security master exists."""
    return [
        SecurityMasterRecord(
            symbol=symbol.strip().upper(),
            exchange=exchange,
            company_name=symbol.strip().upper(),
            source="runtime_universe",
        )
        for symbol in symbols
        if symbol.strip()
    ]
